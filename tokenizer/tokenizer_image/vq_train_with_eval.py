# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch
import torch.nn.functional as F
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms

import os
import time
import argparse
from glob import glob
from copy import deepcopy
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
import itertools
from cleanfid import fid

import sys
sys.path.append(os.getcwd())
from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.augmentation import random_crop_arr, center_crop_arr
from dataset.build import build_dataset
# from tokenizer.tokenizer_image.vq_model import VQ_models
from tokenizer.tokenizer_image.efficient_vq_model import VQ_models
from tokenizer.tokenizer_image.vq_loss import VQLoss

import warnings
warnings.filterwarnings('ignore')


def build_train_dataloader(args, rank):
    args.data_path = args.train_data_path
    train_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    train_dataset = build_dataset(args, transform=train_transform)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    return train_dataset, train_loader, train_sampler


def build_valid_dataloader(args, rank):
    args.data_path = args.valid_data_path
    valid_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    valid_dataset = build_dataset(args, transform=valid_transform)
    valid_sampler = DistributedSampler(
        valid_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=valid_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    return valid_loader, valid_sampler


def create_npz_from_sample_folder(sample_dir, num=50000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in range(num):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def validation(logger, vq_model, sample_folder_dir, num_fid_samples, device, rank, args):
    vq_model.eval()
    model = vq_model.module if isinstance(vq_model, DDP) else vq_model

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    valid_loader, _ = build_valid_dataloader(args, rank)

    psnr_val_rgb = []
    ssim_val_rgb = []
    total = 0
    for x, _ in valid_loader:
        if args.image_size_eval != args.image_size:
            rgb_gts = F.interpolate(x, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
        else:
            rgb_gts = x
        rgb_gts = (rgb_gts.permute(0, 2, 3, 1).to("cpu").numpy() + 1.0) / 2.0 # rgb_gt value is between [0, 1]
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            latent, _, [_, _, indices] = model.encode(x)
            samples = model.decode_code(indices, latent.shape) # output value is between [-1, 1]
            if args.image_size_eval != args.image_size:
                samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, (sample, rgb_gt) in enumerate(zip(samples, rgb_gts)):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            # metric
            rgb_restored = sample.astype(np.float32) / 255. # rgb_restored value is between [0, 1]
            psnr = psnr_loss(rgb_restored, rgb_gt)
            ssim = ssim_loss(rgb_restored, rgb_gt, multichannel=True, data_range=2.0, channel_axis=-1)
            psnr_val_rgb.append(psnr)
            ssim_val_rgb.append(ssim)

        total += args.global_batch_size

    # ------------------------------------
    #       Summary
    # ------------------------------------
    # Make sure all processes have finished saving their samples
    dist.barrier()
    world_size = dist.get_world_size()
    gather_psnr_val = [None for _ in range(world_size)]
    gather_ssim_val = [None for _ in range(world_size)]
    dist.all_gather_object(gather_psnr_val, psnr_val_rgb)
    dist.all_gather_object(gather_ssim_val, ssim_val_rgb)

    if rank == 0:
        gather_psnr_val = list(itertools.chain(*gather_psnr_val))
        gather_ssim_val = list(itertools.chain(*gather_ssim_val))        
        psnr_val_rgb = sum(gather_psnr_val) / len(gather_psnr_val)
        ssim_val_rgb = sum(gather_ssim_val) / len(gather_ssim_val)
        fid_val = fid.compute_fid(sample_folder_dir, dataset_name=args.dataset, mode="clean", dataset_split="custom")
        logger.info("rFID: %f, PSNR: %f, SSIM: %f" % (fid_val, psnr_val_rgb, ssim_val_rgb))

        # result_file = f"{sample_folder_dir}_results.txt"
        # logger.info("writing results to {}".format(result_file))
        # with open(result_file, 'w') as f:
        #     logger.info("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb), file=f)

        # create_npz_from_sample_folder(sample_folder_dir, num_fid_samples)
        # print("Done.")

    dist.barrier()


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.vq_model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
        cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        codebook_heads=args.codebook_heads,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        dropout_p=args.dropout_p,
    )
    logger.info(f"VQ Model Parameters: {sum(p.numel() for p in vq_model.parameters()):,}")
    if args.ema:
        ema = deepcopy(vq_model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"VQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")
    vq_model = vq_model.to(device)

    vq_loss = VQLoss(
        disc_start=args.disc_start, 
        disc_weight=args.disc_weight,
        disc_type=args.disc_type,
        disc_loss=args.disc_loss,
        gen_adv_loss=args.gen_loss,
        image_size=args.image_size,
        perceptual_weight=args.perceptual_weight,
        reconstruction_weight=args.reconstruction_weight,
        reconstruction_loss=args.reconstruction_loss,
        codebook_weight=args.codebook_weight,  
    ).to(device)
    logger.info(f"Discriminator Parameters: {sum(p.numel() for p in vq_loss.discriminator.parameters()):,}")

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Setup optimizer
    optimizer = torch.optim.Adam(vq_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_disc = torch.optim.Adam(vq_loss.discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Setup data:
    train_dataset, train_loader, train_sampler = build_train_dataloader(args, rank)
    logger.info(f"Dataset contains {len(train_dataset):,} images ({args.train_data_path})")

    # Prepare models for training:
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"])
        if args.ema:
            ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        vq_loss.discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        if not args.finetune:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vq_ckpt.split('/')[-1].split('.')[0])
            start_epoch = int(train_steps / int(len(train_dataset) / args.global_batch_size))
            train_steps = int(start_epoch * int(len(train_dataset) / args.global_batch_size))
        else:
            train_steps = 0
            start_epoch = 0           
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, vq_model, decay=0)  # Ensure EMA is initialized with synced weights
    
    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        vq_model = torch.compile(vq_model) # requires PyTorch 2.0        
    
    vq_model = DDP(vq_model.to(device), device_ids=[args.gpu])
    vq_model.train()
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode
    vq_loss = DDP(vq_loss.to(device), device_ids=[args.gpu])
    vq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    if args.dataset == 'imagenet':
        num_fid_samples = 50000
    elif args.dataset == 'coco':
        num_fid_samples = 5000
    else:
        raise Exception("please check dataset")

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    # vq_model.module.quantize.enable = False

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        vq_model.train()
        train_sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in train_loader:
            # if train_steps + 1 == args.disc_start:
            #     vq_model.module.quantize.enable = True
            imgs = x.to(device, non_blocking=True)

            # generator training
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                recons_imgs, codebook_loss = vq_model(imgs)
                loss_gen = vq_loss(codebook_loss, imgs, recons_imgs, optimizer_idx=0, global_step=train_steps+1, 
                                   last_layer=vq_model.module.decoder.last_layer, 
                                   logger=logger, log_every=args.log_every)
            scaler.scale(loss_gen).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vq_model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if args.ema:
                update_ema(ema, vq_model.module._orig_mod if args.compile else vq_model.module)

            # discriminator training            
            optimizer_disc.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):
                loss_disc = vq_loss(codebook_loss, imgs, recons_imgs, optimizer_idx=1, global_step=train_steps+1,
                                    logger=logger, log_every=args.log_every)
            scaler_disc.scale(loss_disc).backward()
            if args.max_grad_norm != 0.0:
                scaler_disc.unscale_(optimizer_disc)
                torch.nn.utils.clip_grad_norm_(vq_loss.module.discriminator.parameters(), args.max_grad_norm)
            scaler_disc.step(optimizer_disc)
            scaler_disc.update()

            # # Log loss values:
            running_loss += loss_gen.item() + loss_disc.item()
            
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if args.compile:
                        model_weight = vq_model.module._orig_mod.state_dict()
                    else:
                        model_weight = vq_model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "discriminator": vq_loss.module.discriminator.state_dict(),
                        "optimizer_disc": optimizer_disc.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, cloud_checkpoint_path)
                    logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                dist.barrier()

        folder_name = (f"{args.vq_model}-{args.dataset}-size-{args.image_size}-size-{args.image_size_eval}"
                    f"-codebook-size-{args.codebook_size}-dim-{args.codebook_embed_dim}-seed-{args.global_seed}")
        sample_folder_dir = f"{args.sample_dir}/{folder_name}"
        validation(logger, vq_model, sample_folder_dir, num_fid_samples, device, rank, args)

    vq_model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group('Dataset parameters')
    # parser.add_argument("--data-path", type=str, required=True)
    group.add_argument("--train-data-path", type=str, required=True)
    group.add_argument("--valid-data-path", type=str, required=True)
    group.add_argument("--data-face-path", type=str, default=None, help="face datasets to improve vq model")
    group.add_argument("--dataset", type=str, default='imagenet')
    group.add_argument("--image-size", type=int, choices=[128, 256, 512], default=256)
    group.add_argument("--image-size-eval", type=int, choices=[128, 256, 512], default=256)

    group = parser.add_argument_group('Model parameters')
    group.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    group.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    group.add_argument("--finetune", action='store_true', help="finetune a pre-trained vq model")
    group.add_argument("--ema", action='store_true', help="whether using ema training")
    group.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    group.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    group.add_argument("--codebook-l2-norm", action='store_true', default=True, help="l2 norm codebook")
    group.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    group.add_argument("--codebook-heads", type=int, default=1, help="codebook heads for vector quantization")

    group = parser.add_argument_group('Loss parameters')
    group.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    group.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    group.add_argument("--reconstruction-weight", type=float, default=1.0, help="reconstruction loss weight of image pixel")
    group.add_argument("--reconstruction-loss", type=str, default='l2', help="reconstruction loss type of image pixel")
    group.add_argument("--perceptual-weight", type=float, default=1.0, help="perceptual loss weight of LPIPS")
    group.add_argument("--disc-weight", type=float, default=0.5, help="discriminator loss weight for gan training")
    group.add_argument("--disc-start", type=int, default=20000, help="iteration to start discriminator training and loss")
    group.add_argument("--disc-type", type=str, choices=['patchgan', 'stylegan'], default='patchgan', help="discriminator type")
    group.add_argument("--disc-loss", type=str, choices=['hinge', 'vanilla', 'non-saturating'], default='hinge', help="discriminator loss")
    group.add_argument("--gen-loss", type=str, choices=['hinge', 'non-saturating'], default='hinge', help="generator loss for gan training")

    group = parser.add_argument_group('Optimizer parameters')
    group.add_argument("--epochs", type=int, default=40)
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    group.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    group.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    group.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    group.add_argument("--global-batch-size", type=int, default=128)

    group = parser.add_argument_group('Miscellaneous parameters')
    parser.add_argument("--cloud-save-path", type=str, required=True, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_image")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    # It is weird to use gradient accumulation with GAN training. Odd gradient coupling.
    # parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--sample-dir", type=str, default="reconstructions")
    args = parser.parse_args()
    main(args)
