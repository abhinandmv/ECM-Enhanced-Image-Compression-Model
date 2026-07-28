
import os
import csv
import time
import math
import random
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.utils as vutils

from model_enhanced import EnhancedCompressionModel
from dataUtils import build_train_dataloader, build_val_dataloader
from losses_eval import RateDistortionLoss, MSSSIMRateDistortionLoss


# ============================================================
# Utilities
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state, is_best, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = os.path.join(checkpoint_dir, f"epoch_{state['epoch']}.pt")
    torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, "best_model.pt"))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_bpp = 0.0
    total_psnr = 0.0
    total_msssim = 0.0
    total_mse = 0.0
    total_lpips = 0.0
    n = 0

    from pytorch_msssim import ms_ssim
    import math
    import lpips as lpips_module

    lpips_fn = lpips_module.LPIPS(net="alex").to(device)

    for images in loader:
        images = images.to(device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            output = model(images)
            losses = criterion(output, images)

        mse = losses.get("mse_loss", None)
        bpp = losses["bpp_loss"]

        x_hat = output["x_hat"].clamp(0, 1)

        if mse is not None:
            total_mse += mse.item()
            total_psnr += 10 * math.log10(1.0 / mse.item())

        total_bpp += bpp.item()

        msssim_val = ms_ssim(
            x_hat.float(),
            images.float(),
            data_range=1.0
        ).item()

        total_msssim += msssim_val

        # LPIPS — same metric used for every other method in the table
        lp = lpips_fn(x_hat.float(), images.float()).item()
        total_lpips += lp

        n += 1

    return {
        "bpp": total_bpp / n,
        "psnr": total_psnr / n if total_psnr > 0 else 0,
        "msssim": total_msssim / n,
        "lpips": total_lpips / n,
    }


# ============================================================
# Reconstruction Saving
# ============================================================

@torch.no_grad()
def save_reconstruction(model, loader, device, epoch, save_dir):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    images = next(iter(loader))
    images = images.to(device)

    output = model(images)
    x_hat = output["x_hat"].clamp(0, 1)

    vutils.save_image(
        images,
        os.path.join(save_dir, f"epoch_{epoch}_original.png"),
        nrow=4
    )

    vutils.save_image(
        x_hat,
        os.path.join(save_dir, f"epoch_{epoch}_reconstruction.png"),
        nrow=4
    )


# ============================================================
# Training Function
# ============================================================

def train_one_epoch(model, criterion, optimizer, aux_optimizer,
                    train_loader, device, scaler, clip_max_norm=1.0):

    model.train()
    total_loss = 0.0
    total_bpp = 0.0
    total_mse = 0.0

    pbar = tqdm(train_loader, desc="Training", leave=False)

    for images in pbar:

        images = images.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            output = model(images)
            losses = criterion(output, images)

        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        scaler.step(optimizer)
        scaler.update()

        aux_optimizer.zero_grad(set_to_none=True)
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        total_loss += losses["loss"].item()
        total_bpp += losses["bpp_loss"].item()
        if "mse_loss" in losses:
            total_mse += losses["mse_loss"].item()

        pbar.set_postfix({
            "loss": f"{losses['loss'].item():.4f}",
            "bpp": f"{losses['bpp_loss'].item():.3f}"
        })

    n = len(train_loader)

    return {
        "loss": total_loss / n,
        "bpp": total_bpp / n,
        "mse": total_mse / n if total_mse > 0 else None
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aux_lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--msssim", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_loader = build_train_dataloader(
        args.train_path,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
    )

    val_loader = build_val_dataloader(
        args.val_path,
        batch_size=1,
        num_workers=args.num_workers,
    )

    model = EnhancedCompressionModel().to(device)

    if args.msssim:
        criterion = MSSSIMRateDistortionLoss(args.lmbda)
    else:
        criterion = RateDistortionLoss(args.lmbda)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    aux_optimizer = torch.optim.Adam(model.aux_parameters(), lr=args.aux_lr)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 1
    best_psnr = -1.0

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_psnr = checkpoint.get("psnr", -1.0)
        print("Resumed from:", args.resume)

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, "training_log.csv")

    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_bpp", "val_psnr", "val_msssim", "val_lpips"])

    for epoch in range(start_epoch, args.epochs + 1):

        print(f"\nEpoch {epoch}/{args.epochs}")

        train_stats = train_one_epoch(
            model, criterion, optimizer, aux_optimizer,
            train_loader, device, scaler
        )

        model.update(force=True)

        val_metrics = evaluate(model, val_loader, criterion, device)

        val_bpp = val_metrics["bpp"]
        val_psnr = val_metrics["psnr"]
        val_msssim = val_metrics["msssim"]
        val_lpips = val_metrics["lpips"]

        print(f"Train Loss: {train_stats['loss']:.4f}")
        print(f"Val BPP: {val_bpp:.4f} | PSNR: {val_psnr:.2f} | MS-SSIM: {val_msssim:.4f} | LPIPS: {val_lpips:.4f}")

        save_reconstruction(
            model,
            val_loader,
            device,
            epoch,
            os.path.join(args.log_dir, "reconstructions")
        )

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_stats["loss"], val_bpp, val_psnr, val_msssim, val_lpips])

        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr

        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "psnr": val_psnr,
        }, is_best, args.checkpoint_dir)

    print("Training Complete.")


if __name__ == "__main__":
    main()
