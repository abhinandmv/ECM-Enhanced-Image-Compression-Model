import os
import csv
import math
import random
import argparse

import torch
from tqdm import tqdm
from pytorch_msssim import ms_ssim
from compressai.models import ScaleHyperprior

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


def configure_optimizers(model, lr, aux_lr):
    """
    Main optimizer: all params except entropy bottleneck quantiles
    Aux optimizer : only entropy bottleneck quantiles
    """
    params_dict = dict(model.named_parameters())

    main_parameters = [
        p for n, p in params_dict.items()
        if p.requires_grad and "quantiles" not in n
    ]

    aux_parameters = [
        p for n, p in params_dict.items()
        if p.requires_grad and "quantiles" in n
    ]

    optimizer = torch.optim.Adam(main_parameters, lr=lr)
    aux_optimizer = torch.optim.Adam(aux_parameters, lr=aux_lr)

    return optimizer, aux_optimizer


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_bpp = 0.0
    total_psnr = 0.0
    total_msssim = 0.0
    n = 0

    for images in loader:
        images = images.to(device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            output = model(images)
            losses = criterion(output, images)

        bpp = losses["bpp_loss"]
        x_hat = output["x_hat"].clamp(0, 1)

        # PSNR
        mse = torch.mean((x_hat - images) ** 2).item()
        if mse > 0:
            psnr = 10 * math.log10(1.0 / mse)
        else:
            psnr = float("inf")

        # MS-SSIM
        msssim_val = ms_ssim(
            x_hat.float(),
            images.float(),
            data_range=1.0
        ).item()

        total_bpp += bpp.item()
        total_psnr += psnr
        total_msssim += msssim_val
        n += 1

    return {
        "bpp": total_bpp / n,
        "psnr": total_psnr / n,
        "msssim": total_msssim / n,
    }


# ============================================================
# Train One Epoch
# ============================================================

def train_one_epoch(
    model,
    criterion,
    optimizer,
    aux_optimizer,
    train_loader,
    device,
    scaler,
    clip_max_norm=1.0
):
    model.train()

    total_loss = 0.0
    total_bpp = 0.0
    total_aux = 0.0

    pbar = tqdm(train_loader, desc="Training", leave=False)

    for images in pbar:
        images = images.to(device)

        # -------------------------
        # Main network update
        # -------------------------
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            output = model(images)
            losses = criterion(output, images)

        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        scaler.step(optimizer)
        scaler.update()

        # -------------------------
        # Auxiliary entropy update
        # -------------------------
        aux_optimizer.zero_grad(set_to_none=True)
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        total_loss += losses["loss"].item()
        total_bpp += losses["bpp_loss"].item()
        total_aux += aux_loss.item()

        pbar.set_postfix({
            "loss": f"{losses['loss'].item():.4f}",
            "bpp": f"{losses['bpp_loss'].item():.4f}",
            "aux": f"{aux_loss.item():.4f}",
        })

    n = len(train_loader)

    return {
        "loss": total_loss / n,
        "bpp": total_bpp / n,
        "aux_loss": total_aux / n,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aux_lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_scale_hyperprior")
    parser.add_argument("--log_dir", type=str, default="./logs_scale_hyperprior")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--msssim", action="store_true")
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--M", type=int, default=192)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # -------------------------
    # Dataloaders
    # -------------------------
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

    # -------------------------
    # Model
    # -------------------------
    model = ScaleHyperprior(N=args.N, M=args.M).to(device)

    # -------------------------
    # Loss
    # -------------------------
    if args.msssim:
        criterion = MSSSIMRateDistortionLoss(args.lmbda)
    else:
        criterion = RateDistortionLoss(args.lmbda)

    # -------------------------
    # Optimizers
    # -------------------------
    optimizer, aux_optimizer = configure_optimizers(
        model,
        lr=args.lr,
        aux_lr=args.aux_lr,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # -------------------------
    # Resume
    # -------------------------
    start_epoch = 1
    best_psnr = -1.0

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "aux_optimizer" in checkpoint:
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_psnr = checkpoint.get("psnr", -1.0)
        print("Resumed from:", args.resume)

    # -------------------------
    # Logging
    # -------------------------
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, "training_log.csv")

    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_bpp",
                "train_aux_loss",
                "val_bpp",
                "val_psnr",
                "val_msssim"
            ])

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            aux_optimizer=aux_optimizer,
            train_loader=train_loader,
            device=device,
            scaler=scaler,
            clip_max_norm=1.0,
        )

        model.update(force=True)

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        val_bpp = val_metrics["bpp"]
        val_psnr = val_metrics["psnr"]
        val_msssim = val_metrics["msssim"]

        print(
            f"Train Loss: {train_stats['loss']:.4f} | "
            f"Train BPP: {train_stats['bpp']:.4f} | "
            f"Aux Loss: {train_stats['aux_loss']:.4f}"
        )
        print(
            f"Val BPP: {val_bpp:.4f} | "
            f"PSNR: {val_psnr:.2f} | "
            f"MS-SSIM: {val_msssim:.4f}"
        )

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_stats["loss"],
                train_stats["bpp"],
                train_stats["aux_loss"],
                val_bpp,
                val_psnr,
                val_msssim
            ])

        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr

        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "aux_optimizer": aux_optimizer.state_dict(),
            "psnr": val_psnr,
        }, is_best, args.checkpoint_dir)

    print("Training Complete.")


if __name__ == "__main__":
    main()