import os
import csv
import tempfile
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

DATASET_NAME = "kodak"
DATASET_PATH = "kodak"
RESULTS_DIR = "results"
CSV_PATH = f"{RESULTS_DIR}/{DATASET_NAME}_webp_metrics.csv"
IMAGE_SIZE = 256
QUALITY_LIST = [75]

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_metrics(orig_img, recon_img):
    orig_np = np.array(orig_img)
    recon_np = np.array(recon_img)
    psnr = peak_signal_noise_ratio(orig_np, recon_np, data_range=255)
    ssim = structural_similarity(orig_np, recon_np, channel_axis=2, data_range=255)
    return psnr, ssim

def compute_bpp(size_kb):
    return (size_kb * 1024 * 8) / (IMAGE_SIZE * IMAGE_SIZE)

def compress_webp(img, quality):
    tmp = tempfile.NamedTemporaryFile(suffix=".webp", delete=False)
    tmp.close()
    img.save(tmp.name, "WEBP", quality=quality, method=6)
    return tmp.name

def decompress_webp(path):
    return Image.open(path).convert("RGB")

def compressed_size_kb(path):
    return os.path.getsize(path) / 1024.0

image_files = sorted([
    f for f in os.listdir(DATASET_PATH)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
])

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Dataset", "Agent", "Quality", "PSNR", "SSIM", "BPP"])

    for img_name in image_files:
        img_path = os.path.join(DATASET_PATH, img_name)
        img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

        for q in QUALITY_LIST:
            webp_path = compress_webp(img, q)
            try:
                recon = decompress_webp(webp_path)
                size_kb = compressed_size_kb(webp_path)
                bpp = compute_bpp(size_kb)
                psnr, ssim = compute_metrics(img, recon)

                writer.writerow([
                    DATASET_NAME,
                    f"WebP_Q{q}",
                    q,
                    psnr,
                    ssim,
                    bpp
                ])

                print(f"{img_name} | q={q} | PSNR={psnr:.3f} | SSIM={ssim:.4f} | BPP={bpp:.4f}")
            finally:
                if os.path.exists(webp_path):
                    os.remove(webp_path)

print(f"\nSaved results to: {CSV_PATH}")