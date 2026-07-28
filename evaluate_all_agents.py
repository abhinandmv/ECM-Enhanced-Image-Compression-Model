import os
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

# ================= DEVICE SELECTION =================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"✅ Using device: {device}")

# ================= IMPORT AGENTS =================
from agents.jpeg_agent import JPEGAgent
from agents.png_agent import PNGAgent
from agents.webp_agent import WebPAgent
from agents.ae_agent import AEAgent
from agents.vae_agent import VAEAgent
from agents.nic_agent import NICAgent
from agents.gan_agent import GANAgent
from agents.sd_agent import SD_Agent
from agents.Scale_Hyperprior import ScaleHyperpriorAgent

# ================= CONFIG =================
DATASET_NAME = "kodak"        # change to "kodak"
DATASET_PATH = "kodak"
RESULTS_DIR = "results"
CSV_PATH = f"{RESULTS_DIR}/{DATASET_NAME}_metrics.csv"

IMAGE_SIZE = 256
NUM_IMAGES = 50    # increase later on GPU PC

os.makedirs(RESULTS_DIR, exist_ok=True)

# ================= TRANSFORMS =================
to_tensor = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()

# ================= METRICS =================
lpips_fn = lpips.LPIPS(net="alex").to(device)

def compute_metrics(orig_img, recon_img):
    orig_np = np.array(orig_img)
    recon_np = np.array(recon_img)

    psnr = peak_signal_noise_ratio(orig_np, recon_np, data_range=255)
    ssim = structural_similarity(orig_np, recon_np, channel_axis=2, data_range=255)

    t1 = to_tensor(orig_img).unsqueeze(0).to(device)
    t2 = to_tensor(recon_img).unsqueeze(0).to(device)

    with torch.no_grad():
        lp = lpips_fn(t1, t2).item()

    return psnr, ssim, lp

def compute_bpp(size_kb):
    return (size_kb * 1024 * 8) / (IMAGE_SIZE * IMAGE_SIZE)

# ================= LOAD AGENTS (RATE CONTROLLED) =================
agents = []

# ---- JPEG rate control ----
for q in [30, 50, 70, 90]:
    agents.append(JPEGAgent(quality=q))

# ---- Traditional ----
agents.append(PNGAgent())
agents.append(WebPAgent())

# ---- Learned ----
agents.append(AEAgent(device))
agents.append(VAEAgent(device))

# ---- NIC rate control (CPU only) ----
for q in [1, 3, 5]:
    agents.append(NICAgent(quality=q))

# ---- Scale Hyperprior baseline ----
for q in [1, 3, 5]:
    agents.append(ScaleHyperpriorAgent(quality=q, device=device))

# ---- Perceptual / Diffusion ----
agents.append(GANAgent(device))
agents.append(SD_Agent(quantize=False, device=device))
agents.append(SD_Agent(quantize=True, device=device))

print("✅ Loaded agents:")
for a in agents:
    print(" -", a.name)

# ================= LOAD DATA =================
image_files = sorted(os.listdir(DATASET_PATH))[:NUM_IMAGES]

# ================= RUN EVALUATION =================
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Dataset", "Agent", "PSNR", "SSIM", "LPIPS", "BPP"])

    for img_name in image_files:
        img_path = os.path.join(DATASET_PATH, img_name)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

        img_tensor = to_tensor(img).unsqueeze(0).to(device)

        for agent in agents:

            # ---------- COMPRESS ----------
            if agent.name.startswith("JPEG") or agent.name in ["PNG", "WebP"]:
                compressed = agent.compress(img)

            elif agent.name.startswith("NIC"):
                compressed = agent.compress(img_tensor.cpu())  # NIC CPU only

            elif agent.name.startswith("ScaleHyperprior"):
                compressed = agent.compress(img_tensor)

            else:
                compressed = agent.compress(img_tensor)

            # ---------- DECOMPRESS ----------
            recon = agent.decompress(compressed)

            if torch.is_tensor(recon):
                recon = recon.squeeze().clamp(0, 1).cpu()
                recon = to_pil(recon)

            # ---------- METRICS ----------
            size_kb = agent.compressed_size_kb(compressed)
            bpp = compute_bpp(size_kb)
            psnr, ssim, lp = compute_metrics(img, recon)

            writer.writerow([
                DATASET_NAME,
                agent.name,
                psnr,
                ssim,
                lp,
                bpp
            ])

print(f"\n✅ Evaluation complete. Results saved to {CSV_PATH}")
