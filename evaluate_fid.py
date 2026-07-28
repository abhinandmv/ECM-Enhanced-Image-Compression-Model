import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch_fidelity import calculate_metrics

# ================= PATH FIX =================
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ================= IMPORT AGENTS =================
from agents.jpeg_agent import JPEGAgent
from agents.png_agent import PNGAgent
from agents.webp_agent import WebPAgent
from agents.ae_agent import AEAgent
from agents.vae_agent import VAEAgent
from agents.gan_agent import GANAgent
from agents.nic_agent import NICAgent
from agents.sd_agent import SD_Agent
from model_enhanced import EnhancedCompressionModel

# ================= CONFIG =================
DATASET_NAME = "kodak"
DATASET_DIR = "kodak"                       # change if needed
FID_ROOT = Path("fid_eval_kodak")
CSV_PATH = "fid_results_kodak.csv"

IMAGE_LIMIT = None                         # None = use all images
RESIZE_TO = 256                            # IMPORTANT: force same size for FID
ENHANCED_CKPT = "checkpoints/best_model.pt"  # change if needed

# ================= DEVICE =================
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"✅ Using device: {DEVICE}")

to_tensor = transforms.ToTensor()


# ================= HELPERS =================
def preprocess_pil(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img = img.resize((RESIZE_TO, RESIZE_TO))
    return img


def pil_to_tensor(img: Image.Image, device=DEVICE) -> torch.Tensor:
    return to_tensor(img).unsqueeze(0).to(device)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().squeeze(0).clamp(0, 1).cpu()
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def list_images(folder: str):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
    files = []
    for ext in exts:
        files.extend(Path(folder).glob(ext))
    files = sorted(files)
    if IMAGE_LIMIT is not None:
        files = files[:IMAGE_LIMIT]
    return files


# ================= ENHANCED AGENT =================
class EnhancedAgent:
    def __init__(self, checkpoint_path, device=DEVICE, N=128, M=192):
        self.name = "Model_Enhanced"
        self.device = device
        self.model = EnhancedCompressionModel(N=N, M=M).to(device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=device)

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.replace("module.", "")] = v

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)

        print(f"✅ Loaded Model_Enhanced from: {checkpoint_path}")
        if len(missing) > 0:
            print(f"⚠ Missing keys: {len(missing)}")
        if len(unexpected) > 0:
            print(f"⚠ Unexpected keys: {len(unexpected)}")

        self.model.eval()

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out["x_hat"].clamp(0, 1)


# ================= UNIVERSAL RECONSTRUCTION =================
@torch.no_grad()
def reconstruct_with_agent(agent, pil_img: Image.Image) -> torch.Tensor:
    # PIL-based codecs
    if isinstance(agent, JPEGAgent):
        compressed = agent.compress(pil_img)
        recon = agent.decompress(compressed)
        return pil_to_tensor(recon, device="cpu")

    if isinstance(agent, PNGAgent):
        compressed = agent.compress(pil_img)
        recon = agent.decompress(compressed)
        try:
            if isinstance(compressed, str) and os.path.exists(compressed):
                os.remove(compressed)
        except Exception:
            pass
        return pil_to_tensor(recon, device="cpu")

    if isinstance(agent, WebPAgent):
        compressed = agent.compress(pil_img)
        recon = agent.decompress(compressed)
        try:
            if isinstance(compressed, str) and os.path.exists(compressed):
                os.remove(compressed)
        except Exception:
            pass
        return pil_to_tensor(recon, device="cpu")

    # NIC
    if isinstance(agent, NICAgent):
        x = pil_to_tensor(pil_img, device="cpu")
        compressed = agent.compress(x)
        recon = agent.decompress(compressed)
        return recon.clamp(0, 1).cpu()

    # Enhanced model
    if isinstance(agent, EnhancedAgent):
        x = pil_to_tensor(pil_img, device=agent.device)
        recon = agent.reconstruct(x)
        return recon.cpu()

    # Tensor-based agents
    x = pil_to_tensor(pil_img, device=agent.device if hasattr(agent, "device") else DEVICE)
    compressed = agent.compress(x)
    recon = agent.decompress(compressed)

    if isinstance(recon, Image.Image):
        return pil_to_tensor(recon, device="cpu")

    return recon.clamp(0, 1).cpu()


# ================= MAIN =================
def main():
    real_dir = FID_ROOT / "real"
    real_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(DATASET_DIR)
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {DATASET_DIR}")

    print(f"📂 Found {len(image_paths)} images in {DATASET_DIR}")

    # Save real images
    print("📥 Saving real images...")
    for i, path in enumerate(tqdm(image_paths)):
        img = preprocess_pil(Image.open(path))
        img.save(real_dir / f"{i:04d}.png")

    # Agents
    agents = []

    for q in [30, 50, 70, 90]:
        agents.append(JPEGAgent(quality=q))

    agents.append(PNGAgent())
    agents.append(WebPAgent())

    agents.append(AEAgent(DEVICE))
    agents.append(VAEAgent(DEVICE))
    agents.append(GANAgent(DEVICE))

    for q in [1, 3, 5]:
        agents.append(NICAgent(quality=q))

    agents.append(SD_Agent(quantize=False, device=DEVICE))
    agents.append(SD_Agent(quantize=True, device=DEVICE))

    agents.append(EnhancedAgent(ENHANCED_CKPT, device=DEVICE))

    print("\n✅ Loaded agents:")
    for a in agents:
        print(" -", a.name)

    results = []

    # FID loop
    for agent in agents:
        fake_dir = FID_ROOT / agent.name
        fake_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🎯 Computing FID for {agent.name}")

        for i, path in enumerate(tqdm(image_paths)):
            img = preprocess_pil(Image.open(path))
            recon = reconstruct_with_agent(agent, img)
            out_img = tensor_to_pil(recon)
            out_img.save(fake_dir / f"{i:04d}.png")

        metrics = calculate_metrics(
            input1=str(real_dir),
            input2=str(fake_dir),
            fid=True,
            cuda=(DEVICE == "cuda"),
            batch_size=1,     # IMPORTANT FIX
            num_workers=0,    # IMPORTANT FIX
            verbose=False,
        )

        fid_value = metrics["frechet_inception_distance"]
        results.append((DATASET_NAME, agent.name, fid_value))
        print(f"✅ {agent.name}: FID = {fid_value:.4f}")

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Agent", "FID"])
        writer.writerows(results)

    print("\n✅ FID evaluation complete")
    print(f"📄 Saved to {CSV_PATH}")


if __name__ == "__main__":
    main()