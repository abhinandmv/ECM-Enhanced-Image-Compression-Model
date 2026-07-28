import torch
from PIL import Image
from torchvision import transforms
from compressai.zoo import bmshj2018_hyperprior
from agents.base import CompressionAgent


class ScaleHyperpriorAgent(CompressionAgent):
    def __init__(self, quality=3, device="cpu"):
        self.quality = quality
        self.name = f"ScaleHyperprior_q{quality}"

        # CompressAI works reliably on CPU/CUDA
        if str(device) == "mps":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = bmshj2018_hyperprior(
            quality=quality,
            metric="mse",
            pretrained=True
        ).to(self.device)

        self.model.eval()
        self.model.update()

        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def compress(self, image):
        """
        image: torch tensor [1, 3, H, W] or PIL image
        returns: dict with encoded strings + shape
        """
        if isinstance(image, Image.Image):
            x = self.to_tensor(image).unsqueeze(0).to(self.device)
        else:
            x = image.to(self.device)
            if x.dim() == 3:
                x = x.unsqueeze(0)

        with torch.no_grad():
            out = self.model.compress(x)

        return {
            "strings": out["strings"],
            "shape": out["shape"]
        }

    def decompress(self, compressed):
        with torch.no_grad():
            out = self.model.decompress(
                compressed["strings"],
                compressed["shape"]
            )
            x_hat = out["x_hat"].clamp(0, 1).cpu().squeeze(0)

        return self.to_pil(x_hat)

    def compressed_size_kb(self, compressed):
        total_bytes = 0
        for stream_group in compressed["strings"]:
            for s in stream_group:
                total_bytes += len(s)
        return total_bytes / 1024.0