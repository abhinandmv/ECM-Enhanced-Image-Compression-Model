import torch
from diffusers import AutoencoderKL
from agents.Scale_Hyperprior import CompressionAgent

class SD_Agent(CompressionAgent):
    def __init__(self, quantize=False, device="cpu"):
        self.quantize = quantize
        self.device = device
        self.name = "SD_INT8" if quantize else "SD_FP16"

        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse"
        ).to(device).eval()

    def compress(self, tensor):
        with torch.no_grad():
            latents = self.vae.encode(tensor).latent_dist.sample()
            latents = latents * 0.18215

            if self.quantize:
                latents = (latents * 127).clamp(-128, 127).to(torch.int8)

            return latents

    def decompress(self, latents):
        with torch.no_grad():
            if latents.dtype == torch.int8:
                latents = latents.float() / 127

            recon = self.vae.decode(latents / 0.18215).sample
            return recon

    def compressed_size_kb(self, latents):
        return latents.numel() * latents.element_size() / 1024
