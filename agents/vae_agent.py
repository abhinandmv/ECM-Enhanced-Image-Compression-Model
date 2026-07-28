import torch
import torch.nn as nn
from agents.Scale_Hyperprior import CompressionAgent

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

class VAEAgent(CompressionAgent):
    name = "VAE"

    def __init__(self, device):
        self.device = device
        self.model = VAE().to(device).eval()

    def compress(self, tensor):
        with torch.no_grad():
            return self.model.enc(tensor)

    def decompress(self, latent):
        with torch.no_grad():
            return self.model.dec(latent)

    def compressed_size_kb(self, latent):
        return latent.numel() * latent.element_size() / 1024
