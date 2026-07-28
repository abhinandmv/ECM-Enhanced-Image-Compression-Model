import torch
import torch.nn as nn
from agents.Scale_Hyperprior import CompressionAgent

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Sigmoid()
        )

class AEAgent(CompressionAgent):
    name = "AE"

    def __init__(self, device):
        self.device = device
        self.model = AE().to(device).eval()

    def compress(self, tensor):
        with torch.no_grad():
            return self.model.enc(tensor)

    def decompress(self, latent):
        with torch.no_grad():
            return self.model.dec(latent)

    def compressed_size_kb(self, latent):
        return latent.numel() * latent.element_size() / 1024
