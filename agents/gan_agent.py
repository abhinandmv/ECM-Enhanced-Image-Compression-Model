import torch
import torch.nn.functional as F
from agents.Scale_Hyperprior import CompressionAgent

class GANAgent(CompressionAgent):
    name = "GAN"

    def __init__(self, device):
        self.device = device

    def compress(self, tensor):
        with torch.no_grad():
            return F.avg_pool2d(tensor, kernel_size=4)

    def decompress(self, latent):
        with torch.no_grad():
            return F.interpolate(latent, scale_factor=4, mode="bilinear", align_corners=False)

    def compressed_size_kb(self, latent):
        return latent.numel() * latent.element_size() / 1024
