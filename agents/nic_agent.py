import torch
from compressai.zoo import bmshj2018_factorized
from agents.Scale_Hyperprior import CompressionAgent

class NICAgent(CompressionAgent):
    def __init__(self, quality=3):
        self.quality = quality
        self.name = f"NIC_Q{quality}"

        self.device = torch.device("cpu")
        self.model = bmshj2018_factorized(
            quality=quality,
            pretrained=True
        ).to(self.device).eval()

    def compress(self, tensor):
        with torch.no_grad():
            tensor = tensor.to(self.device)
            return self.model.compress(tensor)

    def decompress(self, strings):
        with torch.no_grad():
            return self.model.decompress(
                strings["strings"],
                strings["shape"]
            )["x_hat"]

    def compressed_size_kb(self, strings):
        bits = sum(len(s[0]) for s in strings["strings"]) * 8
        return bits / 1024
