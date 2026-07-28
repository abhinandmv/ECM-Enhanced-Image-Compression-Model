import os, tempfile
from PIL import Image
from agents.Scale_Hyperprior import CompressionAgent


class PNGAgent(CompressionAgent):
    name = "PNG"

    def compress(self, image):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(tmp.name, "PNG")
        return tmp.name

    def decompress(self, path):
        return Image.open(path).convert("RGB")

    def compressed_size_kb(self, path):
        return os.path.getsize(path) / 1024
