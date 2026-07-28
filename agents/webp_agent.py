import os, tempfile
from PIL import Image
from agents.Scale_Hyperprior import CompressionAgent


class WebPAgent(CompressionAgent):
    name = "WebP"

    def compress(self, image):
        tmp = tempfile.NamedTemporaryFile(suffix=".webp", delete=False)
        image.save(tmp.name, "WEBP", quality=50)
        return tmp.name

    def decompress(self, path):
        return Image.open(path).convert("RGB")

    def compressed_size_kb(self, path):
        return os.path.getsize(path) / 1024
