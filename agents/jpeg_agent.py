from PIL import Image
import io
from agents.Scale_Hyperprior import CompressionAgent

class JPEGAgent(CompressionAgent):
    def __init__(self, quality=75):
        self.quality = quality
        self.name = f"JPEG_Q{quality}"

    def compress(self, img):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        return buffer

    def decompress(self, buffer):
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def compressed_size_kb(self, buffer):
        return len(buffer.getvalue()) / 1024
