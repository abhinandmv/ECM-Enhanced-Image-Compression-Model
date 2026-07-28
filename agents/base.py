class CompressionAgent:
    name = "base"

    def compress(self, image):
        raise NotImplementedError

    def decompress(self, compressed):
        raise NotImplementedError

    def compressed_size_kb(self, compressed):
        raise NotImplementedError
