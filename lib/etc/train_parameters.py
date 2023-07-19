

class CustomConfig:
    def __init__(self):
        self.batch_size = 32
        self.epochs = 1

        self.sequence_length = 64
        self.d_k = 256
        self.d_v = 256
        self.n_heads = 16
        self.ff_dim = 256
        self.interval = "minute240"
