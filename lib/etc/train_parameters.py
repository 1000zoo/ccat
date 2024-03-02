

class CustomConfig:
    def __init__(self):
        self.batch_size = 256
        self.epochs = 100

        self.sequence_length = 128
        self.d_k = 256
        self.d_v = 256
        self.n_heads = 12
        self.ff_dim = 256
        self.interval = "minute1"
