from datasets import load_dataset

class FineWeb():
    def __init__(self, tokeniser, context_size, batch_size, size, device):
        self.data = load_dataset("HuggingFaceFW/fineweb-edu", name='sample-10BT', trust_remote_code=True, split='train', streaming=True)
        self.tokeniser = tokeniser
        self.context_size = context_size
        self.batch_size = batch_size
        self.size = size
        self.device = device

    def __len__(self):
        return self.size 

    def __shuffle__(self):
        self.data = self.data.shuffle(seed=411, buffer_size=10_000)

    def __nextbatch__(self):
        text = next(iter(self.data))['text']
        tokens = self.tokeniser.encode(text)

        x = tokens[:, :-1]
        y = tokens[:, 1:]

        B, T = x.size()
        batch_size = min(self.batch_size, T//self.context_size)

        if batch_size > 0:
            x = x[:, :batch_size*self.context_size].view(batch_size, self.context_size)
            y = y[:, :batch_size*self.context_size].view(batch_size, self.context_size)

        x, y = x.to(self.device), y.to(self.device)
        return x, y  