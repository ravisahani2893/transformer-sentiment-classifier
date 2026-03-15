import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):

    def __init__(self, data, word2idx, seq_len=10):
        """
        data: list of (sentence, label)
        word2idx: dictionary mapping words to indices
        seq_len: fixed input length
        """
        self.sentences = [d[0] for d in data]
        self.labels = [d[1] for d in data]

        self.word2idx = word2idx
        self.seq_len = seq_len

    def encode(self, sentence):
        """
        Convert sentence -> list of token ids
        """
        tokens = sentence.lower().split()

        ids = [self.word2idx.get(w, 0) for w in tokens]

        # pad if sentence is shorter
        if len(ids) < self.seq_len:
            ids += [0] * (self.seq_len - len(ids))

        # truncate if longer
        return ids[:self.seq_len]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):

        x = torch.tensor(self.encode(self.sentences[idx]))
        y = torch.tensor(self.labels[idx])

        return x, y
