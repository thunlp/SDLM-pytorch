import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.idx2freq = []
        self.threshold = -1

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.idx2freq.append(0)
            self.word2idx[word] = len(self.idx2word) - 1
        self.idx2freq[self.word2idx[word]] += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def freq_le(self, k):
        tot = 0
        for idx in range(len(self.idx2word)):
            if self.idx2freq[idx] < k:
                tot += 1
        return tot

    def freq_ge(self, k):
        tot = 0
        for idx in range(len(self.idx2word)):
            if self.idx2freq[idx] >= k:
                tot += 1
        return tot

    def set_threshold(self, k):
        self.threshold = k

    def valid(self, word):
        if word not in self.word2idx:
            raise ValueError("word don't occur in vocabulary")
        return self.idx2freq[self.word2idx[word]] >= self.threshold


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


class SememeCorpus(object):
    def __init__(self, path):
        self.dictionary = SememeDictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids