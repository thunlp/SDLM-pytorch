import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import data
import hownet
import hownet_utils


class BuildModel(object):
    def __init__(self, rnn_size, use_cuda):
        self.ntoken = 0
        self.nhid = rnn_size
        self.nsememes = 0
        self.nsenses = 0
        self.word_idx_s = None
        self.mymm1 = None
        self.mymm2 = None
        self.use_cuda = use_cuda
        vocab = torch.load('../data/train/light_textsum.vocab.pt')
        vocab = dict(vocab)
        word_idx = vocab['tgt'].itos

        overall_dict = hownet.SememeDictionary()
        overall_dict.summary()
        corpus = data.Corpus(word_idx)

        for word in word_idx:
            overall_dict.add_word_f(word)

        overall_dict.set_threshold(1)
        sememe_word_pair, sememe_idxs, sememe_sense_pair, word_sense = \
            overall_dict.sememe_word_visit(corpus.dictionary.word2idx)
        nsememes = max(sememe_word_pair[0]) + 1
        nsenses = max(sememe_sense_pair[1]) + 1
        ntokens = len(corpus.dictionary)
        npairs = len(sememe_word_pair[0])

        print('Total tokens: {}'.format(ntokens))

        # -----------------------------------------------------------------------------
        # SOU PART
        # -----------------------------------------------------------------------------

        sense_nsememes = [0] * nsenses
        for i in range(npairs):
            sense_nsememes[sememe_sense_pair[1][i]] += 1
        pair_sense_values = []
        for i in range(npairs):
            pair_sense_values.append(1.0 / sense_nsememes[sememe_sense_pair[1][i]])
        sou_sememe_sense = torch.sparse.FloatTensor(torch.LongTensor([sememe_sense_pair[0], sememe_sense_pair[1]]),
                                                    torch.FloatTensor(pair_sense_values),
                                                    torch.Size([nsememes, nsenses]))
        sou_sememe_sense_t = torch.sparse.FloatTensor(torch.LongTensor([sememe_sense_pair[1], sememe_sense_pair[0]]),
                                                      torch.FloatTensor(pair_sense_values),
                                                      torch.Size([nsenses, nsememes]))
        sssp1 = []
        sssp2 = []
        sssp_tot = 0
        d_set = set([])
        word_idx_s = [-1] * nsenses
        for i in range(npairs):
            rec_pair = (sememe_sense_pair[1][i], sememe_word_pair[1][i])
            if rec_pair in d_set:
                continue
            d_set.add(rec_pair)
            sssp_tot += 1
            sssp1.append(sememe_sense_pair[1][i])
            sssp2.append(sememe_word_pair[1][i])
            word_idx_s[sememe_sense_pair[1][i]] = sememe_word_pair[1][i]
        sou_sense_word = torch.sparse.FloatTensor(torch.LongTensor([sssp1, sssp2]),
                                                  torch.FloatTensor([1.0] * sssp_tot),
                                                  torch.Size([nsenses, ntokens]))
        sou_sense_word_t = torch.sparse.FloatTensor(torch.LongTensor([sssp2, sssp1]),
                                                    torch.FloatTensor([1.0] * sssp_tot),
                                                    torch.Size([ntokens, nsenses]))
        if use_cuda:
            sou_sememe_sense = sou_sememe_sense.cuda()
            sou_sememe_sense_t = sou_sememe_sense_t.cuda()
            sou_sense_word = sou_sense_word.cuda()
            sou_sense_word_t = sou_sense_word_t.cuda()
        sou_sparsemm1 = hownet_utils.spmm(sou_sememe_sense, sou_sememe_sense_t)
        sou_sparsemm2 = hownet_utils.spmm(sou_sense_word, sou_sense_word_t)
        self.ntoken = ntokens
        self.nsememes = nsememes
        self.nsenses = nsenses
        self.word_idx_s = word_idx_s
        self.mymm1 = sou_sparsemm1
        self.mymm2 = sou_sparsemm2

    def get_model(self):
        return RNNModel(self.ntoken, self.nhid, self.nsememes, self.nsenses, self.word_idx_s, self.mymm1, self.mymm2,
                        self.use_cuda)


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, nhid, nsememes, nsenses, word_idx_s, mymm1, mymm2, use_cuda,
                  nbasis=3):  # 5
        # ninp: emsize
        # word_idx_s: sense -> word_idx
        super(RNNModel, self).__init__()
        self.encoder = nn.Linear(nhid, ntoken)

        self.sense_bias = Parameter(torch.FloatTensor(1, nsenses))
        self.word_idx_s = Variable(torch.LongTensor(word_idx_s))
        self.spect_weight = Parameter(torch.FloatTensor(nsememes, nhid))
        self.multi_trans = Parameter(torch.FloatTensor(nbasis, nhid, nhid))
        self.multi_weight = Parameter(torch.FloatTensor(nbasis, 1, nsememes))
        self.mymm1 = mymm1
        self.mymm2 = mymm2

        if use_cuda:
            self.word_idx_s = self.word_idx_s.cuda()
        self.gdecoder = nn.Linear(nhid, nsememes)
        self.softmax0 = nn.Softmax(0)
        self.softmax = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()

        self.use_cuda = use_cuda
        self.nhid = nhid
        self.ntoken = ntoken
        self.nsememe = nsememes
        self.nsenses = nsenses
        self.nbasis = nbasis

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.gdecoder.bias.data.fill_(0)
        self.gdecoder.weight.data.uniform_(-initrange, initrange)
        self.sense_bias.data.fill_(0)
        self.multi_trans.data.uniform_(-initrange, initrange)
        self.multi_weight.data.uniform_(-initrange, initrange)

    def forward(self, output):
        # output: seq_len, batch, hidden_size
        output = output.view(output.size(0) * output.size(1), output.size(2))  # output: seq_len * batch, hidden_size
        batch_size = output.size(0)  # new_batch_size = seq_len * batch
        gate = self.sigmoid(self.gdecoder(output))  # gate: batch_size, nsememes
        multi_output = output.view(1, output.size(0), output.size(1))  # output_size(0): seq_len * batch, output_size(1): hidden_size
        multi_output = multi_output.expand(self.nbasis, -1, -1)  # multi_output: nbasis, batch_size, hidden_size
        multi_transed = torch.matmul(multi_output, self.multi_trans)  # multi_trans: nbasis, nhid, nhid
        multi_context = torch.index_select(self.encoder.weight, 0, self.word_idx_s).t()
        multi_logits = torch.matmul(multi_transed, multi_context)  # multi_logits: nbasis, batch_size, nsenses

        multi_gate = gate.view(1, gate.size(0), gate.size(1))
        multi_gate = multi_gate.expand(self.nbasis, -1, -1)  # multi_gate: nbasis, batch_size, nsememes
        multi_coeff = torch.mul(multi_gate, self.softmax0(self.multi_weight))  # multi_weight: nbasis, 1, nsememes
        # multi_coeff: nbasis, batch_size, nsememes
        flat_multi_coeff = multi_coeff.view(self.nbasis * batch_size, self.nsememe)
        # flat_multi_coeff: nbasis * batch_size, nsememes
        flat_multi_coeff = self.mymm1(flat_multi_coeff).contiguous()  # mymm1: nsememes, nsenses
        # flat_multi_coeff: nbasis * batch_size, nsenses
        multi_coeff = flat_multi_coeff.view(self.nbasis, batch_size, self.nsenses)
        # multi_coeff: nbasis, batch_size, nsenses

        logits = torch.sum(torch.mul(multi_logits, multi_coeff), 0) + self.sense_bias  # sense_bias: 1, nsenses
        sense_prob = self.softmax(logits)  # logits: batch_size, nsenses
        word_prob = self.mymm2(sense_prob)  # word_prob: batch_size, nword

        output_prob = word_prob

        output_prob = torch.log(output_prob.clamp(min=1e-8))
        return output_prob