import torch
import torch.nn as nn
from torch.autograd import Variable
import hownet_utils
from torch.nn.parameter import Parameter


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False,
                 nsememes=-100, nsenses=-100, nbasis=5, word_idx_s=None, use_cuda=True):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(0.2 + 0.5 * dropout)
        self.drop3 = nn.Dropout(0.5 * dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        self.sense_bias = Parameter(torch.FloatTensor(1, nsenses))
        self.word_idx_s = Variable(torch.LongTensor(word_idx_s))
        self.dropout_mask = Variable(torch.ones(1, ninp))
        self.spect_weight = Parameter(torch.FloatTensor(nsememes, nhid))
        self.multi_trans = Parameter(torch.FloatTensor(nbasis, nhid, nhid))
        self.multi_weight = Parameter(torch.FloatTensor(nbasis, 1, nsememes))
        if use_cuda:
            self.word_idx_s = self.word_idx_s.cuda()
            self.dropout_mask = self.dropout_mask.cuda()

        self.gdecoder = nn.Linear(nhid, nsememes)
        self.softmax0 = nn.Softmax(0)
        self.softmax = nn.Softmax(1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.use_cuda = use_cuda
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.ninp = ninp
        self.nlayers = nlayers
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
        self.multi_trans.data.uniform_(-initrange * 0.5, initrange * 0.5)
        self.multi_weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, mymm1, mymm2, sense=False):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop2(output)

        output = output.view(output.size(0) * output.size(1), output.size(2))
        batch_size = output.size(0)
        gate = self.sigmoid(self.gdecoder(output))
        multi_output = output.view(1, output.size(0), output.size(1))
        multi_output = multi_output.expand(self.nbasis, -1, -1)
        multi_transed = torch.matmul(multi_output, self.multi_trans)
        dropout_mask = self.drop3(self.dropout_mask)
        dropout_weight = torch.mul(dropout_mask.expand(self.ntoken, -1), self.encoder.weight)
        multi_context = torch.index_select(dropout_weight, 0, self.word_idx_s).t()
        multi_logits = torch.matmul(multi_transed, multi_context)

        multi_gate = gate.view(1, gate.size(0), gate.size(1))
        multi_gate = multi_gate.expand(self.nbasis, -1, -1)
        multi_coeff = torch.mul(multi_gate, self.softmax0(self.multi_weight))
        flat_multi_coeff = multi_coeff.view(self.nbasis * batch_size, self.nsememe)
        flat_multi_coeff = mymm1(flat_multi_coeff).contiguous()
        multi_coeff = flat_multi_coeff.view(self.nbasis, batch_size, self.nsenses)

        logits = torch.sum(torch.mul(multi_logits, multi_coeff), 0) + self.sense_bias
        sense_prob = self.softmax(logits)
        word_prob = mymm2(sense_prob)

        output_prob = word_prob
        s_output_prob = gate

        if sense:
            return output_prob, s_output_prob, hidden, sense_prob
        else:
            return output_prob, s_output_prob, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

