import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0,
                 tie_weights=False, nsememes=-100, nsenses=-100, nbasis=5, word_idx_s=None, use_cuda=True):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(0.25)
        self.nlhid = ninp
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (self.nlhid if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        print(self.rnns)
        self.nmid = 200
      
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.sense_bias = Parameter(torch.FloatTensor(1, nsenses))
        self.word_idx_s = Variable(torch.LongTensor(word_idx_s))
        self.dropout_mask = Variable(torch.ones(1, 1, 1, ninp))
        self.multi_trans = Parameter(torch.FloatTensor(nbasis, 1, self.nlhid, ninp))
        self.multi_weight = Parameter(torch.FloatTensor(nbasis, 1, nsememes))
        
        #self.coeff_weight_l = Parameter(torch.FloatTensor(nbasis, self.nlhid, self.nmid))
        #self.coeff_weight_r = Parameter(torch.FloatTensor(nbasis, self.nmid, nsememes))
        if use_cuda:
            self.word_idx_s = self.word_idx_s.cuda()
            self.dropout_mask = self.dropout_mask.cuda()
        
        self.gate_embed = Parameter(torch.FloatTensor(self.nlhid, nsememes))
        #self.gate_embed_l = Parameter(torch.FloatTensor(self.nlhid, self.nmid))
        #self.gate_embed_r = Parameter(torch.FloatTensor(self.nmid, nsememes))
        #self.gdecoder = nn.Linear(self.nlhid, nsememes)
        self.softmax0 = nn.Softmax(0)
        self.softmax = nn.Softmax(1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.use_cuda = use_cuda
        self.nsememe = nsememes
        self.nsenses = nsenses
        self.nbasis = nbasis

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.ntoken = ntoken
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.use_cuda = use_cuda
        self.tie_weights = tie_weights
        self.init_weights()

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.10
        self.encoder.weight.data.uniform_(-initrange, initrange)
        #self.gdecoder.bias.data.fill_(0)
        #self.gdecoder.weight.data.uniform_(-initrange, initrange)
        #self.gate_embed_l.data.uniform_(-initrange, initrange)
        self.gate_embed.data.uniform_(-initrange, initrange)
        #self.coeff_weight_l.data.uniform_(-initrange, initrange)
        #self.coeff_weight_r.data.uniform_(-initrange, initrange)
        self.sense_bias.data.fill_(0)
        #self.basis_bias.data.fill_(0)
        self.multi_trans.data.uniform_(-initrange * 0.5, initrange * 0.5)
        self.multi_weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, mymm1, mymm2, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden
        output = self.lockdrop(raw_output, 0.35)
        outputs.append(output)
        
        #gate_embed = torch.matmul(self.gate_embed_l, self.gate_embed_r)
        #coeff_weight = torch.matmul(self.coeff_weight_l, self.coeff_weight_r)

        multi_output = output.view(1, output.size(0), output.size(1), output.size(2))
        multi_output = multi_output.expand(self.nbasis, -1, -1, -1)
        multi_transed = torch.matmul(multi_output, self.multi_trans.expand(-1, output.size(0), -1, -1)) #+ self.basis_bias
        multi_transed = self.tanh(multi_transed)
        dropout_mask = self.drop3(self.dropout_mask.expand(-1, -1, output.size(1), -1))
        multi_transed = torch.mul(multi_transed, dropout_mask)
        multi_transed = multi_transed.view(self.nbasis, output.size(0) * output.size(1), self.ninp)
        multi_context = torch.index_select(self.encoder.weight, 0, self.word_idx_s).t()
        multi_logits = torch.matmul(multi_transed, multi_context)

        output = output.view(output.size(0) * output.size(1), output.size(2))
        batch_size = output.size(0)
        gate = self.sigmoid(torch.matmul(output, self.gate_embed))

        multi_gate = gate.view(1, gate.size(0), gate.size(1))
        multi_gate = multi_gate.expand(self.nbasis, -1, -1)
        multi_coeff = torch.mul(multi_gate, self.softmax0(self.multi_weight))#'''torch.matmul(output.view(1, output.size(0), output.size(1)).expand(self.nbasis, -1, -1), coeff_weight) +''' self.multi_weight))
        flat_multi_coeff = multi_coeff.view(self.nbasis * batch_size, self.nsememe)
        flat_multi_coeff = mymm1(flat_multi_coeff).contiguous()
        multi_coeff = flat_multi_coeff.view(self.nbasis, batch_size, self.nsenses)

        #bias_coeff = mymm3(gate)
        logits = torch.sum(torch.mul(multi_logits, multi_coeff), 0) + self.sense_bias#torch.mul(bias_coeff, self.sense_bias.expand(batch_size, -1))
        sense_prob = self.softmax(logits)
        word_prob = mymm2(sense_prob)
        #word_prob = self.softmax(torch.matmul(output, torch.mul(self.encoder.weight, self.drop2(self.dropout_mask)).t()) + self.sense_bias)
        output_prob = word_prob
        #s_output_prob = gate

        if return_h:
            return output_prob, hidden, raw_outputs, outputs
        return output_prob, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.nlhid if self.tie_weights else self.nhid)).zero_()),
                    Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.nlhid if self.tie_weights else self.nhid)).zero_()))
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
