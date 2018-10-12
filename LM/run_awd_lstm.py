import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append("..")
import hownet
import hownet_utils
from torch.autograd import Variable

import data
import awd_lstm_rnn as rnn

from utils import batchify, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)
ntokens = len(corpus.dictionary)

###############################################################################
# Load Hownet Resources
###############################################################################

overall_dict = hownet.SememeDictionary()
overall_dict.summary()


def add_word_overall(source):
    for id in source:
        word = corpus.dictionary.idx2word[id]
        overall_dict.add_word_f(word)


add_word_overall(corpus.train)
add_word_overall(corpus.test)
add_word_overall(corpus.valid)
overall_dict.set_threshold(1)
sememe_word_pair, sememe_idxs, sememe_sense_pair, word_sense = \
    overall_dict.sememe_word_visit(corpus.dictionary.word2idx)
nsememes = max(sememe_word_pair[0]) + 1
nsenses = max(sememe_sense_pair[1]) + 1
npairs = len(sememe_word_pair[0])

print('Total tokens: {}'.format(ntokens))

###############################################################################
# SOU PART
###############################################################################

sense_nsememes = [0] * nsenses
sememe_nsenses = [0] * nsememes
for i in range(npairs):
    sense_nsememes[sememe_sense_pair[1][i]] += 1
    sememe_nsenses[sememe_sense_pair[0][i]] += 1
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

if args.cuda:
    sou_sememe_sense = sou_sememe_sense.cuda()
    sou_sememe_sense_t = sou_sememe_sense_t.cuda()
    sou_sense_word = sou_sense_word.cuda()
    sou_sense_word_t = sou_sense_word_t.cuda()
sou_sparsemm1 = hownet_utils.spmm(sou_sememe_sense, sou_sememe_sense_t)
sou_sparsemm2 = hownet_utils.spmm(sou_sense_word, sou_sense_word_t)


###############################################################################
# Build the model
###############################################################################

model = rnn.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth,
                     args.dropouti, args.dropoute, args.wdrop, args.tied,
                     nsememes=nsememes, use_cuda=args.cuda, nsenses=nsenses, word_idx_s=word_idx_s)
if args.cuda:
    model.cuda()
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
print('Args:', args)
print('Model total parameters:', total_params)

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################


def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden, sou_sparsemm1, sou_sparsemm2)
        batch_size = output.size(0)
        labels = torch.zeros(batch_size, ntokens).scatter_(1, targets.data.view(batch_size, 1), 1)
        if args.cuda:
            labels = labels.cuda()
        total_loss += len(data) * torch.mean(torch.sum(torch.mul(-torch.log(output), Variable(labels)), 1)).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, sou_sparsemm1,
                                                       sou_sparsemm2, return_h=True)

        batch_size = targets.size(0)
        labels = torch.zeros(batch_size, ntokens).scatter_(1, targets.data.view(batch_size, 1), 1)
        if args.cuda:
            labels = labels.cuda()
        cs_e = torch.mul(-torch.log(output + 1e-9 * (1 - Variable(labels))), Variable(labels))

        #print(torch.sum(cs_e, dim=1))
        raw_loss = torch.mean(torch.sum(cs_e, dim=1))
        loss = raw_loss
        # Activiation Regularization
        loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        #print(raw_loss.data[0])
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

try:
    with open(args.save, 'rb') as f:
        model = torch.load(f)
except:
    pass

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        if False:
        #if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss2, math.exp(val_loss2)))
            print('-' * 89)

            if val_loss2 < stored_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)

            if val_loss < stored_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                print('Saving Normal!')
                stored_loss = val_loss
            else:
            #if 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
            #    print('Switching!')
            #    optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                optimizer.param_groups[0]['lr'] /= 2.
            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

