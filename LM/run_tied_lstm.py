# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import data
from tied_lstm_rnn import RNNModel
import hownet
import hownet_utils

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/renmin_sample',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=30,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=30,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--lamb', type=float, default=0.9,
                    help='the cross entropy error')
parser.add_argument('--output', type=str, default='normal',
                    help='output mode: normal / sou / highrank')
parser.add_argument('--mode', type=str, default='train',
                    help='the mode: train / sample / pred')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

overall_dict = hownet.SememeDictionary()
overall_dict.summary()
corpus = data.Corpus(args.data)


def add_word_overall(source):
    for id in source:
        word = corpus.dictionary.idx2word[id]
        overall_dict.add_word_f(word)


add_word_overall(corpus.train)
add_word_overall(corpus.test)
add_word_overall(corpus.valid)
overall_dict.set_threshold(1)
sememe_word_pair, sememe_idxs, sememe_sense_pair, _____ = \
    overall_dict.sememe_word_visit(corpus.dictionary.word2idx)
nsememes = max(sememe_word_pair[0]) + 1
nsenses = max(sememe_sense_pair[1]) + 1


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
npairs = len(sememe_word_pair[0])

print('Total tokens: {}'.format(ntokens))

# -----------------------------------------------------------------------------
# SOU PART
# -----------------------------------------------------------------------------

sense_nsememes = [0] * nsenses
sememe_nsenses = [0] * nsememes
for i in range(npairs):
    sense_nsememes[sememe_sense_pair[1][i]] += 1
    sememe_nsenses[sememe_sense_pair[0][i]] += 1
pair_sense_values = []
for i in range(npairs):
    pair_sense_values.append(1.0 / ((sense_nsememes[sememe_sense_pair[1][i]] * sense_nsememes[sememe_sense_pair[1][i]]) ** 0.5))

if True:
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
word_sense = []
for i in range(ntokens):
    word_sense.append([])
for i in range(npairs):
    rec_pair = (sememe_sense_pair[1][i], sememe_word_pair[1][i])
    if rec_pair in d_set:
        continue
    d_set.add(rec_pair)
    sssp_tot += 1
    sssp1.append(sememe_sense_pair[1][i])
    sssp2.append(sememe_word_pair[1][i])
    word_idx_s[sememe_sense_pair[1][i]] = sememe_word_pair[1][i]
    word_sense[sememe_word_pair[1][i]].append(sememe_sense_pair[1][i])

if True:
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


model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied,
                 nsememes=nsememes, use_cuda=args.cuda, nsenses=nsenses, word_idx_s=word_idx_s)

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
logsoftmax = nn.LogSoftmax()

###############################################################################
# Training code
###############################################################################


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    if args.cuda:
        data = data.cuda()
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, _, hidden = model(data, hidden, sou_sparsemm1,
                                  sou_sparsemm2)
        batch_size = output.size(0)
        labels = torch.zeros(batch_size, ntokens).scatter_(1, targets.data.view(batch_size, 1), 1)
        if args.cuda:
            labels = labels.cuda()
        total_loss += len(data) * torch.mean(torch.sum(torch.mul(-torch.log(output), Variable(labels)), 1)).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def get_sememe(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    tokens = source[i+1:i+1+seq_len].view(-1)
    sememe_targets = []
    max_senses = 0
    for i in range(len(tokens)):
        token_id = tokens[i]
        word = corpus.dictionary.idx2word[token_id]
        sememe_token_id = overall_dict.word2idx[word]
        sememe_senses = overall_dict.idx2senses[sememe_token_id]
        max_senses = max(len(sememe_senses), max_senses)
        sememe_targets.append(sememe_senses)
    return sememe_targets, max_senses


def get_sememe_target(target_semes):
    stargets = torch.zeros(len(target_semes), nsememes)
    for i in range(len(target_semes)):
        sememes = []
        for sense in target_semes[i]:
            for sememe in sense:
                sememes.append(sememe)
        sememes = set(sememes)
        for sememe in sememes:
            stargets[i, sememe_idxs[sememe]] = 1.0
    stargets = Variable(stargets)
    if args.cuda:
        stargets = stargets.cuda()
    return stargets


def get_sense_labels(tokens):
    sense_target = torch.zeros(len(tokens), nsenses)
    for i in range(tokens.size(0)):
        token_id = tokens[i, 0]
        for sense_id in word_sense[token_id]:
            sense_target[i, sense_id] = 1.0
    sense_target = Variable(sense_target)
    if args.cuda:
        sense_target = sense_target.cuda()
    return sense_target


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    total_sememe_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        batch_size = targets.size(0)
        labels = torch.zeros(batch_size, ntokens).scatter_(1, targets.data.view(batch_size, 1), 1)
        if args.cuda:
            labels = labels.cuda()
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        # model
        output_prob, s_output_prob, hidden, sense_prob = \
            model(data, hidden, sou_sparsemm1, sou_sparsemm2, sense=True)

        # calc word loss
        # we add this term for numerical stability during training
        cs_e = torch.mul(-torch.log(output_prob + 1e-9 * (1 - Variable(labels))), Variable(labels))
        loss = torch.mean(torch.sum(cs_e, dim=1))

        # concat loss
        loss_concat = loss
        loss_concat.backward()
        #print(loss.data[0])

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

        for p in model.parameters():
            if p.grad is not None:
                p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                  epoch, batch, len(train_data) // args.bptt, lr,
                  elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def demo(num_sentences, len):
    model.eval()
    hidden = model.init_hidden(num_sentences)
    data = torch.LongTensor([corpus.dictionary.word2idx['<eos>']] * num_sentences).view(1, num_sentences)
    data = Variable(data, volatile=True)
    print(data)
    sentences = []
    for i in range(num_sentences):
        sentences.append([])
    for i in range(len):
        output_prob, sememe_prob, hidden = model(data, hidden)
        dist = torch.distributions.Categorical(output_prob)
        data = dist.sample()
        #print(data)
        print('generating sentences ... {}/{}'.format(i+1, len))
        for i in range(num_sentences):
            sentences[i].append(corpus.dictionary.idx2word[data.data[i]])
        data = data.view(1, num_sentences)
        hidden = repackage_hidden(hidden)
    for i in range(num_sentences):
        print(' '.join(sentences[i]))


def bdemo(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, s_output, hidden = model(data, hidden, sou_sparsemm1,
                                  sou_sparsemm2)
        batch_size = output.size(0)
        labels = torch.zeros(batch_size, ntokens).scatter_(1, targets.data.view(batch_size, 1), 1)
        if args.cuda:
            labels = labels.cuda()
        total_loss += len(data) * torch.mean(torch.sum(torch.mul(-torch.log(output), Variable(labels)), 1)).data
        batch_demo(corpus.dictionary, overall_dict, sememe_idxs,
                   labels, output.data, None, s_output.data)
        return total_loss / len(data)
    return total_loss[0] / len(data_source)

# Loop over epochs.
lr = args.lr
best_val_loss = None

try:
    with open(args.save, 'rb') as f:
        model = torch.load(f)
    print('success')
except:
    pass

if args.mode == 'train':
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if epoch < 15 or (not best_val_loss or val_loss < best_val_loss):
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 2.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
