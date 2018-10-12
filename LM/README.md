# Code for our Langauge Modeling experiments

The code for the language modeling task using SDLM in **Language Modeling with Sparse Product of Sememe Experts** (EMNLP 2018).

## Data

Here we provide the detailed information about every components in our proposed `People's Daily` dataset.

- `HowNet.txt`: HowNet annotations
- `renmin.txt`: The original text of corpus
- `renmin/*.txt`: The original corpus, which is divided into train/valid/test set
- `renmin_hownet/*.txt`: Our processed corpus.

## Code

The code is compatible with PyTorch 0.3.1 and Python 3.6. (and be tested on the environment)

### Utils

- `hownet.py`: Build a sememe dictionary based on the annotations in HowNet.
- `data.py`: Directly copy from `word\_language\_modeling` of `\pytorch-examples`.
- `hownet_utils.py`: `spmm` targets at making it feasible to backpropagate the sparse matrix multiplication (when we finished our experiments, PyTorch don't have such feature).
- `hownet_dataset.py`: Target at building a dataset given the corpus.

### Tied LSTM

- `tied_lstm_rnn.py`: Neural network module
- `run_tied_lstm.py`: Script to run the code

### Awd-lstm

- `run_awd_lstm.py`: Neural network module
- `awd_lstm_rnn.py`: Script to run the code

## Run Language Modeling

Here are the commands to run our code:

For Medium Tied lstm:

```
python3 run_tied_lstm.py --emsize 650 --nhid 650 --dropout 0.6 --data ./data/renmin_hownet --save TM.pt --lr 20 --epoch 80 --cuda
```

For Large Tied lstm:

```
python3 run_tied_lstm.py --emsize 1500 --nhid 1500 --dropout 0.7 --data ./data/renmin_hownet --save TL.pt --output sou --lr 20 --epoch 80 --cuda
```

For Awd lstm:

```
python3 run_awd_lstm.py --batch_size 15 --data ./data/renmin_hownet --dropouti 0.5 --dropouth 0.2 --seed 141 --epoch 100 --save AWD.pt --cuda
```

There are some randomness, we run the code again, the results is (valid/test)

- Medium:   96.66/96.02
- Large:    93.05/92.88
- AWD:      88.17/87.40

### Running Baseline


For Medium Tied lstm:

```
python3 run_tied_lstm_bl.py --emsize 650 --nhid 650 --dropout 0.6 --data ./data/renmin_hownet --save TMBL.pt --lr 20 --epoch 80 --cuda
```

For Large Tied lstm:

```
python3 run_tied_lstm_bl.py --emsize 1500 --nhid 1500 --dropout 0.7 --data ./data/renmin_hownet --save TLBL.pt --output sou --lr 20 --epoch 80 --cuda
```

For Awd lstm:

```
python3 run_awd_lstm_bl.py --batch_size 15 --data ./data/renmin_hownet --dropouti 0.5 --dropouth 0.2 --seed 141 --epoch 100 --save AWDBL.pt --cuda
```


