# Headline Generation with Sememe-Driven Language Model

The code for the headline generation task using SDLM in **Language Modeling with Sparse Product of Sememe Experts** (EMNLP 2018).

# Prerequisite

The code has been tested on:

- Python 3.6
- PyTorch 0.3.1

It's based on [OpenNMT/OpenNMT-py (version: 6215e73)](OpenNMT/OpenNMT-py (version: 6215e73)).

## Usage

### Preparation

1. To acquire the LCSTS dataset (A Large-Scale Chinese Short Text Summarization Dataset), please visit: 

   http://icrc.hitsz.edu.cn/Article/show/139.html

2. Tokenize the corpus to ensure that all words are included in HowNet, which can alleviate the OOV problems. Our practice below is for your reference: 

   1. Extract headline-content pairs with score 3, 4 and 5 from the corpus. Take pairs from PART-I
      that do not occur in PART-II as the train set, PART-II as the development set and PART-III as the test set.
   2. Build a [JieBa](https://pypi.org/project/jieba/) user dictionary by counting the word frequency of the processed People’s Daily Corpus  used in our language modeling task.
   3. Use JieBa to do text segmentation with the user dictionary and the `--no-hmm` option on the corpus.

3. Rename files we get as: `train.article.txt`, `train.title.txt`, `valid.article.txt`, `valid.title.txt`, `test.article.txt`, `test.title.txt`. Their statistics are as below:

| set   | \# pairs    |
| ----- | ----------- |
| train | 2, 393, 662 |
| valid | 8, 685      |
| test  | 725         |

4. Create char-level test titles with:

```
python split.py test.title.txt
```

5. Clone this repository, reorganize files as follows and put `HowNet.txt` into `OpenNMT-py/data/`:

   **Project Directory Structure:**

   - data/
     - train/
       - train.article.txt
       - train.title.txt
       - valid.article.txt
       - valid.title.txt
     - test/
       - test.article.txt
       - test.title.txt
       - test.title_char.txt
       - split.py (included in this repository)
       - ROUGE_with_ranked.pl (included in this repository)
   - OpenNMT-py/ (this repository)

6. `cd OpenNMT-py`

### Preprocess

```
python preprocess.py -train_src ../data/train/train.article.txt -train_tgt ../data/train/train.title.txt -valid_src ../data/train/valid.article.txt -valid_tgt ../data/train/valid.title.txt -src_vocab_size 40000 -tgt_vocab_size 40000 -save_data ../data/train/light_textsum -share_vocab -src_seq_length 1000 -tgt_seq_length 1000 -max_shard_size 26214400
```

Running the above command will preprocess the train and validation set as well as build a vocab file.

Full reference for the meanings of the parameters can be found here: [OpenNMT-py documentation](http://opennmt.net/OpenNMT-py/).

### Train

```
python train.py -encoder_type brnn -layers 1 -data ../data/train/light_textsum -gpuid 0 \
-epochs 15 \
-word_vec_size 250 -rnn_size 250 \
-batch_size 64 \
-dropout 0.2 \
-save_model sum_sememe \
-seed 12345 \
-optim adam \
-learning_rate 0.001 \
-max_grad_norm 5
```

Running the above command will start training and save a checkpoint at the end of each epoch.

### Translate

Suppose the checkpoint we want to evaluate is named `sum_sememe_acc_44.85_ppl_35.17_e11.pt`. Then run below command to do summarization on the test set.

```
python translate.py -model sum_sememe_acc_44.85_ppl_35.17_e11.pt -src ../data/test/test.article.txt -output ../data/test/sum_sememe_acc_44.85_ppl_35.17_e11.txt -gpu 0
```

Go to the `test` directory and split the predictions into characters.

```
cd ../data/test
python split.py sum_sememe_acc_44.85_ppl_35.17_e11.txt
```

Then we will get `sum_sememe_acc_44.85_ppl_35.17_e11_char.txt`.

### Evaluate

```
perl ROUGE_with_ranked.pl 1 N test.title_char.txt sum_sememe_acc_44.85_ppl_35.17_e11_char.txt
perl ROUGE_with_ranked.pl 2 N test.title_char.txt sum_sememe_acc_44.85_ppl_35.17_e11_char.txt
perl ROUGE_with_ranked.pl 1 L test.title_char.txt sum_sememe_acc_44.85_ppl_35.17_e11_char.txt
```

Running the above command will calculate Rouge-1, Rouge-2 and Rouge-L for the predictions.

For the above setting, we got:

```
ROUGE-1
Avg_R | Avg_P | Avg_F
0.352   0.464   0.389

ROUGE-2
Avg_R | Avg_P | Avg_F
0.235   0.316   0.262

ROUGE-L
Avg_R | Avg_P | Avg_F
0.327   0.432   0.362
```

> `ROUGE_with_ranked.pl` is borrowed from [playma/OpenNMT-py](playma/OpenNMT-py). Thanks [@playma](https://github.com/playma)!

