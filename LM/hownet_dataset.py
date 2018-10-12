# coding: utf-8

import torch
import re
from dataset import Dictionary
from dataset_rm import RenminCorpus
from hownet import SememeDictionary


def check_num(word):
    re_nums = []
    re_nums.append(re.compile(u'[0-9．·]+'))
    re_nums.append(re.compile(u'[0-9．·]+年'))
    re_nums.append(re.compile(u'[0-9．·]+日'))
    re_nums.append(re.compile(u'[0-9．·]+时'))
    re_nums.append(re.compile(u'[0-9．·]+万'))
    re_nums.append(re.compile(u'[0-9．·]+亿'))
    re_nums.append(re.compile(u'[0-9．·]+万亿'))
    re_nums.append(re.compile(u'[0-9．·]+％'))
    re_nums.append(re.compile(u'百分之[十百千万亿零一二三四五六七八九]+'))
    re_nums.append(re.compile(u'百分之[十百千万亿零一二三四五六七八九]+点[十百千万亿零一二三四五六七八九]'))
    for re_num in re_nums:
        if re_num.match(word) is not None and re_num.match(word).group(0) == word:
            return True
    return False


def replace_digit(word):
    rp_dict = {'０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
               '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
               '／': '/'}
    rp_tag = False
    for j in range(len(word)):
        if word[j] in rp_dict:
            rp_tag = True
    if rp_tag:
        st = ''
        for j in range(len(word)):
            if word[j] in rp_dict:
                st += rp_dict[word[j]]
            else:
                st += word[j]
        word = st
    return word


def replace_word(word):
    re_nums = []
    re_nums.append(re.compile(u'[0-9．·]+'))
    re_nums.append(re.compile(u'[0-9．·]+/[0-9．·]+'))
    re_nums.append(re.compile(u'[0-9．·]+％'))
    re_nums.append(re.compile(u'[0-9．·]+万'))
    re_nums.append(re.compile(u'[0-9．·]+亿'))
    re_nums.append(re.compile(u'[0-9．·]+万亿'))
    re_nums.append(re.compile(u'[十百千万亿零一二三四五六七八九]+'))
    re_nums.append(re.compile(u'[十百千万亿零一二三四五六七八九]+点[十百千万亿零一二三四五六七八九]+'))
    re_nums.append(re.compile(u'百分之[十百千万亿零一二三四五六七八九]+'))
    re_nums.append(re.compile(u'百分之[十百千万亿零一二三四五六七八九]+点[十百千万亿零一二三四五六七八九]+'))
    re_nums.append(re.compile(u'第[十百千万亿零一二三四五六七八九]+[十百千万亿零一二三四五六七八九]+'))
    for re_num in re_nums:
        if re_num.match(word) is not None and re_num.match(word).group(0) == word:
            return '<N>'
    re_nums = []
    re_nums.append(re.compile(u'[0-9．·]+∶[0-9．·]+'))
    for re_num in re_nums:
        if re_num.match(word) is not None and re_num.match(word).group(0) == word:
            return '<versus>'
    re_nums = []
    re_nums.append(re.compile(u'[0-9．·]+年'))
    re_nums.append(re.compile(u'[十百千万亿零一二三四五六七八九][十百千万亿零一二三四五六七八九]'
                              u'[十百千万亿零一二三四五六七八九][十百千万亿零一二三四五六七八九]年'))
    for re_num in re_nums:
        if re_num.match(word) is not None and re_num.match(word).group(0) == word:
            return '<year>'
    re_nums = []
    re_nums.append(re.compile(u'[0-9．·]+日'))
    re_nums.append(re.compile(u'[十百千万亿零一二三四五六七八九]+日'))
    for re_num in re_nums:
        if re_num.match(word) is not None and re_num.match(word).group(0) == word:
            return '<date>'
    re_nums = []
    re_nums.append(re.compile(u'[0-9．·]+时'))
    re_nums.append(re.compile(u'[十百千万亿零一二三四五六七八九]+时'))
    for re_num in re_nums:
        if re_num.match(word) is not None and re_num.match(word).group(0) == word:
            return '<hour>'
    return word


if __name__ == '__main__':
    overall_dict = SememeDictionary()
    overall_dict.summary()
    raw_dict = Dictionary()

    path_in = './data/renmin/'
    path_out = './data/renmin_hownet/'

    tot_words = 0

    pun = ['，', '。', '！', '¥', '（', '）', '《', '》', '？', '、', '【', '】', '「', '」',
           '“', '”', '°', '／', '……', '『', '』', '●', '△', '℃', '▲', '：', '；', '—', '———', '--']

    def dfs_main(word, i):
        if i == len(word):
            return []
        for j in range(len(word) - i):
            if overall_dict.exist(word[i: len(word) - j]):
                ans = dfs_main(word, len(word) - j)
                if ans is not None:
                    return [word[i: len(word) - j]] + ans
        return None

    def dfs_search(word):
        return dfs_main(word, 0)

    def read_in_dict(filename):
        f = open(path_in + filename)
        tw = 0
        for line in f.readlines():
            words = line.split()
            for word in words:
                #if word in pun:
                #    continue
                word = replace_word(replace_digit(word))
                tw += 1
                if overall_dict.exist(word):
                    overall_dict.add_word_f(word)
                    raw_dict.add_word(word)
                else:
                    raw_dict.add_word(word)

        return tw

    tot_words += read_in_dict('train.txt')
    tot_words += read_in_dict('valid.txt')
    tot_words += read_in_dict('test.txt')

    threshold = 5

    tot_sememe_missing = 0
    for word in raw_dict.idx2word:
        if raw_dict.idx2freq[raw_dict.word2idx[word]] >= threshold and not overall_dict.exist(word):
            #if word in pun:
            #    continue
            search_words = dfs_search(word)
            if search_words is None:
                print(word + ': Not found')
                #raw_dict.add_word(word)
            else:
                for single_word in search_words:
                    for j in range(raw_dict.idx2freq[raw_dict.word2idx[word]]):
                        overall_dict.add_word_f(single_word)
                        raw_dict.add_word(single_word)
                tot_words += (len(search_words) - 1) * raw_dict.idx2freq[raw_dict.word2idx[word]]

    overall_dict.set_threshold(threshold)

    overall_dict.sememe_word_visit(raw_dict.word2idx)
    c_tot_words = 0
    delete_word = []

    def output(filename):
        of = open(path_out + filename, 'w')
        f = open(path_in + filename)
        ctw = 0
        for line in f.readlines():
            words = line.split()
            wordlist = []
            for word in words:
                #if word in pun:
                #    continue
                word = replace_word(replace_digit(word))

                if overall_dict.exist(word):
                    if overall_dict.query_count(word) >= threshold:
                        wordlist.append(word)
                        ctw += 1
                    else:
                        wordlist.append('<unk>')
                else:
                    if raw_dict.idx2freq[raw_dict.word2idx[word]] < threshold:
                        wordlist.append('<unk>')
                    else:
                        search_words = dfs_search(word)
                        #if word in pun:
                        #    wordlist.append(word)
                        #    ctw += 1
                        #    continue

                        if search_words is None:
                            wordlist.append('<unk>')
                            #print(word + ': Not found')
                        else:
                            for single_word in search_words:
                                wordlist.append(single_word)
                            ctw += len(search_words)
            of.write(' '.join(wordlist) + '\n')
        return ctw

    c_tot_words += output('train.txt')
    c_tot_words += output('valid.txt')
    c_tot_words += output('test.txt')

    ntokens = len(overall_dict) - overall_dict.freq_le(1)
    c_ntokens = len(overall_dict) - overall_dict.freq_le(threshold)

    print('=' * 89)
    print('-' * 41 + 'SUMMARY' + '-' * 41)
    words_ratio = (c_tot_words + 0.0) / tot_words
    tokens_ratio = (c_ntokens + 0.0) / ntokens
    print('total tokens = {}/{}, ({}%)'.
          format(c_ntokens, ntokens, tokens_ratio * 100))
    print('total number of words = {}/{}, ({}%)'.
          format(c_tot_words, tot_words, words_ratio * 100))
    print('total number of sememes = {}'.format(overall_dict.sememe_dict.freq_ge(1)))
    print('=' * 89)
