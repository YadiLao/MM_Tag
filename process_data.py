# !/usr/bin/python
# -*- coding:utf-8 -*-
# Author: yadi Lao
import os
import codecs
import numpy as np
import json
import pickle

setting = {
    'dataset': 'conll03',
    'data_path': '/Users/yadi/PycharmProjects/MM_Tag/data/',
    'embedding_file': 'glove.6B.100d.txt',
    'embedding_dim': 100,
    'train_embed': True,
    'tag_format': 'IOB',  # IOB, BIO, BIOES
    'debug': True,
}

BASE_DIR = setting['data_path']
DATASET = setting['dataset']


def readfile(datafile, max_len=None, is_lower=False):
    """
    read raw train/dev/test data
    remain text and do lower when mapping with embedding
    """
    if datafile.endswith('txt'):
        f = open(datafile, 'r')
        lines = f.readlines()
        word = []
        pos = []
        chunk = []
        ner = []
        word_list = []
        pos_list = []
        chunk_list = []
        ner_list = []
        for line in lines:
            if len(line) > 10 and line[0:10] == '-DOCSTART-':
                continue
            data = str(line).replace('\\n', '').split()

            if len(data) < 4 and word != []:
                word_list.append(word)
                pos_list.append(pos)
                chunk_list.append(chunk)
                ner_list.append(ner)
                word = []
                pos = []
                chunk = []
                ner = []
            if len(data) == 4:
                if is_lower:
                    word.append(data[0].lower())
                else:
                    word.append(data[0])
                pos.append(data[1])
                chunk.append(data[2])
                ner.append(data[3])

    else:
        raise NotImplementedError

    # constain on length
    if max_len != None:
        word_list = [l for l in word_list if len(l) < max_len[1] and len(l) > max_len[0]]
        pos_list = [l for l in pos_list if len(l) < max_len[1] and len(l) > max_len[0]]
        chunk_list = [l for l in chunk_list if len(l) < max_len[1] and len(l) > max_len[0]]
        ner_list = [l for l in ner_list if len(l) < max_len[1] and len(l) > max_len[0]]

    return [word_list, pos_list, chunk_list, ner_list]


def merge_vocab(train, valid, test):
    """
    merge vocab
    """
    vocab = {}
    data = [train, valid, test]
    for i in range(len(data)):
        for line in data[i]:
            for item in line:
                if item not in vocab.keys():
                    vocab[item] = len(vocab)

    return vocab


def build_train_vocab(x_train):
    """
    build train vocab
    """
    x_vocab = {}
    for line in x_train:
        for w in line:
            w = w.lower()
            if w not in x_vocab:
                x_vocab[w] = 1
            else:
                x_vocab[w] += 1
    return x_vocab


def build_char_vocab(x_train):
    """
    build char vocab
    """
    char_list = []
    for line in x_train:
        for w in line:
            char_list.append(list(w))
    char_list = list(set([c for cc in char_list for c in cc]))
    char_vocab = dict(zip(char_list, range(2, len(char_list) + 2)))
    char_vocab['<PAD>'] = 0
    char_vocab['<UNK>'] = 1

    return char_vocab


def load_pretrain_embedding(pretrain_embed_file):
    """
    load word embedding
    """
    print('Indexing word vectors.')
    word2embed = {}
    with codecs.open(pretrain_embed_file, 'rb', 'utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = [float(v) for v in values[1:]]
            word2embed[word] = coefs

    return word2embed


def add_unknown_words(word_vecs, vocab, min_df=1, k=100):
    """
    add random vectors of unknown words which are not in pre-trained vector file.
    if pre-trained vectors are not used, then initialize all words in vocab with random value.
    """
    unk_count = 0
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)
            unk_count += 1
    print('add {} unk word'.format(unk_count))


def get_w(word_vecs, k=100):
    """
    get word_embed and word_id_dict
    W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    word_embed = np.zeros(shape=(vocab_size + 2, k), dtype='float32')
    word_embed[0] = np.zeros(k, dtype='float32')
    word_idx_map['<PAD>'] = 0
    word_embed[1] = np.random.uniform(-0.25, 0.25, k)
    word_idx_map['<UNK>'] = 1
    i = 2
    for word in word_vecs:
        word_embed[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return word_embed, word_idx_map


def check_embed_file(fname1, fname2):
    """
    check embedding file with 100d or 200d are in the same word id
    """
    f = open(fname1, 'r', encoding='utf-8')
    f2 = open(fname2, 'r', encoding='utf-8')
    vocab1 = {}
    vocab2 = {}
    for line in f:
        values = line.split()
        word = values[0]
        vocab1[len(vocab1)] = word
    f.close()

    for line in f2:
        values = line.split()
        word = values[0]
        vocab2[len(vocab2)] = word
    f2.close()
    print('size: vocab1={}, vocab2={}'.format(len(vocab1), len(vocab2)))

    for i in range(len(vocab1)):
        assert vocab1[i] == vocab2[i]


def word2index(docs, vocab, max_len, need_pad=False):
    """
    map word
    """
    index_docs = []
    for doc in docs:
        index_d = []
        for char in doc:
            char = char.lower()
            if char in vocab.keys():
                index_d.append(vocab[char])
            else:
                index_d.append(vocab['<UNK>'])
        index_docs.append(index_d)
    if need_pad:
        index_docs = [doc + [vocab['<PAD>']] * (max_len - len(doc)) for doc in index_docs]
        index_docs = np.array(index_docs)
    return np.array(index_docs)


def load_pos_chunk_ner_label():
    """
    load pos, chunk and ner label map
    """
    chunk_vocab = json.load(open(BASE_DIR + '/' + DATASET + '/mapping/chunk_vocab', 'r'))
    pos_vocab = json.load(open(BASE_DIR + '/' + DATASET + '/mapping/pos_vocab', 'r'))
    if setting['tag_format'] == 'BIOES':
        ner_vocab = json.load(open(BASE_DIR + '/' + DATASET + '/mapping/ner_vocab_bioes', 'r'))
    else:
        ner_vocab = json.load(open(BASE_DIR + '/' + DATASET + '/mapping/ner_vocab', 'r'))
    return pos_vocab, chunk_vocab, ner_vocab


def pos2index(pos, pos_vocab, max_len, need_pad=False):
    """
    index pos
    """
    pos_docs = [[pos_vocab[char] for char in doc] for doc in pos]
    if need_pad:
        pos_docs = [doc + [0] * (max_len - len(doc)) for doc in pos_docs]
        pos_docs = np.array(pos_docs)
    return pos_docs


def chunk2index(chunk, chunk_vocab, max_len, need_pad=False):
    """
    index chunk
    """
    chunk_docs = [[chunk_vocab[char] for char in doc] for doc in chunk]
    if need_pad:
        chunk_docs = [doc + [0] * (max_len - len(doc)) for doc in chunk_docs]
        chunk_docs = np.array(chunk_docs)
    return chunk_docs


def ner2index(ner, ner_vocab, max_len, need_pad=False):
    """
    index ner
    """
    ner_docs = [[ner_vocab[char] for char in doc] for doc in ner]
    if need_pad:
        ner_docs = [doc + [0] * (max_len - len(doc)) for doc in ner_docs]
        ner_docs = np.array(ner_docs)
    return np.array(ner_docs)


def char2index(data, char_vocab):
    """
    index char
    """
    char_ids = []
    for line in data:
        chars_per_line = []
        for word in line:
            chars_per_word = []
            for c in list(word):
                if c in char_vocab:
                    chars_per_word.append(char_vocab[c])
                else:
                    chars_per_word.append(char_vocab['<UNK>'])

            chars_per_line.append(chars_per_word)
        char_ids.append(chars_per_line)

    return np.array(char_ids)


def bio_iob(tags):
    """
    BIO -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if 'B' in tag:
            if i == 0:
                new_tags.append('I-' + tag[2:])
            else:
                pre_type = tags[i - 1][2:]
                if pre_type == tag[2:]:
                    new_tags.append(tag)
                else:
                    new_tags.append('I-' + tag[2:])
        else:
            new_tags.append(tag)
    return new_tags


def iob_bio(tags):
    """
    IOB -> BIO
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if 'I' in tag:
            if i == 0:
                new_tags.append('B-' + tag[2:])
            else:
                pre_type = tags[i - 1][2:]
                if pre_type == tag[2:]:
                    new_tags.append(tag)
                else:
                    new_tags.append('B-' + tag[2:])
        else:
            new_tags.append(tag)
    return new_tags


def bio_bioes(tags):
    """
    BIO -> BIOES
    """
    new_tags = []
    length = len(tags)
    for i, tag in enumerate(tags):
        if "-" not in tags[i]:
            new_tags.append(tag)
        else:
            label_type = tag[2:]
            if "B-" in tag:
                if (i == length - 1) or ("I-" not in tags[i + 1]):
                    new_tags.append("S-" + label_type)
                else:
                    new_tags.append("B-" + label_type)
            elif "I-" in tag:
                if (i == length - 1) or ("I-" not in tags[i + 1]):
                    new_tags.append("E-" + label_type)
                else:
                    new_tags.append("I-" + label_type)

    return new_tags


def bioes_bio(tags):
    """
    BIOES -> BIO
    """
    new_tags = []
    length = len(tags)
    for i, tag in enumerate(tags):
        if "-" not in tags[i]:
            new_tags.append(tag)
        else:
            label_type = tag[2:]
            if "E-" in tag:
                new_tags.append("I-" + label_type)
            elif "S-" in tag:
                new_tags.append("B-" + label_type)
            else:
                new_tags.append(tag)

    return new_tags


def process_data():
    """
    process conll03 data, obtain index sen, pos, chunk, ner label
    """
    # load train/valid/test data
    train_data = readfile(BASE_DIR + '/' + DATASET + '/train.txt', is_lower=False)
    valid_data = readfile(BASE_DIR + '/' + DATASET + '/valid.txt', is_lower=False)
    test_data = readfile(BASE_DIR + '/' + DATASET + '/test.txt', is_lower=False)
    print('size: train={}, valid={}, test={}'.format(
        len(train_data[0]), len(valid_data[0]), len(test_data[0])))

    # load label dict
    if os.path.exists(BASE_DIR + '/' + DATASET + '/mapping/ner_vocab'):
        pos_vocab, chunk_vocab, ner_vocab = load_pos_chunk_ner_label()
    else:
        pos_vocab = merge_vocab(train_data[1], valid_data[1], test_data[1])
        chunk_vocab = merge_vocab(train_data[2], valid_data[2], test_data[2])
        ner_vocab = merge_vocab(train_data[3], valid_data[3], test_data[3])
        print('ner={} pos={},chunk={}'.format(ner_vocab, pos_vocab, chunk_vocab))
        json.dump(pos_vocab, open(BASE_DIR + '/' + DATASET + '/mapping/pos_vocab', 'w'))
        json.dump(ner_vocab, open(BASE_DIR + '/' + DATASET + '/mapping/ner_vocab', 'w'))
        json.dump(chunk_vocab, open(BASE_DIR + '/' + DATASET + '/mapping/chunk_vocab', 'w'))

    # load embedding and add UNK word in training data
    word_vecs = load_pretrain_embedding(BASE_DIR + setting['embedding_file'])
    x_train_vocab = build_train_vocab(train_data[0])
    add_unknown_words(word_vecs, vocab=x_train_vocab, k=setting['embedding_dim'])
    word_embed, word_idx_map = get_w(word_vecs=word_vecs, k=setting['embedding_dim'])
    json.dump(word_idx_map, open(BASE_DIR + DATASET + '/mapping/word_vocab', 'w'))

    char_vocab = build_char_vocab(train_data[0])
    json.dump(char_vocab, open(BASE_DIR + DATASET + '/mapping/chars_vocab', 'w'))

    print('char size={}'.format(len(char_vocab)))

    # index file

    print('index chars ...')
    char_train = char2index(train_data[0], char_vocab)
    char_test = char2index(test_data[0], char_vocab)
    char_valid = char2index(valid_data[0], char_vocab)

    print('index word ...')
    x_train = word2index(train_data[0], word_idx_map, max_len=None)
    x_test = word2index(test_data[0], word_idx_map, max_len=None)
    x_valid = word2index(valid_data[0], word_idx_map, max_len=None)

    print('index pos ...')
    pos_train = pos2index(train_data[1], pos_vocab, max_len=None)
    pos_test = pos2index(test_data[1], pos_vocab, max_len=None)
    pos_valid = pos2index(valid_data[1], pos_vocab, max_len=None)

    print('index chunk ...')
    chunk_train = chunk2index(train_data[2], chunk_vocab, max_len=None)
    chunk_test = chunk2index(test_data[2], chunk_vocab, max_len=None)
    chunk_valid = chunk2index(valid_data[2], chunk_vocab, max_len=None)

    print(ner_vocab)
    if setting['tag_format'] == 'IOB':
        iob_train = []
        for ner in train_data[3]:
            iob_train.append(bio_iob(ner))
        train_data[3] = iob_train

        iob_test = []
        for ner in test_data[3]:
            iob_test.append(bio_iob(ner))
        test_data[3] = iob_test

        iob_valid = []
        for ner in valid_data[3]:
            iob_valid.append(bio_iob(ner))
        valid_data[3] = iob_valid

    if setting['tag_format'] == 'BIOES':
        bioes_train = []
        for ner in train_data[3]:
            bioes_train.append(bio_bioes(ner))
        train_data[3] = bioes_train

        bioes_test = []
        for ner in test_data[3]:
            bioes_test.append(bio_bioes(ner))
        test_data[3] = bioes_test

        bioes_valid = []
        for ner in valid_data[3]:
            bioes_valid.append(bio_bioes(ner))
        valid_data[3] = bioes_valid

    print('index ner ...')
    ner_train = ner2index(train_data[3], ner_vocab, max_len=None)
    ner_test = ner2index(test_data[3], ner_vocab, max_len=None)
    ner_valid = ner2index(valid_data[3], ner_vocab, max_len=None)

    # dump data
    print('dump data..')
    data_file = BASE_DIR + DATASET + '/' + DATASET + '_'+setting['tag_format']+'_100d.p'

    raw_sen_list = [train_data[0], valid_data[0], test_data[0]]
    sen_list = [x_train, x_valid, x_test]
    pos_list = [pos_train, pos_valid, pos_test]
    chunk_list = [chunk_train, chunk_valid, chunk_test]
    ner_list = [ner_train, ner_valid, ner_test]
    char_list = [char_train, char_valid, char_test]

    pickle.dump(
        [raw_sen_list, sen_list, pos_list, chunk_list, ner_list, char_list, word_embed, word_idx_map],
        open(data_file, 'wb'))
    print('save to {}'.format(data_file))


if __name__ == '__main__':
    process_data()
