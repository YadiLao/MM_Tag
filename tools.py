import pickle
import json
import numpy as np
import operator
from functools import reduce
import matplotlib.pyplot as plt


setting = {
    'dataset': 'conll03',
    'data_path': '/Users/yadi/PycharmProjects/go_master/data/',
    'embedding_file': 'glove.6B.100d.txt',
    'embedding_dim': 100,
    'train_embed': True,
    'tag_format': 'IOB',
    'experiment': 10,
    'debug': True,
    'use_dropout': True,
    'same_drop_prob': 0.1,
}

BASE_DIR = setting['data_path']
DATA_SET = setting['dataset']

word2id = dict(json.load(open(BASE_DIR + DATA_SET + '/mapping/word_vocab', 'r')))
char2id = dict(json.load(open(BASE_DIR + DATA_SET + '/mapping/chars_vocab', 'r')))
pos2id = dict(json.load(open(BASE_DIR + DATA_SET + '/mapping/pos_vocab', 'r')))
chunk2id = dict(json.load(open(BASE_DIR + DATA_SET + '/mapping/chunk_vocab', 'r')))
id2word = dict(zip(word2id.values(), word2id.keys()))
id2char = dict(zip(char2id.values(), char2id.keys()))
id2pos = dict(zip(pos2id.values(), pos2id.keys()))
id2chunk = dict(zip(chunk2id.values(), chunk2id.keys()))


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, sen, char, pos, chunk, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.sen = sen
        self.label = label
        self.char = char
        self.pos = pos
        self.chunk = chunk
        self.sen_len = len(self.sen)
        self.char_len = [len(c) for c in self.char]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):

        s = ""
        s += "qas_id: %s" % (self.guid)
        s += ", text-%s:" % (str(self.sen_len), ' '.join([str(s) for s in self.sen]))
        return s

    def _print(self):
        print('******** print example ***********')
        print('id: %s' % self.guid)
        print("text-%s: %s" % (str(self.sen_len), ' '.join([str(s) for s in self.sen])))
        print("char-%s:" % (' '.join([str(c) for c in self.char_len])))


def create_examples(sen_list, ner_list, char_list, pos_list, chunk_list, set_type):
    examples = []
    for i, sen in enumerate(sen_list):
        guid = "%s-%s" % (set_type, i)
        sen = sen_list[i]
        label = ner_list[i]
        char = char_list[i]
        pos = pos_list[i]
        chunk = chunk_list[i]
        examples.append(InputExample(
            guid=guid, sen=sen, char=char, pos=pos, chunk=chunk, label=label))

    return examples


def load_data():
    print('Loading Data...')
    data_file = open(BASE_DIR + DATA_SET + '/' + DATA_SET +'_IOB_100d.p', 'rb')
    x = pickle.load(data_file)
    data_file.close()
    return x


def bio_iob(tags):
    """
    BIO -> IOB
    :param tags:
    :return:
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
                    new_tags.append(" S-" + label_type)
                else:
                    new_tags.append(" B-" + label_type)
            elif "I-" in tag:
                if (i == length - 1) or ("I-" not in tags[i + 1]):
                    new_tags.append(" E-" + label_type)
                else:
                    new_tags.append(" I-" + label_type)

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


def bio_to_spans(sequence, vocab, strict_iob2=False):
    """
    convert to iob to span
    """
    iobtype = 2 if strict_iob2 else 1
    chunks = []
    current = None

    for i, y in enumerate(sequence):
        label = vocab[y]

        if label.startswith('B-'):
            if current is not None:
                chunks.append('@'.join(current))
            current = [label.replace('B-', ''), '%d' % i]

        elif label.startswith('I-'):

            if current is not None:
                base = label.replace('I-', '')
                if base == current[0]:
                    current.append('%d' % i)
                else:
                    chunks.append('@'.join(current))
                    if iobtype == 2:
                        print('Warning, type=IOB2, unexpected format ([%s] follows other tag type [%s] @ %d)' % (label, current[0], i))

                    current = [label, '%d' % i]

            else:
                # current = [label.replace('I-', ''), '%d' % i]
                chunks.append(label+'@'+str(i))
                if iobtype == 2:
                    print('Warning, unexpected format (I before B @ %d) %s' % (i, label))
        else:
            if current is not None:
                chunks.append('@'.join(current))
            current = None

    if current is not None:
        chunks.append('@'.join(current))

    return set(chunks)


def batchify_with_label(words, chars, labels, n_gram=2):
    """
        input: list of words and labels, various length. [[words, labels],[words,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
        output:
            zero padding for word, with their batch length
            fw_seq_tensor: (batch_size, max_sent_len, n_gram)
            bw_seq_tensor: (batch_size, max_sent_len, n_gram)
            word_seq_lengths: (batch_size,1) Tensor
            label_seq_tensor: (batch_size, max_sent_len)
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            Note!!!! In tensorflow, no need to sort tensors according to their length
    """
    def get_fw_bw_for_sen(seq, seq_len):
        fw_list = []
        bw_list = []
        for idx in range(seq_len):
            fw = [0] * (n_gram - idx) + seq[:idx + 1][-(n_gram + 1):]
            bw = seq[idx:idx + n_gram + 1] + [0] * (n_gram - len(seq[idx:]) + 1)
            bw = bw[::-1]
            fw_list.append(fw)
            bw_list.append(bw)
        # print('fw={}, bw={}'.format(fw_list, bw_list))
        return fw_list, bw_list

    batch_size = len(words)

    word_seq_lengths = np.array(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    fw_seq_tensor = np.zeros((batch_size, max_seq_len, n_gram+1), dtype=int)
    bw_seq_tensor = np.zeros((batch_size, max_seq_len, n_gram+1), dtype=int)
    label_seq_tensor = np.zeros((batch_size, max_seq_len), dtype=int)

    # loop for sen in a batch
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        seqlen = seqlen.item()
        fw, bw = get_fw_bw_for_sen(seq, seqlen)
        fw_seq_tensor[idx, :seqlen, :] = fw
        bw_seq_tensor[idx, :seqlen, :] = bw
        label_seq_tensor[idx, :seqlen] = label

    # deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = np.zeros((batch_size, max_seq_len, max_word_len), dtype=int)
    char_seq_lengths = np.array(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = word

    # char_seq_tensor = char_seq_tensor.reshape(batch_size * max_seq_len, -1)
    # char_seq_lengths = char_seq_lengths.reshape(batch_size * max_seq_len, )

    # print('********** check ***********')
    # print(fw_seq_tensor.shape, bw_seq_tensor.shape)
    # print(word_seq_lengths)
    # print(max_seq_len)
    # print(char_seq_tensor.shape)

    return fw_seq_tensor, bw_seq_tensor, word_seq_lengths, \
           char_seq_tensor, char_seq_lengths, label_seq_tensor


def batchify_pi_reward(feed_data, word_seq_lengths, label_num):
    """
        input:
            feed_data: dict, k=index, v=[[cur_label,..], [pi, ..], [reward,..]]
            word_seq_lengths:
        output:
            zero padding for selected label and pi and reward with their batch length
            cur_label_seq_tensor: (batch_size, max_sent_len) Variable
            pi_seq_tensor: (batch_size, max_sent_len) Tensor
            reward_seq_tensor: (batch_size, max_sent_len) Variable

            mask: (batch_size, max_sent_len)
    """
    batch_size = len(feed_data)
    max_seq_len = word_seq_lengths.max().item()

    cur_label_seq_tensor = np.zeros((batch_size, max_seq_len, label_num), dtype=int)
    pi_seq_tensor = np.zeros((batch_size, max_seq_len, label_num))
    reward_seq_tensor = np.zeros((batch_size, max_seq_len, 1))

    # already sort
    # print('recover', word_seq_recover, 'len', word_seq_lengths)
    for idx in range(batch_size):
        #  print('label ', np.array(feed_data[i][1]).shape)
        #  print('pi ', np.array(feed_data[i][2]).shape)
        #  print('reward ', np.array(feed_data[i][3]).shape)
        cur_label_seq_tensor[idx, :word_seq_lengths[idx]] = np.array(feed_data[idx][0])
        pi_seq_tensor[idx, :word_seq_lengths[idx]] = np.array(feed_data[idx][1])
        reward_seq_tensor[idx, :word_seq_lengths[idx]] = np.array(feed_data[idx][2])

    cur_label_seq_tensor = np.reshape(cur_label_seq_tensor, [batch_size*max_seq_len, label_num])
    pi_seq_tensor = np.reshape(pi_seq_tensor, [batch_size * max_seq_len, label_num])
    reward_seq_tensor = np.reshape(reward_seq_tensor, [batch_size * max_seq_len, 1])
    return cur_label_seq_tensor, pi_seq_tensor, reward_seq_tensor


def batchify_feature(char_seq_tensor, word_seq_lengths, pos_list, feature_dim=192+225):
    """
    output batch feature [b, t, d]. t is the max seq length, d is the feature num.
    """
    batch_size = len(word_seq_lengths)
    max_seq_len = word_seq_lengths.max().item()
    # print(char_seq_tensor.shape, 'batch size', batch_size)
    # char_seq_tensor = np.reshape(char_seq_tensor,
    #     [batch_size, max_seq_len, char_seq_tensor.shape[1]])

    feature_seq_tensor = np.zeros((batch_size, max_seq_len, feature_dim), dtype=int)
    for idx, (chars, seq_len) in enumerate(zip(char_seq_tensor, word_seq_lengths)):
        feature = extract_raw_feature(chars, word_seq_lengths[idx], pos_list[idx])
        feature_seq_tensor[idx, :seq_len] = feature

    return feature_seq_tensor


def extract_raw_feature(chars_list, seq_len,  pos_list, n_gram=2):
    """
    extract feature for single sentence.
    """
    def contain_digit(chars):
        return any([id2char[e].isdigit() for e in chars])

    def is_all_upper(chars):
        return all([id2char[e].isupper() for e in chars])

    def is_title(chars):
        return id2char[chars[0]].isupper()

    def contain_punctuation(chars):
        chars = [id2char[e] for e in chars]
        return len(set(chars) & set('.,\"\'+/-&(]%@?*\[')) != 0

    # print('extract {}, seq_len={}'.format(chars_list.shape, seq_len))

    feature = [[] for _ in range(seq_len)]
    prefix_feature = [0] * len(id2char)*2    # two prefix feature

    for idx in range(seq_len):
        temp = []
        if len(chars_list[idx]) >= 2:
            first_char_id = chars_list[idx][0]
            second_char_id = chars_list[idx][1]
            prefix_feature[first_char_id] = 1
            prefix_feature[second_char_id] = 1

        for i in range(idx - n_gram, idx + n_gram + 1):
            if i < 0 or i >= seq_len:
                temp.append([[0] * 4, [0] * len(id2pos)])
                continue
            word_feature = list(
                map(int, [contain_digit(chars_list[i]), is_all_upper(chars_list[i]),
                          is_title(chars_list[i]), contain_punctuation(chars_list[i])]))
            pos_feature = [0] * len(id2pos)
            pos_feature[pos_list[i]] = 1
            # chunk_feature = [0] * len(id2chunk)
            # chunk_feature[chunk_list[i]] = 1
            temp.append([word_feature, pos_feature])
        temp = reduce(operator.add, reduce(operator.add, list(map(list, list(zip(*temp))))))
        temp.extend(prefix_feature)
        feature[idx] = temp

        # print('word', idx, temp)

    # for i in range(seq_len):
    #     char = ''.join([char_vocab[c] for c in chars_list[i]])
    #     print(i, char, len(feature[i]))

    return feature


def test_feature():
    data = load_data()
    raw_sen_list = data[0]
    sen_list, pos_list, chunk_list, ner_list, char_list = data[1], data[2], data[3], data[4], data[5]
    word_embed, word_idx_map = data[6], data[7]
    print('word_embed={}, dim={}'.format(len(word_embed), len(word_embed[0])))

    x_train, x_dev, x_test = sen_list[0][:4], sen_list[1], sen_list[2]
    y_train, y_dev, y_test = ner_list[0][:4], ner_list[1], ner_list[2]
    char_train, char_dev, char_test = char_list[0][:4], char_list[1], char_list[2]
    pos_train, pos_dev, pos_test = pos_list[0][:4], pos_list[1], pos_list[2]
    chunk_train, chunk_dev, chunk_test = chunk_list[0][:4], chunk_list[1], chunk_list[2]

    examples = create_examples(x_train, y_train, char_train, pos_train, chunk_train, 'train')
    for token in examples[0].char:
        print(''.join(id2char[c] for c in token))

    examples[0]._print()

    sen_list = [e.sen for e in examples]
    label_list = [e.label for e in examples]
    char_list = [e.char for e in examples]
    pos_list = [e.pos for e in examples]
    chunk_list = [e.chunk for e in examples]
    print(sen_list)
    print(label_list)
    print(char_list)

    data = batchify_with_label(sen_list, char_list, label_list)

    fw_seq_tensor, bw_seq_tensor, word_seq_lengths = data[0], data[1], data[2]
    char_seq_tensor, char_seq_lengths = data[3], data[4]
    label_seq_tensor = data[5]

    print('***** check tensor ******')
    print(fw_seq_tensor.shape)
    print(bw_seq_tensor.shape)
    print(word_seq_lengths.shape)
    print(char_seq_tensor.shape)
    print(label_seq_tensor.shape)

    feature_seq_tensor = batchify_feature(
        char_seq_tensor=char_seq_tensor, word_seq_lengths=word_seq_lengths,
        pos_list=pos_list)
    print(feature_seq_tensor.shape)
    print(feature_seq_tensor)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    When used: batches = batch_iter(list(zip(train_x, train_y)), batch_size, epoch)
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            # If x is an integer, randomly permute np.arange(x). If x is an array, make a copy and shuffle the elements randomly.
            # np.random.permutation(10)
            # array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def id2rawlabel(ner_vocab, tags):
    """map between id and label"""
    return [ner_vocab[t] for t in tags]


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def plot_fig(acc, loss, step_data, model_save_dir):

    if len(step_data) != len(loss) or len(step_data) != len(acc):
        raise ValueError('Plot: length of x and y are not the same')

    fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.plot(step_data, loss, linewidth=3)
    ax1.set_title('train_loss')

    ax2.plot(step_data, acc, linewidth=3)
    ax2.set_title('train_acc')
    plt.savefig(model_save_dir+'/train_fig.png', bbox_inches='tight')


if __name__ == '__main__':
    test_feature()

