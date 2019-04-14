# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2018/4/23
class: test for mmtag
@author: Lao Yadi
"""

from tools import *
import numpy as np


class eval_batch:
    """Base class for evaluation, provide method to calculate f1 score and accuracy

    args:
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
    """

    def __init__(self, sen_list, label_list, pred_label_list, reverse_ner_vocab):
        self.sen_list = sen_list
        self.label_list = label_list
        self.pred_label_list = pred_label_list
        self.reverse_ner_vocab = reverse_ner_vocab
        self.reset()

    def reset(self):
        """
        re-set all states
        """
        self.correct_labels = 0
        self.total_labels = 0
        self.gold_count = 0
        self.guess_count = 0
        self.overlap_count = 0

    def calc_f1_batch(self, raw_sen, decoded_data, target_data):
        """
        update statics for f1 score

        args:
            decoded_data (batch_size, seq_len): prediction sequence
            target_data (batch_size, seq_len): ground-truth
        """
        for sen, decoded, target in zip(raw_sen, decoded_data, target_data):
            # print('decode={}, target={}'.format(decoded, target))

            length = len(decoded)
            assert len(decoded) == len(target)
            gold = target[:length]
            best_path = decoded[:length]

            correct_labels_i, total_labels_i, gold_count_i, guess_count_i, overlap_count_i = self.eval_instance(sen, best_path, gold)
            self.correct_labels += correct_labels_i
            self.total_labels += total_labels_i
            self.gold_count += gold_count_i
            self.guess_count += guess_count_i
            self.overlap_count += overlap_count_i

        print('correct_label ={}, total_labels={}, gold_count={}, guess_count={}, overlap_count={}'.format(self.correct_labels, self.total_labels, self.gold_count, self.guess_count, self.overlap_count))

    def calc_acc_batch(self, decoded_data, target_data):
        """
        update statics for accuracy

        args:
            decoded_data (batch_size, seq_len): prediction sequence
            target_data (batch_size, seq_len): ground-truth
        """

        for decoded, target in zip(decoded_data, target_data):
            assert len(decoded) == len(target)

            self.total_labels += len(decoded)
            self.correct_labels += np.sum(np.equal(decoded, target))

    def f1_score(self):
        """
        calculate f1 score based on statics
        """
        if self.guess_count == 0:
            return 0.0, 0.0, 0.0, 0.0
        precision = self.overlap_count / float(self.guess_count)
        recall = self.overlap_count / float(self.gold_count)
        if precision == 0.0 or recall == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        f = 2 * (precision * recall) / (precision + recall)
        accuracy = float(self.correct_labels) / self.total_labels
        return f, precision, recall, accuracy

    def acc_score(self):
        """
        calculate accuracy score based on statics
        """
        if 0 == self.total_labels:
            return 0.0
        accuracy = float(self.correct_labels) / self.total_labels
        return accuracy

    def eval_instance(self, sen, best_path, gold):
        """
        update statics for one instance

        args:
            best_path (seq_len): predicted
            gold (seq_len): ground-truth
        """
        total_labels = len(best_path)
        correct_labels = np.sum(np.equal(best_path, gold))
        gold_chunks = bio_to_spans(gold, self.reverse_ner_vocab)
        gold_count = len(gold_chunks)

        guess_chunks = bio_to_spans(best_path, self.reverse_ner_vocab)
        guess_count = len(guess_chunks)

        overlap_chunks = gold_chunks & guess_chunks
        overlap_count = len(overlap_chunks)
        print('sen={}|len={}|acc={}|gold_chunks={}|guess_chunks={}'.format(
            ' '.join([str(s) for s in sen]), total_labels,
            correct_labels*1.0/total_labels,gold_chunks,guess_chunks))

        return correct_labels, total_labels, gold_count, guess_count, overlap_count

    def calc_score(self, raw_sen, y_true, y_pred):
        self.calc_f1_batch(raw_sen=raw_sen, decoded_data=y_pred, target_data=y_true)

        return self.f1_score()

