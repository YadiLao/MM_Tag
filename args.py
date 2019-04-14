# !/usr/bin/python
# -*- coding:utf-8 -*-


class Parameter(object):
    def __init__(self):
        self.config = {
            'label_num': 9,
            'vocab_num': 402596,
            'word_embed_dim': 100,
            'hidden_dim': 200,
            'fc_dim': 128,
            'feature_dim': 400 + 4*5 + 172+225,  # 'pos 45*5, chunk 21*5 word 4*5'
            'raw_feature_dim': 4*5 + 172+225,
            'n_gram': 2,
            'context_only': True,

            'lr': 0.015,                # init set to 0.01
            'optim_type': 'adagrad',
            'activate_fun': 'sigmoid',  # sigmoid or relu
            'use_rnn': True,
            'rnn_type': 'gru',
            'concate': False,           # mean_sum
            'use_pretrain_word_embed': True,
            'trainable_embed': True,
            'use_l2': False,
            'keep_prob': 1.0,

            'look_ahead_depth': None,
            'epoch': 200,
            'n_playout': 600,
            'c_puct': 1.0,
            'reward_type': 'f1',  # f1 or acc
            'use_gain': False,
            'discount_factor': 1.0,

            'batch_size': 10,
            'tag_format': 'IOB',
            'checkpoint_dir': './debug_runs/2019-01-06'

        }
