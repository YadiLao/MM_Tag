# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2018-12-02
class: RL4SL
@author: Lao Yadi
@ main update: label is not part of state, used to calculate value with dot_product
"""

import tensorflow as tf
import numpy as np
import os
from args import Parameter

tf.set_random_seed(1)
config = Parameter().config
print('pv net={}'.format(config))


def create_initializer(initializer_range=0.02):
    """
    Creates a `truncated_normal_initializer` with the given range.
    """
    return tf.truncated_normal_initializer(stddev=initializer_range)


def gelu(input_tensor):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      input_tensor: float Tensor to perform activation.
    Returns:
      `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def fc_layer(inputs, in_dim, out_dim, keep_prob=1.0, initializer_range=0.02):
    """
    fully connected layer
    """
    weight = tf.get_variable(name='fc_u', shape=[in_dim, out_dim],
                             initializer=create_initializer(initializer_range))
    bias = tf.get_variable(name='fc_b', shape=[1, out_dim],
                           initializer=create_initializer(initializer_range))
    # weight = tf.Variable(tf.random_uniform([in_dim, out_dim], -0.1, 0.1), name='fc_u')
    # bias = tf.Variable(tf.zeros([1, out_dim]) + 0.1, name='fc_b')
    wx_plus_b1 = tf.matmul(inputs, weight) + bias
    output = tf.nn.relu(wx_plus_b1)
    # output = tf.nn.dropout(output, keep_prob)
    return output


def rnn_layer(input, dim):
    """
    RNN or LSTM with dropout
    """
    if config['rnn_type'] == 'lstm':
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=dim, reuse=False)
        outputs, states = tf.nn.dynamic_rnn(rnn_cell, input, dtype=tf.float32)
        states = tf.concat([states[0], states[1]], axis=1)

    elif config['rnn_type'] == 'rnn':
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=dim, reuse=False)
        _, states = tf.nn.dynamic_rnn(rnn_cell, input, dtype=tf.float32)

    elif config['rnn_type'] == 'gru':
        print('use gru ...')
        gru_cell = tf.nn.rnn_cell.GRUCell(num_units=dim, reuse=False)
        _, states = tf.nn.dynamic_rnn(cell=gru_cell, inputs=input, dtype=tf.float32)

    elif config['rnn_type'] == 'bilstm':
        fw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=dim, reuse=False)
        bw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=dim, reuse=False)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, input, dtype=tf.float32)

        fw_state = tf.concat(states[0], 1)
        bw_state = tf.concat(states[1], 1)
        states = tf.concat([fw_state, bw_state], axis=1)

    elif config['rnn_type'] == 'bigru':
        print('use bigru ...')
        gru_fw = tf.nn.rnn_cell.GRUCell(num_units=dim)
        gru_bw = tf.nn.rnn_cell.GRUCell(num_units=dim)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            gru_fw, gru_bw, input, dtype=tf.float32)
        fw_state = tf.concat(states[0], 1)
        bw_state = tf.concat(states[1], 1)
        states = tf.concat([fw_state, bw_state], axis=1)

    else:
        states = None
    return states


def init_embed_layer(pretrain_word_embed):
    """
    init word embedding (pretrain or random static) and label embedding (one hot)
    """
    if config['use_pretrain_word_embed']:
        word_embedding = tf.Variable(pretrain_word_embed, name='emb_W',
                                     trainable=config['trainable_embed'])
    else:
        word_embedding = tf.Variable(tf.random_uniform(
            [config['vocab_num'], config['word_embed_dim']], -1.0 / 100, 1.0 / 100),
            name='emb_W', trainable=config['trainable_embed'])

    label_embedding = np.eye(config['label_num'], dtype='float32')
    # print('word_embed={}, label_embed={}'.format(word_embedding, label_embedding))
    return word_embedding, label_embedding


def init_policy_value_param():
    """
    init policy value network param
    """
    w_policy = tf.Variable(tf.random_uniform(
        [config['fc_dim'], config['label_num']], -1. / 100, 1. / 100))

    # TODO: change to from [d,1] to [d,9]
    w_value = tf.Variable(tf.random_uniform(
        [config['fc_dim'], 9], -1. / 100, 1. / 100))

    return w_policy, w_value


def create_train_op(lr):
    """
    Selects the training algorithm and creates a train operation with it
    TODO: lr decay mode
    """
    if config['optim_type'] == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(lr)
    elif config['optim_type'] == 'adam':
        optimizer = tf.train.AdamOptimizer(lr)
    elif config['optim_type'] == 'rprop':
        optimizer = tf.train.RMSPropOptimizer(lr)
    elif config['optim_type'] == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr)
    else:
        raise NotImplementedError('Unsupported optimizer: {}'.format(config['optim_type']))

    return optimizer


class PvnetWithFeature(object):
    """
    policy value network to guide mcts
    """

    def __init__(self, pretrain_word_embed=None, seed=0, path=None):
        print(">>>>>>>>>>>> init model >>>>>>>>>>>")
        self.config = config
        self.seed = seed
        self.pretrain_word_embed = pretrain_word_embed
        self.path = path
        self.current_epoch = tf.Variable(0, trainable=False)
        self.model = self.build_net()
        self.session = tf.Session()
        self.saver = tf.train.Saver()

        if self.path is None:
            self.session.run(tf.global_variables_initializer())
            self.session.run(tf.local_variables_initializer())
        else:
            print('restore model from {}'.format(self.path))
            self.saver.restore(self.session, tf.train.latest_checkpoint(self.path))

    def build_net(self, reuse_word_state=False):
        """
        PVnet Graph
        state = [fw_word, bw_word, extra_fea]
        fw_word, bw_word: [b, t, d]
        tag_label: [b, label_num]

        """
        fw_word = tf.placeholder(tf.int32, [None, None], name='fw_word')
        bw_word = tf.placeholder(tf.int32, [None, None], name='bw_word')
        extra_fea = tf.placeholder(
            tf.float32, [None, self.config['raw_feature_dim']], name='extra_fea')
        batch_size = tf.placeholder(tf.int32, name='batch_size')
        fw_shape = tf.shape(fw_word)

        real_P = tf.placeholder(tf.float32, [None, self.config['label_num']], name='real_P')
        real_V = tf.placeholder(tf.float32, [None, 1], name='real_V')

        # become current tag label. tag_label =[b*t, label_num]
        tag_label = tf.placeholder(
            tf.float32, [None, self.config['label_num']], name='tag_label')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        learning_rate = tf.Variable(self.config['lr'], dtype=tf.float32)

        # embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embed_W, embed_L = init_embed_layer(self.pretrain_word_embed)
            fw_embed_word = tf.nn.embedding_lookup(embed_W, fw_word)
            bw_embed_word = tf.nn.embedding_lookup(embed_W, bw_word)
            print('fw_embed={}, bw_embed={}'.format(fw_embed_word, bw_embed_word))

        # word state layer
        if reuse_word_state:
            with tf.variable_scope("word_state", reuse=None):
                fw_word_state = rnn_layer(fw_embed_word, dim=config['hidden_dim'])
            with tf.variable_scope("word_state", reuse=True):
                bw_word_state = rnn_layer(bw_embed_word, dim=config['hidden_dim'])
        else:
            with tf.variable_scope("fw_state"):
                fw_word_state = rnn_layer(fw_embed_word, dim=config['hidden_dim'])
            with tf.variable_scope("bw_state"):
                bw_word_state = rnn_layer(bw_embed_word, dim=config['hidden_dim'])

        print('lstm, fw={}, bw={}'.format(fw_word_state, bw_word_state))
        final_state = tf.concat(
            [fw_word_state, bw_word_state, extra_fea], axis=1)
        print('final_state={}'.format(final_state))

        with tf.variable_scope("fc", reuse=None):
            #  final_state --> [b*t, d]
            fc_output = fc_layer(
                final_state, self.config['feature_dim'], self.config['fc_dim'], keep_prob)

        with tf.name_scope('pv_net'):
            w_policy, w_value = init_policy_value_param()
            print('w_policy={}, w_value={}'.format(w_policy, w_value))

            logits = tf.matmul(fc_output, w_policy)    # [b*t, label_num]
            prob = tf.nn.softmax(logits)
            max_prob_id = tf.argmax(prob, axis=1)

            value_state = tf.matmul(fc_output, w_value)   # [b*t, label_num]
            # similarity = [b, t]
            similarity = tf.expand_dims(
                tf.reduce_sum(tf.multiply(value_state, tag_label), 1), -1)
            print('value_state={}, sim={}'.format(value_state, similarity))

            if self.config['activate_fun'] == 'relu':
                value = tf.nn.relu(similarity)
            else:
                value = tf.sigmoid(similarity)
            print('prob={}, value={}'.format(prob, value))

            # ***********  calculate loss **************
            value_loss = tf.square(real_V - value)   # [b*t, 1]
            value_loss = tf.reshape(value_loss, [batch_size, -1])
            value_loss = tf.expand_dims(tf.reduce_sum(value_loss, axis=1), -1)

            prob_loss = tf.expand_dims(tf.reduce_sum(tf.multiply(real_P, tf.log(tf.clip_by_value(prob, 1e-30, 1.0))), 1), -1)
            prob_loss = tf.expand_dims(tf.reduce_sum(tf.reshape(prob_loss, [batch_size, -1]), 1), -1)

            # L2 penalty (regularization)
            if self.config['use_l2']:
                l2_penalty_beta = 1e-6
                vars = tf.trainable_variables()
                l2_penalty = l2_penalty_beta * tf.add_n(
                    [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
                loss = value_loss - prob_loss + l2_penalty
            else:
                loss = value_loss - prob_loss

            # assign_lr = tf.assign(learning_rate, learning_rate.value() * 0.9)
            # optimizer = create_train_op(learning_rate).minimize(loss)
            optimizer = create_train_op(learning_rate).minimize(
                loss, global_step=self.current_epoch)

            print('final state = {}'.format(final_state))
            print('logit={}, value ={}'.format(logits, value))
            print('value loss ={}, prob_loss ={}'.format(value_loss, prob_loss))
            print('loss ={}'.format(loss))

        print('trainable variables are ...')
        ws = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES)
        for v in ws:
            print(v.name)

        return {'fw_word': fw_word, 'bw_word': bw_word, 'tag_label': tag_label,
                'extra_fea': extra_fea,  'keep_prob': keep_prob, 'batch_size': batch_size,
                'real_V': real_V, "real_P": real_P,
                'learning_rate': learning_rate, 'fc_output': fc_output,
                'prob': prob, 'max_prob_id': max_prob_id,'value': value,
                'optimizer': optimizer, 'w_policy': w_policy, 'w_value': w_value,
                'prob_loss': prob_loss, 'value_loss': value_loss, 'loss': loss}

    def update_model(self, fw_word, bw_word, label, extra_fea, real_p, real_v, batch_size):
        '''
        print('fw shape={}'.format(np.array(feed_dict[0]).shape))
        print('bw shape={}'.format(np.array(feed_dict[1]).shape))
        print('label shape={}'.format(np.array(feed_dict[2]).shape))
        print('V shape={}'.format(np.array(feed_dict[3]).shape))
        print('P shape={}'.format(np.array(feed_dict[4]).shape))
        '''

        pad_feed_dict = {self.model['fw_word']: fw_word,
                         self.model['bw_word']: bw_word,
                         self.model['tag_label']: label,
                         self.model['extra_fea']: extra_fea,
                         self.model['real_P']: real_p,
                         self.model['real_V']: real_v,
                         self.model['keep_prob']: self.config['keep_prob'],
                         self.model['batch_size']: batch_size
                         }

        opt, loss, p_loss, v_loss = self.session.run(
            [self.model['optimizer'], self.model['loss'],
             self.model['prob_loss'], self.model['value_loss']],
            feed_dict=pad_feed_dict)
        return loss, p_loss, v_loss

    def get_loss(
            self, fw_word, bw_word, label, extra_fea, real_p, real_v, keep_prob, batch_size):
        '''
        print('fw shape={}'.format(np.array(feed_dict[0]).shape))
        print('bw shape={}'.format(np.array(feed_dict[1]).shape))
        print('label shape={}'.format(np.array(feed_dict[2]).shape))
        print('V shape={}'.format(np.array(feed_dict[3]).shape))
        print('P shape={}'.format(np.array(feed_dict[4]).shape))
        '''

        pad_feed_dict = {self.model['fw_word']: fw_word,
                         self.model['bw_word']: bw_word,
                         self.model['tag_label']: label,
                         self.model['extra_fea']: extra_fea,
                         self.model['real_P']: real_p,
                         self.model['real_V']: real_v,
                         self.model['keep_prob']: keep_prob,
                         self.model['batch_size']: batch_size
                         }

        loss, p_loss, v_loss = self.session.run(
            [self.model['loss'], self.model['prob_loss'],
             self.model['value_loss']], feed_dict=pad_feed_dict)
        return loss, p_loss, v_loss

    def get_value(self, fw_word, bw_word, extra_fea, label, keep_prob):
        """
        get value by feeding fc_output_copy and label
        """
        pad_feed_dict = {self.model['fw_word']: fw_word,
                         self.model['bw_word']: bw_word,
                         self.model['extra_fea']: extra_fea,
                         self.model['tag_label']: label,
                         self.model['keep_prob']: keep_prob
                         }
        value = self.session.run(
            self.model['value'], feed_dict=pad_feed_dict)
        return value

    def get_policy(self, fw_word, bw_word, extra_fea, keep_prob):
        """
        get policy by feeding fc_output_copy
        """
        # perform padding first
        pad_feed_dict = {self.model['fw_word']: fw_word,
                         self.model['bw_word']: bw_word,
                         self.model['extra_fea']: extra_fea,
                         self.model['keep_prob']: keep_prob
                         }
        prob = self.session.run(
            self.model['prob'], feed_dict=pad_feed_dict)

        return prob

    def get_fc_output(self, fw_word, bw_word, extra_fea, keep_prob):
        # perform padding first
        pad_feed_dict = {self.model['fw_word']: fw_word,
                         self.model['bw_word']: bw_word,
                         self.model['extra_fea']: extra_fea,
                         self.model['keep_prob']: keep_prob
                         }
        # print(f'fw={fw_word}, bw={bw_word}, label={label}')
        fc_output = self.session.run(
            self.model['fc_output'], feed_dict=pad_feed_dict)

        return fc_output


    def get_lr(self):
        """
        get learning rate
        """
        lr = self.session.run(self.model['learning_rate'])
        return lr

    def _save(self, model_dir):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.session, os.path.join(model_dir))

    def _restore(self):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.session, tf.train.latest_checkpoint(self.path))


