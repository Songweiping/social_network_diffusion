#!/usr/bin/env python
# coding=utf-8
import sys
import random
import numpy as np
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string("train_data", None, "training file.")
flags.DEFINE_string("test_data", None, "testing file.")
flags.DEFINE_string("save_path", None, "path to save the model.")
flags.DEFINE_integer("max_epoch", 10, "max epochs.")
flags.DEFINE_integer("emb_dim", 100, "embedding dimension.")
flags.DEFINE_float("lr", 0.025, "initial learning rate.")

FLAGS = flags.FLAGS


class Options(object):
    """options used by CDK model."""
    
    def __init__(self):
        #model options.

        #embedding dimension.
        self.emb_dim = FLAGS.emb_dim

        #train file path.
        self.train_data = FLAGS.train_data

        #test file path.
        self.test_data = FLAGS.test_data

        #save path.
        self.save_path = FLAGS.save_path

        #max epoch.
        self.max_epoch = FLAGS.max_epoch

        #initial learning rate.
        self.lr = FLAGS.lr



class CDK(object):
    """Basic diffusion kernel model."""

    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._u2idx = {}
        self._train_cascades = self.readFromFile(options.train_data)
        self._test_cascades = self.readFromFile(options.test_data)
        self._options.train_size = len(self._train_cascades)
        self._options.test_size = len(self._test_cascades)
        self._options.samples_to_train = self._options.max_epoch * self.options.train_size
        self.buildGraph()

    def _buildIndex(self):
        #compute an index of the users that appear at least once in the training and testing cascades.
        opts = self._options

        train_user_set = set()
        test_user_set = set()

        for line in open(opts.train_data):
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                user, timestamp = chunk.split(',')
                train_user_set.add(user)

        for line in open(opts.test_data):
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                user, timestamp = chunk.split(',')
                test_user_set.add(user)

        user_set = train_user_set & test_user_set

        pos = 1
        for user in user_set:
            self._u2idx[user] = pos
            pos += 1

        self._user_size = len(user_set)

    def _readFromFile(self, filename):
        """read all cascade from training or testing files. """
        t_cascades = []
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            userlist = []
            chunks = line.strip().split()
            for chunk in chunks:
                user, timestamp = chunk.split(',')
                if self._u2idx.has_key(user):
                    userlist.append(self._u2idx[user])

            if len(userlist) > 1:
                t_cascades.append(userlist)

        return t_cascades

    def buildGraph(self):
        opts = self._options
        contaminated1 = tf.placeholder(tf.int32, shape=[1])
        self._contaminated1 = contaminated1
        contaminated2  = tf.placeholder(tf.int32, shape=[1])
        self._contaminated2 = contaminated2
        further = tf.placeholder(tf.int32,shape=[1])
        self._further = further
        
        emb_user = tf.Variable(tf.random_uniform([self._user_size, opts.emb_dim], -1, 1), name="emb_user")

        self.global_step = tf.Variable(0, trainable=False, name="global_step",dtype=tf.int32)
        
        emb_contaminated1 = tf.nn.embedding_lookup(emb_user, contaminated1)
        emb_contaminated2 = tf.nn.embedding_lookup(emb_user, contaminated2)
        emb_further = tf.nn.embedding_lookup(emb_user, further)

        d1 = tf.mul(emb_contaminated1, emb_contaminated2)
        d2 = tf.mul(emb_contaminated1, emb_further)

        #loss = d1 + (1 - d2)
        zero = tf.consant(0.0, dtype=tf.float32, shape=[1])
        one = tf.constant(1.0, dtype=tf.float32, shape=[1])

        loss = tf.add(d1, tf.maximum(zero, tf.sub(one, d2)))

        self._lr = opts.lr * (1.0 - tf.cast(self.global_step, tf.float32) / float(opts.samples_to_train))  
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        train = optimizer.minimize(loss,
                                  global_step = self.global_step,
                                  gate_gradients = optimizer.GATE_NONE)
        self._train = train

    def SampleUsers():
        






