#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import logging
import random
import numpy as np
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string("train_data", None, "training file.")
flags.DEFINE_string("test_data", None, "testing file.")
flags.DEFINE_string("save_path", None, "path to save the model.")
flags.DEFINE_integer("max_epoch", 10, "max epochs.")
flags.DEFINE_integer("emb_dim", 50, "embedding dimension.")
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
        self._buildIndex()
        self._train_cascades = self._readFromFile(options.train_data)
        self._test_cascades = self._readFromFile(options.test_data)
        self._options.train_size = len(self._train_cascades)
        self._options.test_size = len(self._test_cascades)
        logging.info("training set size:%d    testing set size:%d" % (self._options.train_size, self._options.test_size))
        self._options.samples_to_train = self._options.max_epoch * self._options.train_size
        self.buildGraph()
        self.buildEvalGraph()


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

        #user_set = train_user_set | test_user_set

        pos = 0
        for user in train_user_set:
            self._u2idx[user] = pos
            pos += 1

        opts.user_size = len(train_user_set)
        logging.info("user size : %d" % (opts.user_size))


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
        
        emb_user = tf.Variable(tf.random_uniform([opts.user_size, opts.emb_dim], -1.0, 1.0), name="emb_user")
        self.emb_user = emb_user
        self.global_step = tf.Variable(0, trainable=False, name="global_step",dtype=tf.int32)
        
        emb_contaminated1 = tf.nn.embedding_lookup(emb_user, contaminated1)
        emb_contaminated2 = tf.nn.embedding_lookup(emb_user, contaminated2)
        emb_further = tf.nn.embedding_lookup(emb_user, further)

        d1 = tf.reduce_sum(tf.square(tf.sub(emb_contaminated1, emb_contaminated2)))
        d2 = tf.reduce_sum(tf.square(tf.sub(emb_contaminated1, emb_further)))
        #self.d1 = d1
        #d1 = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(emb_contaminated1, emb_contaminated2))))
        #d2 = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(emb_contaminated1, emb_further))))
        #hinge loss
        zero = tf.constant(0.0, dtype=tf.float32, shape=[1])
        one = tf.constant(1.0, dtype=tf.float32, shape=[1])
        loss = tf.maximum(zero, d1 - d2 + one)
        self._loss = loss
        self._lr = opts.lr * (1.0 - tf.cast(self.global_step, tf.float32) / float(opts.samples_to_train))  
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        train = optimizer.minimize(loss,
                                  global_step = self.global_step,
                                  gate_gradients = optimizer.GATE_NONE)
        self._train = train

        tf.initialize_all_variables().run()

        self.saver = tf.train.Saver()

    def SampleUsers(self, cascade):
        opts = self._options
        """sample three users, u_i,u_j and u_k, t(u_i)<t(u_j), and u_k is not in this cascade or d(u_k) > d(u_j)."""
        contaminated1 = np.ndarray(shape=(1), dtype=np.int32)
        contaminated2 = np.ndarray(shape=(1), dtype=np.int32)
        further = np.ndarray(shape=(1), dtype=np.int32)
        u_i = 0
        contaminated1[0] = cascade[u_i]
        u_j = random.randint(u_i, len(cascade) - 1)
        contaminated2[0] = cascade[u_j]
        u_k = random.randint(0, opts.user_size - 1)
        flag = True
        while flag:
            flag = False
            for i in xrange(u_i, u_j + 1):
                if cascade[i] == u_k:
                    flag = True
                    break
            if flag:
                u_k = random.randint(0, opts.user_size - 1)
        further[0] = u_k
        return contaminated1, contaminated2, further

    def train(self):
        """train the model."""
        opts = self._options
        loss_list = []
        last_count = 0
        last_time = time.time()
        for i in xrange(opts.max_epoch):
            for j in xrange(opts.train_size):
                idx = random.randint(0, opts.train_size - 1)
                cascade = self._train_cascades[idx]
                contaminated1, contaminated2, further = self.SampleUsers(cascade)
                feed_dict = {self._contaminated1:contaminated1, 
                             self._contaminated2:contaminated2, 
                             self._further:further}

                (lr, loss, step, _) = self._session.run([self._lr, self._loss, self.global_step, self._train],
                                                       feed_dict=feed_dict)

                if (step - last_count) > 100:
                    loss_list.append(loss)
                    average_loss = np.mean(np.array(loss_list))
                    progress = 100 * float(step) / float(opts.samples_to_train)
                    now = time.time()
                    rate = float(step - last_count) / (now - last_time)
                    last_time = now
                    print ("learning rate:%f  loss:%f average loss:%f rate:%f progress:%f%%\r" % (
                        lr, loss, average_loss, rate, progress),
                            end="")
                    sys.stdout.flush()
                    last_count = step
        print("")

    def computeDistanceVector(self, user_source):
        """compute euclidean distance matrix among all users."""
        opts = self._options
        nemb = tf.nn.l2_normalize(self.emb_user, 1)
        emb_source = tf.gather(nemb, user_source)
        emb_source_paded = tf.tile(emb_source,[1, opts.user_size])
        emb_source_matrix = tf.reshape(emb_source_paded,[opts.user_size, opts.emb_dim])
        #self.distance_vector = tf.matmul(emb_source, nemb, transpose_b = True, name="distance_vector")
        self.distance_vector = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(emb_source_matrix, nemb)),1))
    
    def computeMAP(self, cascade):
        opts = self._options
        #distance = tf.gather(self.distance_matrix, user_fisrt)
        _, indices = tf.nn.top_k(self.distance_vector, opts.user_size, sorted=True)
        self.indices = indices


    def buildEvalGraph(self):
        """build evaluate graph."""
        cascade = tf.placeholder(tf.int32)
        self.cascade = cascade
        user_source = tf.slice(cascade, [0], [1])
        self.computeDistanceVector(user_source)
        self.computeMAP(cascade)

    def eval(self):
        final_avgp = 0.0
        opts = self._options
        for cascade in self._test_cascades:
            cascade_size = len(cascade)
            #inputs = np.ndarray(shape=(cascade_size), dtype = np.int32)
            indices = self._session.run(self.indices, feed_dict = {self.cascade:cascade})
            nb_positive = 0
            rank = 0
            avgp = 0.0
            user_set = set(cascade)
            while nb_positive < cascade_size:
                if indices[opts.user_size - rank - 1] in user_set:
                    nb_positive += 1
                    pre = float(nb_positive) / float(rank + 1)
                    avgp += pre
                rank += 1
            avgp /= float(cascade_size)
            final_avgp += avgp
        final_avgp /= float(self._options.test_size)
        print ("Average precision:%f" % (final_avgp))

def main(_):
    logging.basicConfig(level = logging.INFO)
    if not FLAGS.train_data:
        logging.error("train file not found.")
        sys.exit(1)
    options = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device("/cpu:0"):
            model = CDK(options, session)
        model.train()
        model.eval()
        model.saver.save(session,
                        os.path.join(os.path.abspath(os.path.dirname(__file__)),options.save_path),
                        global_step = model.global_step)

if __name__ == '__main__':
    tf.app.run()
