import numpy as np
import tensorflow as tf
from nn import *
import pandas as pd
from plot import plot
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

class Unbias():
    def __init__(self, params, vocab, sgt_W, hate_W, off_W):
        for key in params:
            setattr(self, key, params[key])
        self.vocab = vocab
        self.params = params
        self.embeddings = load_embedding(self.vocab,
                                         "/home/aida/Data/word_embeddings/"
                                         "GloVe/glove.6B.300d.txt",
                                         300)
        self.sgt_W = sgt_W
        self.hate_W = hate_W
        self.off_W = off_W
        self.adversarial_build()



    def adversarial_build(self):
        tf.reset_default_graph()
        self.keep_prob = tf.placeholder(tf.float32)
        self.X = tf.placeholder(tf.int32,
                                            [None, None],
                                            name="inputs")
        self.X_len = tf.placeholder(tf.int32,
                                              [None],
                                              name="sequence_len")
        self.SGT_onehot = tf.placeholder(tf.float32, [None, self.num_SGT + 1], name="SGT_onehot")
        self.SGT_weights = tf.placeholder(tf.float32, [None], name="SGT_weights")


        self.y_off = tf.placeholder(tf.int64, [None], name="offensive_labels")
        self.y_hate = tf.placeholder(tf.int64, [None], name="hate_labels")
        self.y_SGT = tf.placeholder(tf.int64, [None], name="SGT_labels")

        embedding_W = tf.Variable(tf.constant(0.0,
                                        shape=[len(self.vocab), 300]),
                                        trainable=False, name="Embed")

        self.embedding_placeholder = tf.placeholder(tf.float32,
                                                    shape=[len(self.vocab), 300])

        self.embedding_init = embedding_W.assign(self.embedding_placeholder)

        # [batch_size, sent_length, emb_size]
        self.encoder_input_embed = tf.nn.embedding_lookup(self.embedding_placeholder,
                                                          self.X)

        # encoder
        with tf.variable_scope("encode", reuse=tf.AUTO_REUSE):
            fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.h_size, name="ForwardRNNCell",
                                              state_is_tuple=False)
            bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.h_size, name="BackwardRNNCell",
                                              state_is_tuple=False, reuse=False)
            _, self.states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                        self.encoder_input_embed,
                                                        dtype=tf.float32,
                                                        sequence_length=self.X_len)
            self.H = tf.concat(self.states, 1)
        #self.offensive_size = self.hidden_size - self.SGT_size
        #self.offensive = tf.slice(self.representation,
        #                    begin=[0, 0],
        #                    size=[shape[0], self.offensive_size])

        self.task = {"offensive": self.classify(self.H, "offensive",
                                                num_labels=2,
                                                labels=self.y_off,
                                                scope_name="task",
                                                weights=self.off_W),
                     "SGT_off": self.classify(self.H, "SGT",
                                              weights=self.sgt_W,
                                              num_labels=self.num_SGT + 1,
                                              labels=self.y_SGT,
                                              scope_name="advers")}

        #self.extended_offend = tf.tile(tf.expand_dims(
        #    tf.cast(self.task["offensive"]["predicted"], tf.float32), 1),
        #                               [1, self.SGT_size])

        #self.hate = tf.multiply(self.extended_offend, self.task["SGT"]["logits"])
        #self.SGT = tf.multiply(self.extended_offend, tf.layers.dense(self.SGT_onehot, self.SGT_size,
        #                                                       activation=tf.nn.sigmoid))
        with tf.variable_scope("hate_task", reuse=tf.AUTO_REUSE):
            self.H_SGT = tf.layers.dense(self.SGT_onehot, self.SGT_h_size)

        self.hate = tf.concat([self.H, self.H_SGT], axis=-1)
        self.task["hate"] = self.classify(self.hate, "hate",
                                          num_labels=2,
                                          labels=self.y_hate,
                                          scope_name="hate_task",
                                          weights=self.hate_W)

        self.loss = {
            "all": self.task["offensive"]["loss"] +
                       self.task["hate"]["loss"] -
                       0.1 * self.task["SGT_off"]["loss"],
            "off_sgt": self.task["offensive"]["loss"] -
                      0.05 * self.task["SGT_off"]["loss"]
        }

        #self.hate_step = tf.train.AdamOptimizer(learning_rate=self.hate_learning_rate).minimize(self.task["hate"]["loss"])
        self.steps = {
            "hate": tf.train.AdamOptimizer(learning_rate=self.hate_lr) \
                .minimize(self.task["hate"]["loss"], var_list=tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,scope="hate_task")),

            "off": tf.train.AdamOptimizer(learning_rate=self.off_lr)\
            .minimize(self.task["offensive"]['loss'],
                      var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="task/offensive") +
                               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="encode")
                      ),

            "max_off_sgt": tf.train.AdamOptimizer(learning_rate=self.max_off_sgt_lr)\
            .minimize(-self.loss["off_sgt"], var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                scope="advers")),
            "min_off_sgt": tf.train.AdamOptimizer(learning_rate=self.min_off_sgt_lr)\
            .minimize(self.loss["off_sgt"], var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                scope="task") +
                          tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="encode")),

            "min_all": tf.train.AdamOptimizer(learning_rate=self.min_all_lr)\
            .minimize(self.loss["all"], var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                            scope="advers")),
            "max_all": tf.train.AdamOptimizer(learning_rate=self.max_all_lr)\
            .minimize(self.loss["all"], var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                            scope="hate_task"))
        }


    def classify(self, latent, task_name, num_labels, labels,
                 weights=None, logit_weights=None, scope_name=None):
        task = dict()
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            task["logits"] = tf.layers.dense(latent, num_labels, name=task_name)
            if weights:
                logit_weights = tf.gather(weights, labels)
            task["xentropy"] = tf.losses.sparse_softmax_cross_entropy(
                labels=tf.cast(labels, tf.int32),
                logits=task["logits"], weights=logit_weights if weights else 1.0)
            task["loss"] = tf.reduce_mean(task["xentropy"])
            task["predicted"] = tf.argmax(task["logits"], 1)
            task["accuracy"] = tf.reduce_mean(
            tf.cast(tf.equal(task["predicted"], labels), tf.float32))

        return task

    def feed_dict(self, batch, test=False, predict=False):
        feed_dict = {
            self.X: [t["enc_input"] for t in batch],
            self.X_len: [t["length"] for t in batch],
            self.keep_prob: 1 if test else self.keep_ratio,
            self.embedding_placeholder: self.embeddings,
            self.SGT_onehot: [[0 for i in range(self.num_SGT + 1)] for t in batch]
            }
        for i, t in enumerate(batch):
            try:
                feed_dict[self.SGT_onehot][i][t["SGT"]] = 1
            except Exception:
                if predict:
                    print([self.vocab[r] for r in t["enc_input"]])
                    print(t["SGT"])
                    print(self.num_SGT)
                    exit(1)
        if not predict:
            feed_dict[self.y_hate] = [t["hate"] for t in batch]
            feed_dict[self.y_off] = [t["offensive"] for t in batch]
            feed_dict[self.y_SGT] = [t["SGT"] for t in batch]
        return feed_dict

    def train(self, batches):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as self.sess:
            init.run()
            epoch = 1
            while True:
                epoch_losses = {task: 0 for task in ["off", "hate", "sgt_off", "all"]}
                train_acc = {task: 0 for task in ["off", "hate", "sgt"]}
                test_acc = {task: 0 for task in ["off", "hate", "sgt"]}
                pred = {task: list() for task in ["off", "hate"]}
                labels = {task: list() for task in ["off", "hate"]}

                train_idx, test_idx = train_test_split(np.arange(len(batches)),
                                                       test_size=0.14, shuffle=True)
                train_batches = [batches[i] for i in train_idx]
                test_batches = [batches[i] for i in test_idx]


                for batch in train_batches:
                    s, h, sgt, hsgt = self.sess.run([self.states, self.H, self.H_SGT,
                                                  self.hate], feed_dict=self.feed_dict(batch))
                    if epoch > self.hate_epochs:
                        _, hate_l, hate_a = self.sess.run(
                            [self.steps["hate"], self.task["hate"]["loss"],
                             self.task["hate"]["accuracy"]],
                            feed_dict=self.feed_dict(batch))
                        epoch_losses["hate"] += hate_l
                        train_acc["hate"] += hate_a

                    # training "Offensive - SGT" or "SGT" in every other epoch
                    elif epoch % 2 == 0:
                        _, sgt_off_l, all_l = self.sess.run(
                            [self.steps["max_off_sgt"], self.task["SGT_off"]["loss"],
                             self.loss["off_sgt"]],
                            feed_dict=self.feed_dict(batch))
                        epoch_losses["sgt_off"] += sgt_off_l
                        epoch_losses["all"] += all_l

                    else:
                        _, off_l, off_a = self.sess.run(
                            [self.steps["min_off_sgt"], self.task["offensive"]["loss"],
                             self.task["offensive"]["accuracy"]],
                            feed_dict=self.feed_dict(batch))
                        epoch_losses["off"] += off_l
                        train_acc["off"] += off_a

                for batch in test_batches:
                    off_a_test, hate_a_test = self.sess.run(
                        [self.task["offensive"]["accuracy"],
                         self.task["hate"]["accuracy"]], feed_dict=self.feed_dict(batch, test=True))
                    off_p, hate_p = self.sess.run([self.task["hate"]["predicted"],
                        self.task["offensive"]["predicted"]], feed_dict=self.feed_dict(batch, test=True))
                    pred["off"].extend(off_p)
                    pred["hate"].extend(hate_p)
                    labels["hate"].extend([t["hate"] for t in batch])
                    labels["off"].extend([t["offensive"] for t in batch])
                    test_acc["hate"] += hate_a_test
                    test_acc["off"] += off_a_test

                print("Epoch: ", epoch)
                if epoch > self.hate_epochs:
                    print("Hate loss: %.4f, train: %.4f, test: %.4f" %
                          (epoch_losses["hate"] / len(train_batches),
                           train_acc["hate"] / len(train_batches),
                           test_acc["hate"] / len(test_batches)))

                    print("Hate F1: %.4f, Precision: %.4f, Recall: %.4f" %
                        (f1_score(labels["hate"], pred["hate"]),
                         precision_score(labels["hate"], pred["hate"]),
                         recall_score(labels["hate"], pred["hate"])))

                elif epoch % 2 == 0:
                    print("Adversarial loss: %.4f, Overall loss: %.4f" %
                          (epoch_losses["sgt_off"] / len(train_batches),
                           epoch_losses["all"] / len(train_batches)))

                else:
                    print("Offensive loss: %.4f, train: %.4f, test: %.4f" %
                          (epoch_losses["off"] / len(train_batches),
                           train_acc["off"] / len(train_batches),
                           test_acc["off"] / len(test_batches)))

                    print("Offensive F1: %.4f, Precision: %.4f, Recall: %.4f" %
                        (f1_score(labels["off"], pred["off"]),
                         precision_score(labels["off"], pred["off"]),
                         recall_score(labels["off"], pred["off"])))
                epoch += 1
                if epoch == self.epochs:
                    saver.save(self.sess, "saved_model/adversary/hate")
                    break


    def predict_hate(self, batches, labels):
        saver = tf.train.Saver()
        predicted = {label: list() for label in labels}
        with tf.Session() as self.sess:
            saver.restore(self.sess, "saved_model/adversary/hate")
            for batch in batches:
                for label in labels:
                    predicted[label].extend(list(self.sess.run(
                        self.task[label]["predicted"],
                        feed_dict=self.feed_dict(batch, True, True))))
        return predicted
