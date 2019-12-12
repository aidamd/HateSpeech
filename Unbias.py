import numpy as np
import tensorflow as tf
from nn import *
import pandas as pd
from plot import plot
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

class Unbias():
    def __init__(self, params, vocab, SGT_weights):
        self.vocab = vocab

        self.params = params
        for key in params:
            setattr(self, key, params[key])
        self.embeddings = load_embedding(self.vocab,
                                         "/home/aida/Data/word_embeddings/GloVe/glove.6B.300d.txt",
                                         self.embedding_size)
        self.adversarial_build()
        self.weights = SGT_weights


    def adversarial_build(self):
        tf.reset_default_graph()
        self.keep_prob = tf.placeholder(tf.float32)

        #[batch_size, length]
        self.encoder_input = tf.placeholder(tf.int32, [None, None], name="encode_inputs")
        self.sequence_length = tf.placeholder(tf.int32, [None], name="seq_len")

        # [batch_size]
        self.offensive_label = tf.placeholder(tf.int64, [None], name="offensive_labels")
        self.hate_label = tf.placeholder(tf.int64, [None], name="hate_labels")
        self.SGT_label = tf.placeholder(tf.int64, [None], name="SGT_labels")
        self.SGT_onehot = tf.placeholder(tf.float32, [None, self.num_SGT + 1], name="SGT_onehot")
        self.SGT_weights = tf.placeholder(tf.float32, [None], name="SGT_weights")

        emb_W = tf.Variable(tf.constant(0.0,
                                        shape=[len(self.vocab), self.embedding_size]),
                                        trainable=False, name="Embed")
        self.embedding_placeholder = tf.placeholder(tf.float32,
                                                    shape=[len(self.vocab),
                                                           self.embedding_size])

        self.embedding_init = emb_W.assign(self.embedding_placeholder)

        # [batch_size, sent_length, emb_size]
        self.encoder_input_embed = tf.nn.embedding_lookup(self.embedding_placeholder,
                                                          self.encoder_input)

        # encoder
        fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, name="ForwardRNNCell",
                                          state_is_tuple=False)
        bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, name="BackwardRNNCell",
                                          state_is_tuple=False, reuse=False)
        _, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                    self.encoder_input_embed,
                                                    dtype=tf.float32,
                                                    sequence_length=self.sequence_length,)
        self.representation = tf.concat(states, 1)
        shape = tf.shape(self.encoder_input)

        self.offensive_size = self.hidden_size - self.SGT_size
        #self.offensive = tf.slice(self.representation,
        #                    begin=[0, 0],
        #                    size=[shape[0], self.offensive_size])

        self.task = {"offensive": self.classify(self.representation, "offensive",
                                                num_labels=2,
                                                labels=self.offensive_label,
                                                scope_name="task"),
                     "SGT_off": self.classify(self.representation, "SGT",
                                             num_labels=self.num_SGT + 1,
                                             labels=self.SGT_label,
                                             scope_name="advers")}

        #print(self.num_SGT)
        self.extended_offend = tf.tile(tf.expand_dims(
            tf.cast(self.task["offensive"]["predicted"], tf.float32), 1),
                                       [1, self.SGT_size])

        #self.hate = tf.multiply(self.extended_offend, self.task["SGT"]["logits"])
        self.SGT = tf.multiply(self.extended_offend, tf.layers.dense(self.SGT_onehot, self.SGT_size,
                                                               activation=tf.nn.sigmoid))
        self.hate = tf.concat([self.representation, self.SGT], axis=-1)
        self.task["hate"] = self.classify(self.hate, "hate",
                                          num_labels=2,
                                          labels=self.hate_label,
                                          scope_name="hate_task")

        #self.hate_step = tf.train.AdamOptimizer(learning_rate=self.hate_learning_rate).minimize(self.task["hate"]["loss"])
        self.loss = self.task["offensive"]["loss"] + self.task["hate"]["loss"] - 0.1 * self.task["SGT_off"]["loss"]
        self.off_loss = self.task["offensive"]["loss"] - 0.1 * self.task["SGT_off"]["loss"]

        self.off_step = tf.train.AdamOptimizer(learning_rate=self.offensive_learning_rate)\
            .minimize(self.task["offensive"]['loss'])
        self.first_min_step = tf.train.AdamOptimizer(learning_rate=self.hate_learning_rate)\
            .minimize(self.off_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="task"))
        self.first_max_step = tf.train.AdamOptimizer(learning_rate=self.SGT_off_learning_rate)\
            .minimize(self.off_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="advers"))
        self.min_step = tf.train.AdamOptimizer(learning_rate=self.hate_learning_rate)\
            .minimize(self.loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="task"))
        self.max_step = tf.train.AdamOptimizer(learning_rate=self.SGT_off_learning_rate)\
            .minimize(self.loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="advers"))
        self.hate_step = tf.train.AdamOptimizer(learning_rate=self.hate_learning_rate)\
            .minimize(self.task["hate"]["loss"], var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="hate_task"))

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
            self.encoder_input: [t["enc_input"] for t in batch],
            self.sequence_length: [t["length"] for t in batch],
            self.keep_prob: 1 if test else self.keep_ratio,
            self.embedding_placeholder: self.embeddings,
            self.SGT_weights: self.weights,
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
            feed_dict[self.hate_label] = [t["hate"] for t in batch]
            feed_dict[self.offensive_label] = [t["offensive"] for t in batch]
            feed_dict[self.SGT_label] = [t["SGT"] for t in batch]

        return feed_dict

    def train(self, batches):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as self.sess:
            init.run()
            epoch = 1
            losses = {"offensive": list(),
                      "hate": list(),
                      "SGT": list(),
                      "SGT_off": list(),
                      "task": list()}

            hate_accuracy = {"train": list(),
                             "test": list()}
            offensive_accuracy = {"train": list(),
                                  "test": list()}

            while True:
                off_loss, hate_loss, sgt_loss, sgt_off_loss, task_loss = 0, 0, 0, 0, 0
                off_acc, hate_acc, sgt_acc = 0, 0 ,0
                off_acc_test, hate_acc_test, sgt_acc_test = 0, 0 ,0
                train_idx, test_idx = train_test_split(np.arange(len(batches)), test_size=0.14, shuffle=True)
                train_batches = [batches[i] for i in train_idx]
                test_batches = [batches[i] for i in test_idx]
                for batch in train_batches:
                    if epoch < self.offensive_epochs:
                        _, off_l = self.sess.run(
                            [self.off_step, self.task["offensive"]["loss"]],
                            feed_dict=self.feed_dict(batch))
                        off_loss += off_l

                    elif epoch > self.hate_epochs:
                        _, hate_l, hate_a = self.sess.run(
                            [self.hate_step, self.task["hate"]["loss"],
                             self.task["hate"]["accuracy"]],
                            feed_dict=self.feed_dict(batch))
                        hate_loss += hate_l
                        hate_acc += hate_a
                        
                    elif epoch % 2 == 0:
                        _, sgt_off_l = self.sess.run(
                            [self.first_max_step, self.task["SGT_off"]["loss"]],
                            feed_dict=self.feed_dict(batch))
                        sgt_off_loss += sgt_off_l

                    else:
                        _, task_l, off_l, hate_l = self.sess.run(
                            [self.first_min_step, self.loss, self.task["offensive"]["loss"],
                             self.task["hate"]["loss"]],
                            feed_dict=self.feed_dict(batch))
                        hate_loss += hate_l
                        task_loss += task_l
                        off_loss += off_l


                hate_pred = []
                off_pred = []
                hate_labels = []
                off_labels = []
                for batch in test_batches:
                    off_a_test, hate_a_test = self.sess.run(
                        [self.task["offensive"]["accuracy"],
                         self.task["hate"]["accuracy"]], feed_dict=self.feed_dict(batch, test=True))
                    off_p, hate_p = self.sess.run([self.task["hate"]["predicted"],
                        self.task["offensive"]["predicted"]], feed_dict=self.feed_dict(batch, test=True))
                    off_pred.extend(off_p)
                    hate_pred.extend(hate_p)
                    hate_labels.extend([t["hate"] for t in batch])
                    off_labels.extend([t["offensive"] for t in batch])
                    hate_acc_test += hate_a_test

                print("Hate train: %.4f, test: %.4f" % (hate_acc / len(train_batches),
                                                         hate_acc_test / len(test_batches)))
                print("Hate F1: %.4f, Precision: %.4f, Recall: %.4f" % 
                    (f1_score(hate_labels, hate_pred), precision_score(hate_labels, hate_pred),
                    recall_score(hate_labels, hate_pred)))
                print("Offensive F1: %.4f, Precision: %.4f, Recall: %.4f" %
                    (f1_score(off_labels, off_pred), precision_score(off_labels, off_pred),
                    recall_score(off_labels, off_pred)))


                if epoch % 2 == 0:
                    print("Epoch: %d, Adversarial loss: %.4f" %
                          (epoch, sgt_off_loss / len(train_batches)))
                    losses["SGT_off"].append(sgt_off_loss / len(train_batches))
                else:
                    print("Epoch: %d, Task loss: %.4f, Hate loss: %.4f, Offensive loss: %.4f" %
                          (epoch, task_loss / len(train_batches),
                           hate_loss / len(train_batches), off_loss / len(train_batches)))
                    losses["hate"].append(hate_loss / len(train_batches))
                    losses["offensive"].append(off_loss / len(train_batches))
                    losses["task"].append(task_loss / len(train_batches))
                    hate_accuracy["train"].append(hate_acc / len(train_batches))
                    offensive_accuracy["train"].append(off_acc / len(train_batches))
                    hate_accuracy["test"].append(hate_acc_test / len(test_batches))
                    offensive_accuracy["test"].append(off_acc_test / len(test_batches))
                epoch += 1

                if epoch == self.epochs:
                    saver.save(self.sess, "saved_model/adversary/hate")
                    #pd.DataFrame.from_dict(losses).to_csv("plots/losses.csv")
                    #pd.DataFrame.from_dict(SGT_accuracy).to_csv("plots/SGT.csv")
                    pd.DataFrame.from_dict(hate_accuracy).to_csv("plots/hate.csv")
                    pd.DataFrame.from_dict(offensive_accuracy).to_csv("plots/offensive.csv")
                    plot()
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
