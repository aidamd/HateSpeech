import numpy as np
import tensorflow as tf
from nn import *
import pandas as pd
from plot import plot
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

class Context():
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

        self.drop_prob = tf.placeholder(tf.float32)

        # Posts are inputs of variable length
        self.X = tf.placeholder(tf.int32,[None, None], name="inputs")
        self.X_len = tf.placeholder(tf.int32, [None], name="sequence_len")

        # SGTs are defined as multi-label
        self.y_SGT = tf.placeholder(tf.float32, [None, self.num_SGT],
                                    name="SGT_multi")
        self.SGT_weights = tf.placeholder(tf.float32, [None],
                                          name="SGT_weights")

        # Context is also defined as multi-label
        self.y_context = tf.placeholder(tf.int64, [None, self.num_context],
                                        name="Context_multi")
        self.context_weights = tf.placeholder(tf.float32, [None],
                                              name="context_weights")

        # hate is a binary label
        self.y_hate = tf.placeholder(tf.int64, [None],
                                     name="hate_labels")

        embedding_W = tf.Variable(tf.constant(0.0,
                                        shape=[len(self.vocab), 300]),
                                        trainable=False, name="Embed")

        self.embedding_placeholder = tf.placeholder(tf.float32,
                                                    shape=[len(self.vocab), 300])

        self.embedding_init = embedding_W.assign(self.embedding_placeholder)

        # [batch_size, sent_length, emb_size]
        self.encoder_input_embed = tf.nn.embedding_lookup(
            self.embedding_placeholder,self.X)

        # encoder
        with tf.variable_scope("encode", reuse=tf.AUTO_REUSE):
            fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.h_size,
                                              name="ForwardRNNCell",
                                              state_is_tuple=False)
            bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.h_size,
                                              name="BackwardRNNCell",
                                              state_is_tuple=False,
                                              reuse=False)
            _, self.states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                        self.encoder_input_embed,
                                                        dtype=tf.float32,
                                                        sequence_length=self.X_len)
            # the sentence representation
            self.H = tf.nn.dropout(tf.concat(self.states, 1), self.drop_prob)

        # defining 3 fully connected layers to predict hate, context and SGT
        self.task = {"context": self.classify(self.H, "context",
                                              num_labels=self.num_context,
                                              labels=self.y_context,
                                              scope_name="task",
                                              weights=self.context_W,
                                              multi=True),
                     "hate": self.classify(self.H, "hate",
                                           num_labels=2,
                                           labels=self.y_hate,
                                           scope_name="task",
                                           weights=self.hate_W),
                     "SGT": self.classify(self.H, "SGT",
                                          weights=self.sgt_W,
                                          num_labels=self.num_SGT,
                                          labels=self.y_SGT,
                                          scope_name="advers",
                                          multi=True)}

        self.loss = self.task["hate"]["loss"] + \
                    self.task["context"]["loss"] - \
                    self.adv_coeff * self.task["SGT"]["loss"]

        self.steps = {
            "hate": tf.train.AdamOptimizer(learning_rate=self.hate_lr) \
                .minimize(self.loss, var_list=tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,scope="task")),

            "adverse": tf.train.AdamOptimizer(learning_rate=self.adv_lr)\
            .minimize(-self.loss,
                      var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="advers")
                      )
        }


    def classify(self, latent, task_name, num_labels, labels,
                 weights=None, logit_weights=None, scope_name=None, multi=False):
        task = dict()
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            task["logits"] = tf.layers.dense(latent, num_labels, name=task_name)
            if weights:
                logit_weights = tf.gather(weights, labels)
            if multi:
                task["xentropy"] = tf.nn.weighted_cross_entropy_with_logits(
                    tf.cast(labels, tf.int32), logits=task["logits"],
                    pos_weight=logit_weights if weights else 1.0)
                task["predicted"] = tf.cast(tf.to_int32(tf.sigmoid(task["logits"]) > 0.5),
                                            tf.float32)
                # here we only need the loss of context and SGT for positive instances of hate
                # based on the equality of opportunity
                task["loss"] = tf.reduce_mean(tf.multiply(task["xentropy"], self.y_hate))
            else:
                task["xentropy"] = tf.losses.sparse_softmax_cross_entropy(
                labels=tf.cast(labels, tf.int32),
                logits=task["logits"], weights=logit_weights if weights else 1.0)
                task["predicted"] = tf.argmax(task["logits"], 1)
                task["loss"] = tf.reduce_mean(task["xentropy"])

            task["accuracy"] = tf.reduce_mean(
            tf.cast(tf.equal(task["predicted"], labels), tf.float32))

        return task

    def feed_dict(self, batch, test=False, predict=False):
        feed_dict = {
            self.X: [t["enc_input"] for t in batch],
            self.X_len: [t["length"] for t in batch],
            self.drop_prob: 1 if test else self.drop_ratio,
            self.embedding_placeholder: self.embeddings,
            self.SGT: [[0 for i in range(self.num_SGT + 1)] for t in batch]

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
                epoch_losses = {task: 0 for task in ["hate", "sgt", "context", "all"]}
                train_acc = {task: 0 for task in ["hate", "sgt", "context"]}
                test_acc = {task: 0 for task in ["hate", "sgt", "context"]}
                pred = {task: list() for task in ["hate", "sgt"]}
                labels = {task: list() for task in ["hate"]}

                train_idx, test_idx = train_test_split(np.arange(len(batches)),
                                                       test_size=0.14, shuffle=True)
                train_batches = [batches[i] for i in train_idx]
                test_batches = [batches[i] for i in test_idx]


                for batch in train_batches:
                    if epoch > self.hate_epochs:
                        _, hate_l, hate_a = self.sess.run(
                            [self.steps["hate"], self.task["hate"]["loss"],
                             self.task["hate"]["accuracy"]],
                            feed_dict=self.feed_dict(batch))
                        epoch_losses["hate"] += hate_l
                        train_acc["hate"] += hate_a

                    # training "Hate + Context - SGT" or "Hate" in every other epoch
                    elif epoch % 2 == 0:
                        _, sgt_l, con_l = self.sess.run(
                            [self.steps["adverse"], self.task["SGT"]["loss"],
                             self.loss["context"]],
                            feed_dict=self.feed_dict(batch))
                        epoch_losses["sgt"] += sgt_l
                        epoch_losses["context"] += con_l

                    else:
                        _, hate_l, hate_a = self.sess.run(
                            [self.steps["hate"], self.task["hate"]["loss"],
                             self.task["hate"]["accuracy"]],
                            feed_dict=self.feed_dict(batch))
                        epoch_losses["hate"] += hate_l
                        train_acc["hate"] += hate_a

                for batch in test_batches:
                    off_a_test, hate_a_test = self.sess.run(
                        [self.task["sgt"]["accuracy"],
                         self.task["hate"]["accuracy"],
                         self.task["context"]["accuracy"]],
                        feed_dict=self.feed_dict(batch, test=True))

                    off_p, hate_p = self.sess.run(
                        [self.task["hate"]["predicted"],
                        self.task["SGT"]["predicted"]],
                        feed_dict=self.feed_dict(batch, test=True))

                    """
                    pred["off"].extend(off_p)
                    pred["hate"].extend(hate_p)
                    labels["hate"].extend([t["hate"] for t in batch])
                    labels["off"].extend([t["offensive"] for t in batch])
                    test_acc["hate"] += hate_a_test
                    test_acc["off"] += off_a_test
                    """
                """
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
                """
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
