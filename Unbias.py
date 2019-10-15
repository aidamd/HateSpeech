import numpy as np
import tensorflow as tf
from nn import *
import pandas as pd
from plot import plot
from sklearn.model_selection import train_test_split 

class Unbias():
    def __init__(self, params, data, vocab, hate, offensive, SGT):
        self.vocab = vocab
        self.data = data

        self.params = params
        for key in params:
            setattr(self, key, params[key])
        self.embeddings = load_embedding(self.vocab,
                                         "/home/aida/Data/word_embeddings/GloVe/glove.6B.300d.txt",
                                         self.embedding_size)
        self.adversarial_build()
        batches = get_batches(self.data,
                              self.batch_size,
                              vocab.index("<pad>"),
                              vocab.index("<eos>"),
                              vocab.index("<go>"),
                              hate,
                              offensive,
                              SGT)
        self.train(batches)


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
        self.SGT = tf.slice(self.representation,
                            begin=[0, self.offensive_size] ,
                            size=[shape[0], self.SGT_size])
        self.offensive = tf.slice(self.representation,
                            begin=[0, 0],
                            size=[shape[0], self.offensive_size])

        self.task = {"offensive": self.classify(self.offensive,
                                                num_labels=2,
                                                labels=self.offensive_label,
                                                learning_rate=self.offensive_learning_rate),
                     "hate": self.classify(self.representation,
                                          num_labels=2,
                                          labels=self.hate_label,
                                          learning_rate=self.hate_learning_rate),
                     "SGT": self.classify(self.SGT,
                                         num_labels=self.num_SGT + 1,
                                         labels=self.SGT_label,
                                         learning_rate=self.SGT_learning_rate),
                     "SGT_off": self.classify(self.offensive,
                                             num_labels=self.num_SGT + 1,
                                             labels=self.SGT_label,
                                             learning_rate=self.SGT_off_learning_rate,
                                             maximize=True)}

    def classify(self, latent, num_labels, labels, learning_rate, maximize=False):
        task = dict()
        task["logits"] = tf.layers.dense(latent, num_labels)
        task["xentropy"] = tf.losses.sparse_softmax_cross_entropy(
            labels=tf.cast(labels, tf.int32),
            logits=task["logits"])

        task["loss"] = tf.reduce_mean(task["xentropy"])
        task["step"] = tf.train.AdamOptimizer(learning_rate=learning_rate)\
            .minimize(-task["loss"] if maximize else task["loss"])

        task["predicted"] = tf.argmax(task["logits"], 1)
        task["accuracy"] = tf.reduce_mean(
            tf.cast(tf.equal(task["predicted"], labels), tf.float32))

        return task

    def feed_dict(self, batch, test=False):
        feed_dict = {
            self.encoder_input: [t["enc_input"] for t in batch],
            self.hate_label: [t["hate"] for t in batch],
            self.offensive_label: [t["offensive"] for t in batch],
            self.sequence_length: [t["length"] for t in batch],
            self.SGT_label: [t["SGT"] for t in batch],
            self.keep_prob: 1 if test else self.keep_ratio ,
            self.embedding_placeholder: self.embeddings
        }
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
                      "SGT_off": list()}

            hate_accuracy = {"train": list(),
                             "test": list()}
            offensive_accuracy = {"train": list(),
                                  "test": list()}
            SGT_accuracy = {"train": list(),
                            "test": list(),}

            while True:
                off_loss, hate_loss, sgt_loss, sgt_off_loss = 0, 0, 0, 0
                off_acc, hate_acc, sgt_acc = 0, 0 ,0
                off_acc_test, hate_acc_test, sgt_acc_test = 0, 0 ,0

                #_ = self.sess.run(self.embedding_init,
                #                  feed_dict = {self.embedding_placeholder: self.embeddings})
                train_idx, test_idx = train_test_split(np.arange(len(batches)), test_size=0.2, shuffle=True)
                train_batches = [batches[i] for i in train_idx]
                test_batches = [batches[i] for i in test_idx]
                for batch in train_batches:
                    _, _, _, _, sgt_l, off_l, hate_l, sgt_off_l = self.sess.run(
                        [self.task["SGT"]["step"], self.task["hate"]["step"],
                         self.task["offensive"]["step"], self.task["SGT_off"]["step"],
                         self.task["SGT"]["loss"], self.task["offensive"]["loss"],
                         self.task["hate"]["loss"], self.task["SGT_off"]["loss"]],
                        feed_dict=self.feed_dict(batch))

                    sgt_loss += sgt_l
                    hate_loss += hate_l
                    off_loss += off_l
                    sgt_off_loss += sgt_off_l

                    sgt_a, off_a, hate_a = self.sess.run(
                        [self.task["SGT"]["accuracy"], self.task["offensive"]["accuracy"],
                         self.task["hate"]["accuracy"]], feed_dict=self.feed_dict(batch))

                    sgt_acc += sgt_a
                    hate_acc += hate_a
                    off_acc += off_a

                for batch in test_batches:
                    sgt_a_test, off_a_test, hate_a_test = self.sess.run(
                        [self.task["SGT"]["accuracy"], self.task["offensive"]["accuracy"],
                         self.task["hate"]["accuracy"]], feed_dict=self.feed_dict(batch, True))

                    sgt_acc_test += sgt_a_test
                    hate_acc_test += hate_a_test
                    off_acc_test += off_a_test

                print("Iterations: %d\n Hate loss: %.4f"
                      "\n Offensive loss: %.4f\n SGT loss: %.4f\n "
                      "SGT off loss: %.4f" %
                      (epoch, off_loss / len(batches), hate_loss / len(batches),
                       sgt_loss / len(batches), sgt_off_loss / len(batches)))

                losses["SGT"].append(sgt_loss / len(train_batches))
                losses["hate"].append(hate_loss / len(train_batches))
                losses["offensive"].append(off_loss / len(train_batches))
                losses["SGT_off"].append(sgt_off_loss / len(train_batches))

                SGT_accuracy["train"].append(sgt_acc / len(train_batches))
                hate_accuracy["train"].append(hate_acc / len(train_batches))
                offensive_accuracy["train"].append(off_acc / len(train_batches))

                SGT_accuracy["test"].append(sgt_acc_test / len(test_batches))
                hate_accuracy["test"].append(hate_acc_test / len(test_batches))
                offensive_accuracy["test"].append(off_acc_test / len(test_batches))

                epoch += 1

                if epoch == self.epochs:
                    saver.save(self.sess, "saved_model/model")
                    pd.DataFrame.from_dict(losses).to_csv("plots/losses.csv")
                    pd.DataFrame.from_dict(SGT_accuracy).to_csv("plots/SGT.csv")
                    pd.DataFrame.from_dict(hate_accuracy).to_csv("plots/hate.csv")
                    pd.DataFrame.from_dict(offensive_accuracy).to_csv("plots/offensive.csv")
                    plot()
                    break


