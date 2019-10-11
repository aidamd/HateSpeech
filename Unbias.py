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
        self.offensive_label = tf.placeholder(tf.float32, [None], name="offensive_labels")
        self.hate_label = tf.placeholder(tf.float32, [None], name="hate_labels")
        self.SGT_label = tf.placeholder(tf.float32, [None], name="SGT_labels")

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

        self.offensive_logits = tf.layers.dense(self.offensive, 2)
        self.offensive_xentropy = tf.losses.sparse_softmax_cross_entropy(
            labels=tf.cast(self.offensive_label, tf.int32),
            logits=self.offensive_logits)

        self.offensive_loss = tf.reduce_mean(self.offensive_xentropy)
        self.offensive_step = tf.train.AdamOptimizer(learning_rate=self.offensive_learning_rate)\
            .minimize(self.offensive_loss)

        self.hate_logits = tf.layers.dense(self.representation, 2)
        self.hate_xentropy = tf.losses.sparse_softmax_cross_entropy(
            labels=tf.cast(self.hate_label, tf.int32),
            logits=self.hate_logits)

        self.hate_loss = tf.reduce_mean(self.hate_xentropy)
        self.hate_step = tf.train.AdamOptimizer(learning_rate=self.hate_learning_rate) \
            .minimize(self.hate_loss)

        self.SGT_logits = tf.layers.dense(self.SGT, self.num_SGT + 1)
        self.SGT_xentropy = tf.losses.sparse_softmax_cross_entropy(
            labels=tf.cast(self.SGT_label, tf.int32),
            logits=self.SGT_logits)

        self.SGT_loss = tf.reduce_mean(self.SGT_xentropy)
        self.SGT_step = tf.train.AdamOptimizer(learning_rate=self.SGT_learning_rate) \
            .minimize(self.SGT_loss)

        self.SGT_off_logits = tf.layers.dense(self.offensive, self.num_SGT + 1)
        self.SGT_off_xentropy = tf.losses.sparse_softmax_cross_entropy(
            labels=tf.cast(self.SGT_label, tf.int32),
            logits=self.SGT_off_logits)

        self.SGT_off_loss = tf.reduce_mean(self.SGT_off_xentropy)
        self.SGT_off_step = tf.train.AdamOptimizer(learning_rate=self.SGT_off_learning_rate) \
            .minimize(-self.SGT_off_loss)


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
            while True:
                off_loss, hate_loss, sgt_loss, sgt_off_loss = 0, 0, 0, 0
                #_ = self.sess.run(self.embedding_init,
                #                  feed_dict = {self.embedding_placeholder: self.embeddings})
                train_idx, test_idx = train_test_split(np.arange(len(batches),test_size=0.2), shuffle=True)
                train_batches = [batches[i] for i in train_idx]
                test_batches = [batches[i] for i in test_idx]
                for batch in train_batches:
                    feed_dict = {
                        self.encoder_input: [t["enc_input"] for t in batch],
                        self.hate_label: [t["hate"] for t in batch],
                        self.offensive_label: [t["offensive"] for t in batch],
                        self.sequence_length: [t["length"] for t in batch],
                        self.SGT_label: [t["SGT"] for t in batch],
                        self.keep_prob: self.keep_ratio,
                        self.embedding_placeholder: self.embeddings
                    }


                    _, _, _, _, sgt_l, off_l, hate_l, sgt_off_l = self.sess.run(
                        [self.SGT_step, self.hate_step, self.offensive_step, self.SGT_off_step,
                        self.SGT_loss, self.offensive_loss, self.hate_loss, self.SGT_off_loss],
                        feed_dict=feed_dict)
                    sgt_loss += sgt_l
                    hate_loss += hate_l
                    off_loss += off_l
                    sgt_off_loss += sgt_off_l


                print("Iterations: %d\n Hate loss: %.4f"
                      "\n Offensive loss: %.4f\n SGT loss: %.4f\n "
                      "SGT off loss: %.4f" %
                      (epoch, off_loss / len(batches), hate_loss / len(batches),
                       sgt_loss / len(batches), sgt_off_loss / len(batches)))

                losses["SGT"].append(sgt_loss / len(batches))
                losses["hate"].append(hate_loss / len(batches))
                losses["offensive"].append(off_loss / len(batches))
                losses["SGT_off"].append(sgt_off_loss / len(batches))

                epoch += 1

                if epoch == self.epochs:
                    saver.save(self.sess, "saved_model/model")
                    pd.DataFrame.from_dict(losses).to_csv("losses.csv")
                    plot()
                    break


