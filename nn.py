import tensorflow as tf
import numpy as np
from nltk import tokenize as nltk_token
import operator
import re


def generator(Z, hidden_size, vocab_len, label):
    with tf.variable_scope("GAN/Generator", reuse=tf.AUTO_REUSE):
        h1 = tf.layers.dense(Z, hidden_size, activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h1, vocab_len, activation=tf.tanh)
    return out


def discriminator(X_real, X_fake, filter_sizes, num_filters, keep_ratio, scope, label):
    cnn_real = cnn(X_real, filter_sizes, num_filters, keep_ratio, scope)
    cnn_fake = cnn(X_fake, filter_sizes, num_filters, keep_ratio, scope, reuse=True)

    class_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(cnn_real) if label == 1 else tf.zeros_like(cnn_real)
        , logits=cnn_real))

    disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(cnn_real), logits=cnn_real)) + \
             tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                 labels=tf.zeros_like(cnn_fake), logits=cnn_fake))
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(cnn_real), logits=cnn_fake))
    return disc_loss, gen_loss, class_loss

def cnn(input, filter_sizes, num_filters, keep_ratio, scope, padding="VALID", reuse=False):
    dim = input.get_shape().as_list()[-1]
    input = tf.expand_dims(input, -1)

    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        pooled_outputs = list()
        for size in filter_sizes:
            with tf.variable_scope('conv-maxpool-%s' % size):
                b = tf.get_variable('b', shape=[num_filters])
                W = tf.get_variable('W', shape=[size, dim, 1, num_filters])

                conv = tf.nn.conv2d(input, W,
                    strides=[1, 1, 1, 1], padding=padding)
                relu = tf.nn.relu(tf.nn.bias_add(conv, b))

                pooled = tf.reduce_max(relu, axis=1, keep_dims=True)
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        drop = tf.nn.dropout(h_pool_flat, keep_ratio)

        with tf.variable_scope('output'):
            W = tf.get_variable('W', [num_filters * len(filter_sizes), 1])
            b = tf.get_variable('b', [1])
            logits = tf.reshape(tf.matmul(drop, W) + b, [-1])

    return logits

def get_batches(data, batch_size, pad_idx, hate=None, offensive=None, SGT=None):
    batches = []
    for idx in range(len(data) // batch_size + 1):
        if idx * batch_size !=  len(data):
            data_batch = data[idx * batch_size: min((idx + 1) * batch_size, len(data))]
            hate_batch = hate[idx * batch_size: min((idx + 1) * batch_size, len(hate))] if hate else None
            offensive_batch = offensive[idx * batch_size: min((idx + 1) * batch_size, len(offensive))] \
                if hate else None

            data_info = batch_to_info(data_batch, hate_batch, offensive_batch, SGT, pad_idx)
            batches.append(data_info)
    return batches

def get_balanced_batches(data, batch_size, pad_idx, hate=None, offensive=None, SGT=None):
    batches = list()
    for sub_idx in balanced_batch_indices(len(data), batch_size, hate):
        data_batch = [data[i] for i in sub_idx]
        hate_batch = [hate[i] for i in sub_idx] if hate else None
        offensive_batch = [offensive[i] for i in sub_idx] if hate else None
        SGT_batch = [SGT[i] for i in sub_idx] if hate else None

        data_info = batch_to_info(data_batch, hate_batch, offensive_batch, SGT, pad_idx)
        batches.append(data_info)
    return batches

def balanced_batch_indices(size, batch_size, hate):
    # produce iterable of (start, end) batch indices
    for i in range(0, size, batch_size):
        true_idx = np.random.choice(np.where(np.array(hate) == 1)[0], int(batch_size * .1))
        false_idx = np.random.choice(np.where(np.array(hate) == 0)[0], int(batch_size * .9))
        sub_idx = np.concatenate((true_idx, false_idx), axis=0)
        np.random.shuffle(sub_idx)
        yield sub_idx


def batch_to_info(batch, hate, offensive, SGT, pad_idx):
    max_len = max(len(sent) for sent in batch)
    batch_info = list()
    for i, sent in enumerate(batch):
        padding = [pad_idx] * (max_len - len(sent))
        sentence = {
            "enc_input": sent + padding,
            "length": len(sent),
            "hate": hate[i] if hate else None,
            "offensive": offensive[i] if offensive else None,
            "SGT": SGT[i] if SGT else None
        }
        batch_info.append(sentence)
    return batch_info

def tokenize_data(corpus, col):
    #sent_tokenizer = toks[self.params["tokenize"]]
    for idx, row in corpus.iterrows():
        corpus.at[idx, col] = nltk_token.WordPunctTokenizer().tokenize(clean(row[col]))
    return corpus
"""
def extract_SGT(df_tokens, vocab, SGT_path):
    SGT_dict = read_SGT(vocab, SGT_path)
    SGTs = list()
    for sent in df_tokens:
        s = [tok for tok in SGT_dict.keys() if tok in sent]
        s = [len(SGT_dict)] if s == []
        SGTS.append(s)
        #found = False
        #for tok in SGT_dict.keys():
        #    if tok in sent:
        #        SGTs.append(SGT_dict[tok])
        #        found = True
        #        break
        #if not found:
        #    SGTs.append(len(SGT_dict))
        # seq2seq
        # SGTs.append([tok if tok in SGT_dict.values() else 0 for tok in sent])
    #print(SGT_dict.keys())
    return SGTs, len(SGT_dict)
"""
def read_SGT(vocab, SGT_path):
    s = [tok.replace("\n", "") for tok in open(SGT_path, "r").readlines()]
    SGTs = list(set([tok for tok in s if tok in vocab]))
    #print(SGTs)
    #print({tok: i for i, tok in enumerate(SGTs)})
    SGT_dict = {}
    for i in range(len(SGTs)):
        #print(i)
        #print(SGTs[i])
        SGT_dict[SGTs[i]] = i
    #print(SGT_dict)
    #print(SGTs)
    return SGT_dict

def clean(sent):
    http = re.sub("https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=/]{2,256}"
                  "\.[a-z]{2,4}([-a-zA-Z0-9@:%_\+.~#?&//=]*)", "", sent)
    return re.sub(r'[^a-zA-Z ]', r'', http.lower())


def remove_empty(corpus, col):
    drop = list()
    for i, row in corpus.iterrows():
        if row[col] == "" or len(row[col]) < 4 or len(row[col]) > 100:
            drop.append(i)
    return corpus.dropna().drop(drop)

def learn_vocab(corpus, vocab_size):
    print("Learning vocabulary of size %d" % (vocab_size))
    tokens = dict()
    for sent in corpus:
        for token in sent:
            if token in tokens:
                tokens[token] += 1
            else:
                tokens[token] = 1
    words, counts = zip(*sorted(tokens.items(), key=operator.itemgetter(1), reverse=True))
    return list(words[:vocab_size]) + ["<unk>", "<pad>"]

def tokens_to_ids(corpus, vocab, SGT_path, SGT_dict=None):
    print("Converting corpus of size %d to word indices based on learned vocabulary" % len(corpus))
    if vocab is None:
        raise ValueError("learn_vocab before converting tokens")

    mapping = {word: idx for idx, word in enumerate(vocab)}
    unk_idx = vocab.index("<unk>")
    
    if not SGT_dict:
        SGT_dict = read_SGT(vocab, SGT_path)
    SGTs = list()

    for i, row in enumerate(corpus):
        SGT = len(SGT_dict)
        #SGT = list()
        for j, tok in enumerate(row):
            for s in SGT_dict.keys():
                if s in corpus[i][j]:
                    SGT = SGT_dict[s]
                    #SGT.append(SGT_dict[s])
            try:
                corpus[i][j] = mapping[corpus[i][j]]
            except:
                corpus[i][j] = unk_idx
        #if SGT == len(SGT_dict):
        #    SGT.append(len(SGT_dict))
            #print([vocab[r] for r in row])
            #print(len(SGT_dict))
            #exit(1)
        SGTs.append(SGT)
    return corpus, SGTs, len(SGT_dict), SGT_dict

def load_embedding(vocabulary, file_path, embedding_size):
    embeddings = np.random.randn(len(vocabulary), embedding_size)
    found = 0
    with open(file_path, "r") as f:
        for line in f:
            split = line.split()
            idx = len(split) - embedding_size
            vocab = "".join(split[:idx])
            if vocab in vocabulary:
                embeddings[vocabulary.index(vocab)] = np.array(split[idx:], dtype=np.float32)
                found += 1
    print("Found {}/{} of vocab in word embeddings".
          format(found, len(vocabulary)))
    return embeddings
