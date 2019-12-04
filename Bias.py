from ntap.data import Dataset
from ntap.models import RNN
import pandas as pd
import os
from nn import *

class Bias():
    def __init__(self, data_path, model_path=False, test_data_path=None):
        self.model_path = model_path
        data = self.initialize_dataset(data_path)
        model = RNN("hate ~ seq(text)", data=data, learning_rate=0.001, rnn_dropout=0.8)
        vocabs = data.vocab

        if not os.path.exists(model_path + ".meta"):
            #results = model.CV(data, num_epochs=10, num_folds=10, batch_size=512)
            #results.summary()

            """
            test_data = self.initialize_dataset(test_data_path)
            y = model.predict(test_data, indices=test_idx.tolist(),
                             model_path=model_path)
            labels = dict()
            num_classes = dict()
            for key in y:
                var_name = key.replace("prediction-", "")
                test_y, card = data.get_labels(idx=test_idx, var=var_name)
                labels[key] = test_y
                num_classes[key] = card
            stats = self.evaluate(y, labels, num_classes)  # both dict objects
            results.append(stats)
            CV_Results(results)
            """
            model.train(data, num_epochs=10, batch_size=512, model_path=model_path)

        fake_data_path = self.generate_fake(data_path, vocabs)
        fake_data = self.initialize_dataset(fake_data_path)

        predictions = model.predict(fake_data, orig_data=data, model_path=model_path,
                                    column="text")
        fake_data.data["predicted_hate"] = predictions["prediction-hate"]
        fake_data.data.to_csv("biased/predictions.csv", index=False)

    def initialize_dataset(self, data_dir):
        data = Dataset(data_dir)
        data.set_params(vocab_size=10000,
                        mallet_path="/home/aida/Data/mallet/mallet-2.0.8/bin/mallet",
                        glove_path="/home/aida/Data/word_embeddings/GloVe/glove.840B.300d.txt")
        data.clean("text")
        return data

    def generate_fake(self, data_path, vocabs):
        source_df = pd.read_csv(data_path)
        print(source_df.shape[0], "datapoints")

        source_df = tokenize_data(source_df, "text")
        source_df = remove_empty(source_df, "text")
        print(source_df.shape[0], "datapoints after removing empty strings")
        fake = self.expand(source_df, "extended_SGT.txt", vocabs)
        fake_path = os.path.split(data_path)[0] + "/fake_gab.csv"
        fake.to_csv(fake_path)
        return fake_path

    def expand(self, corpus, SGT_path, vocabs):
        SGTs = [tok.replace("\n", "") for tok in open(SGT_path, "r").readlines()]
        SGTs = [tok for tok in SGTs if tok in vocabs]
        fake_df = {"text": list(),
                   "hate": list(),
                   "sample": list(),
                   "SGT": list(),
                   "origin_SGT": list()}

        for i, row in corpus.iterrows():
            for j, tok in enumerate(row["text"]):
                for s in SGTs:
                    if s in tok:
                        text = " ".join(word for word in row["text"])
                        fake_df["text"].extend([text.replace(tok, _sgt) for _sgt in SGTs])
                        fake_df["hate"].extend([row["hate"] for t in SGTs])
                        fake_df["sample"].extend([i for t in SGTs])
                        fake_df["SGT"].extend([t for t in SGTs])
                        fake_df["origin_SGT"].extend([s for t in SGTs])

        return pd.DataFrame.from_dict(fake_df)
