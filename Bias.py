from ntap.data import Dataset
from ntap.models import RNN
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import os
from nn import *
from ntap.helpers import CV_Results

class Bias():
    def __init__(self, data_path, model_path=False, test_data_path=None, oversample=False):
        self.model_path = model_path
        data = self.initialize_dataset(data_path)
        vocabs = data.vocab
        if oversample:
            data_path = self.generate(pd.read_csv(data_path), vocabs)
        model = RNN("hate ~ seq(text)", data=data, learning_rate=0.001, rnn_dropout=0.8)

        if not os.path.exists(model_path + ".meta"):
            results = model.CV(data, num_epochs=10, num_folds=10, batch_size=512)
            results.summary()
            model.train(data, num_epochs=10, batch_size=512, model_path=model_path)
        else:
            test_data = self.initialize_dataset(test_data_path)
            y = model.predict(test_data,
                             model_path=model_path)
            labels = dict()
            num_classes = dict()
            for key in y:
                var_name = key.replace("prediction-", "")
                test_y, card = data.get_labels(idx=range(test_data.data.shape[0]), var=var_name)
                labels[key] = test_y
                num_classes[key] = 2
            print(labels)
            print(y)
            print(f1_score(labels["prediction-hate"], y["prediction-hate"]))
            stats = model.evaluate(y, labels, num_classes)  # both dict objects
            print(stats)
            #CV_Results(results) 

        #fake_data_path = self.generate_fake(data_path, vocabs)
        fake_data_path = "Data/24k/fake_test.csv"
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
        data.encode_docs("text")
        return data

    def generate(self, corpus, vocabs, SGT_path="extended_SGT.txt"):
        SGTs = [tok.replace("\n", "") for tok in open(SGT_path, "r").readlines()]
        SGTs = [tok for tok in SGTs if tok in vocabs]

        for i, row in corpus.iterrows():
            for j, tok in enumerate(row["text"]):
                for s in SGTs:
                    if s in tok:
                        text = " ".join(word for word in row["text"])
                        texts = [text.replace(tok, _sgt) for _sgt in SGTs]
                        new_df = {col: [row[col] for i in range(len(texts))] for col in corpus.columns}
                        new_df["text"] = texts
                        corpus.append(pd.DataFrame.from_dict(new_df))
        corpus.to_csv("Data/extended_gab.csv", index=False)
        return "Data/extended_gab.csv"

