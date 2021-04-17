import csv
import string
from model import Model
import numpy as np
from progress.bar import Bar
import util
import codecs
from lime.lime_tabular import LimeTabularExplainer
"""
For 10 GB num_samples * feat_cnt should be less than 10^10
batch_size*num_batches = num_training or num_test
MSE ~ 200
SGD for predicting score
"""


class ScoreModel(Model):
    model = 'score'

    def __init__(self, datapath=""):
        super().__init__('score', datapath)
        self.num_samples = 1000000

        # train/test_count is per batch
        self.train_count = 10000
        self.train_batches = 80
        self.test_batches = 20
        self.test_count = 10000

    def data(self, start, count):
        i, lines, scores = 0, [], []
        f = codecs.open(self.datapath + 'Questions-Final.csv', 'r', 'utf-8')
        corpus = csv.reader(f)
        is_first = True
        with Bar("Loading data...", max=count) as bar:
            for line in corpus:
                if is_first:
                    is_first = False
                    i += 1
                if i == start + count + 1:
                    break
                elif i > start:
                    try:
                        tokens = util.clean_tokenize(line[4] + line[5])
                        tokens = [tok.translate(str.maketrans(
                            '', '', string.punctuation)) for tok in tokens]
                        scores.append(int(line[3]))
                        lines.append(' '.join(tokens))
                        bar.next()
                        i += 1
                    except:
                        print("\nerror: skipping")

        return lines, scores

    def run(self, load_data=True, tune_parameter=True):
        if load_data:
            lines, values = self.data(0, self.num_samples)
            self.vectorize_text(lines, values)

        # If tune_parameter is false, we run with our experimented parameters
        if tune_parameter:
            self.tune_parameters()
        else:
            self.index = 0
            self.param = {"alpha": 0.1,
                          "learning_rate": "invscaling", "penalty": "l2"}

        reg = self.train()
        print(reg.densify())
        y_pred = self.test(reg)
        y_test = np.load(self.Y_test, mmap_mode='r')
        print(y_pred.shape)
        self.print_stats(y_pred, y_test)

        # Show a Lime plot of the regression. The labelings will no be correct since we are using a regression model.
        X_train = np.load(self.X_train[self.index], mmap_mode='r')
        X_test = np.load(self.X_test[self.index], mmap_mode='r')
        explainer = LimeTabularExplainer(X_train, mode="regression")
        exp = explainer.explain_instance(X_test[self.text_index], reg.predict)
        exp.as_pyplot_figure()
