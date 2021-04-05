import csv
import string
from model import Model
import numpy as np
from progress.bar import Bar
import util
import codecs
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
        # self.alpha_reg = 0.05
        self.num_samples = 100000

        # train/test_count is per batch
        self.train_count = 1000
        self.train_batches = 80
        self.test_batches = 20
        self.test_count = 1000

    def data(self, start, count):
        i, lines, scores = 0, [], []
        f = codecs.open(self.datapath + 'Questions-Final.csv', 'r', 'utf-8')
        corpus = csv.reader(f)
        with Bar("Loading data...", max=count) as bar:
            for line in corpus:
                if i == start + count + 1:
                    break
                elif i > start:
                    try:
                        tokens = util.clean_tokenize(line[4] + line[5])
                    except:
                        print("\nerror1")
                    tokens = [tok.translate(str.maketrans(
                        '', '', string.punctuation)) for tok in tokens]
                    lines.append(' '.join(tokens))
                    try:
                        scores.append(int(line[3]))
                    except:
                        scores.append(0)
                bar.next()
                i += 1

        return lines, scores

    def run(self, load_data=True):
        # if load_data:
        #     lines, values = self.data(0, self.num_samples)
        #     self.vectorize_text(lines, values)

        self.tune_parameters()

        reg = self.train()
        y_pred = self.test(reg)
        y_test = np.load(self.Y_test, mmap_mode='r')
        self.print_stats(y_pred, y_test)
