import csv
import string
from model import Model
import numpy as np
from progress.bar import Bar
import util
from datetime import datetime


class TimeModel(Model):
    model = 'time'

    def __init__(self, datapath):
        super().__init__('time', datapath)
        self.alpha_reg = 0.05
        # Note: there are less than 1 million questions with answers (that are after the question was created)
        self.num_samples = 900000
        self.train_count = 10000
        self.train_batches = 80
        self.test_batches = 10
        self.test_count = 10000
        self.log_y = True

    def calc_time(self, time1, time2):
        """
        Calculate the number of seconds between time1 and time2

        time1 - answer date
        time2 - quesiton date

        elapsed - seconds between question and answer
        """
        dt1 = datetime.fromisoformat(time1[:-1])
        dt2 = datetime.fromisoformat(time2[:-1])
        delta = dt1 - dt2
        elapsed = delta.total_seconds()
        return elapsed

    def data(self, start, count):
        answers, lines, times, i = {}, [], [], start

        with open(self.datapath + 'Answers-Final.csv', 'r') as f:
            fptr = csv.reader(f, delimiter=',')
            for line in fptr:
                answers[line[2]] = line[1]

        is_first = True
        with Bar("Loading data...", max=count) as bar:
            with open(self.datapath + 'Questions-Final.csv', 'r') as f:
                fptr = csv.reader(f, delimiter=',')
                for line in fptr:
                    if is_first:
                        is_first = False
                        i += 1
                    elif i == start + count + 1:
                        break
                    elif i > start:
                        if line[0] in answers:
                            delta = self.calc_time(answers[line[0]], line[1])
                            # Only considers answers that have a later date than the question
                            if delta >= 0:
                                tokens = util.clean_tokenize(line[4] + line[5])
                                tokens = [tok.translate(str.maketrans('', '', string.punctuation)) for tok in tokens]
                                lines.append(' '.join(tokens))
                                times.append(delta)
                                i += 1
                                bar.next()

        return lines, times

    def run(self, load_data=True):
        if load_data:
            lines, values = self.data(0, self.num_samples)
            self.vectorize_text(lines, values)

        reg = self.train()
        y_pred = self.test(reg)
        # Using log(y) so convert back to seconds with exp(y_pred)
        y_pred = np.exp(y_pred) - 1
        y_test = np.load(self.Y_test, mmap_mode='r')
        self.print_stats(y_pred, y_test)
