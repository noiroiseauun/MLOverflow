from sklearn.linear_model import SGDRegressor
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from progress.bar import Bar
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

"""
Code for SGD Regression from sklearn
"""


class Model:
    def __init__(self, model, datapath):
        #  model - 'score' or 'time'
        self.model = model
        self.datapath = datapath

        self.index = -1
        self.param = {}

        # Set up for the test, now that we know the result, we do not need to run again
        self.X_train = [self.datapath +
                        f"X_stop_binary_train_{self.model}.npy",
                        self.datapath +
                        f"X_Nostop_binary_train_{self.model}.npy",
                        self.datapath +
                        f"X_stop_Nobinary_train_{self.model}.npy",
                        self.datapath +
                        f"X_Nostop_Nobinary_train_{self.model}.npy"]
        self.X_test = [self.datapath +
                       f"X_stop_binary_test_{self.model}.npy",
                       self.datapath +
                       f"X_Nostop_binary_test_{self.model}.npy",
                       self.datapath +
                       f"X_stop_Nobinary_test_{self.model}.npy",
                       self.datapath +
                       f"X_Nostop_Nobinary_test_{self.model}.npy"]

        # self.X_train = self.datapath + f"X_train_{self.model}.npy"
        # self.X_test = self.datapath + f"X_test_{self.model}.npy"
        self.Y_train = self.datapath + f"Y_train_{self.model}.npy"
        self.Y_test = self.datapath + f"Y_test_{self.model}.npy"

        self.num_samples = 200000
        self.num_features = 10000

        # batches - number of batches
        self.train_batches = 20
        self.train_count = 7000

        self.test_batches = 5
        self.test_count = 12000

        # alpha_reg - coefficient of the regularization term
        self.alpha_reg = 0.001

        # iter - how many training iterations to do for each batch
        self.gd_iter = 1

        self.log_y = False

    def train(self):
        """
        Train the SGDRegression model using batches of X_train

        :return trained regression model
        """
        X, y = np.load(self.X_train[self.index], mmap_mode='r'), np.load(
            self.Y_train, mmap_mode='r')
        print(X.shape)

        with Bar("Training...", max=self.train_batches) as bar:
            reg = SGDRegressor(alpha=self.param['alpha'],
                               penalty=self.param['penalty'], learning_rate=self.param['learning_rate'])
            for i in range(self.train_batches):
                self.process_train_batch(X, y, i, reg)
                bar.next()
            
        return reg

    def test(self, reg):
        """
        Test the regression model

        reg - trained regression model
        y_pred - score predictions for X_test
        """
        X_test, y_pred = np.load(
            self.X_test[self.index], mmap_mode='r'), np.array([])

        print(X_test.shape)
        with Bar("Testing...", max=self.test_batches) as bar:
            for i in range(self.test_batches):
                start = self.test_count * i
                end = start + self.test_count
                y_batch = reg.predict(X_test[start:end])
                y_pred = np.append(y_pred, y_batch)
                bar.next()

        return y_pred

    def process_train_batch(self, X, y, idx, reg):
        """
        Train the model for a single batch

        i - batch index
        reg - regression model
        batch_size - number of examples per batch
        iter - number of training iterations
        """
        start = self.train_count * idx
        end = start + self.train_count
        for i in range(self.gd_iter):
            reg.partial_fit(X[start:end], y[start:end])

    def vectorize_text(self, lines, values):
        """
        Create a vector from the corpus in the array lines
        Splits X, y into training and test data
        Writes the data to a file to clear up memory

        lines - array with Title and Body text with ' and \n removed
        values - array with the Score or time
        """
        print("[step] vectorizing text...")

        # Set up for the test, now that we know the result, we do not need to run again
        vectorizers = [CountVectorizer(
            lowercase=True, stop_words='english',
            max_df=1.0, min_df=1, max_features=self.num_features,
            binary=True, dtype=np.int8
        ), CountVectorizer(
            lowercase=True, stop_words=None,
            max_df=1.0, min_df=1, max_features=self.num_features,
            binary=True, dtype=np.int8
        ), CountVectorizer(
            lowercase=True, stop_words='english',
            max_df=1.0, min_df=1, max_features=self.num_features,
            binary=False, dtype=np.int8
        ), CountVectorizer(
            lowercase=True, stop_words=None,
            max_df=1.0, min_df=1, max_features=self.num_features,
            binary=False, dtype=np.int8
        )]

        total_train = self.train_count * self.train_batches
        for index, vectorizer in enumerate(vectorizers):
            X = vectorizer.fit_transform(
                lines).toarray()
            print("[step] vectorizing text " + str(index) + "... DONE")
            print("[step] saving vectors " + str(index) + "...")
            f = open(self.X_train[index], 'wb')
            np.save(f, X[0:total_train, :])
            f = open(self.X_test[index], 'wb')
            np.save(f, X[total_train:, :])
            X = None

        #total_train = self.train_count * self.train_batches
        # vectorizer = CountVectorizer(
        #     lowercase=True, stop_words='english',
        #     max_df=1.0, min_df=1, max_features=self.num_features,
        #     binary=True, dtype=np.int8
        # )
        # X = vectorizer.fit_transform(lines).toarray()
        # print("[step] vectorizing text... DONE")
        # print("[step] saving vectors...")

        # f = open(self.X_train, 'wb')
        # np.save(f, X[0:total_train, :])
        # f = open(self.X_test, 'wb')
        # np.save(f, X[total_train:, :])
        # X = None

        f = open(self.Y_train, 'wb')
        np.save(f, np.log(1 + np.array(values[0:total_train]))
                if self.log_y else np.array(values[0:total_train]))
        f = open(self.Y_test, 'wb')
        np.save(f, np.array(values[total_train:]))

        f = None
        print("[step] saving vectors... DONE")

    def print_stats(self, y_pred, y_test):
        """
        RMSE maybe? also find max and min values
        """
        mse = np.mean(np.square(y_pred - y_test))
        y_mean = np.mean(y_test)
        var = np.mean(np.square(y_mean - y_test))
        print('RMSE: {}'.format(np.sqrt(mse)))
        print('Std Dev: {}'.format(np.sqrt(var)))
        print('MSE: {}'.format(mse))
        print('Variance: {}'.format(var))
        print('R^2: {}'.format(1 - (mse / var)))
        print('Mean Prediction: {}'.format(np.mean(y_pred)))
        print('Test mean: {}'.format(y_mean))
        print('Max Value: {}'.format(y_test.max()))
        print('Max Prediction: {}'.format(y_pred.max()))
        print('Min Value: {}'.format(y_test.min()))
        print('Min Prediction: {}'.format(y_pred.min()))

    def tune_parameters(self):
        # From our testing, having both stop words and binary counting resulted in a best score(R2)
        # i = 0
        # best_Score = -1000

        # print(
        #     "Find the best vectorization before tuning parameters")

        # for index, f in enumerate(self.X_train):
        #     print("File: ", f)

        #     param_grid = {"alpha": [0.20]}

        #     grid = GridSearchCV(estimator=SGDRegressor(max_iter=100, tol=1e-2), param_grid=param_grid,
        #                         scoring='r2', n_jobs=2, verbose=3)

        #     X, y = np.load(f, mmap_mode='r'), np.load(
        #         self.Y_train, mmap_mode='r')
        #     grid_result = grid.fit(X, y)

        #     print('Best Score: ', grid_result.best_score_)
        #     if(grid_result.best_score_ > best_Score):
        #         best_Score = grid_result.best_score_
        #         i = index

        #     grid = None
        #     X = None
        #     y = None

        i = 1
        print("Best file: ", self.X_train[i])
        print("Tuning parameters(alpha: 0.05, 0.10, 0.20, 0.5, 1)...")

        param_grid = [{"alpha": [0.05, 0.10, 0.20, 0.5, 1], "penalty": ["l1"], "learning_rate": ["constant", "invscaling"]},
                      {"alpha": [0.05, 0.10, 0.20, 0.5, 1], "penalty": ["l2"], "learning_rate": ["optimal", "constant", "invscaling"]}]

        grid = GridSearchCV(estimator=SGDRegressor(max_iter=100, tol=1e-2), param_grid=param_grid,
                            scoring="r2", n_jobs=2, verbose=3)

        # X, y = np.load(self.X_train, mmap_mode='r'), np.load(
        #     self.Y_train, mmap_mode='r')

        X, y = np.load(self.X_train[i], mmap_mode='r'), np.load(
            self.Y_train, mmap_mode='r')
        grid_result = grid.fit(X, y)
        self.param = grid_result.best_params_
        print('Best Score for file: ', grid_result.best_score_)
        print('Best Param for file: ', grid_result.best_params_)

        print("Tuning parameters... DONE")
