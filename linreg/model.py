from sklearn.linear_model import SGDRegressor
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from progress.bar import Bar
from sklearn.model_selection import GridSearchCV

"""
Code for SGD Regression from sklearn
"""


class Model:
    def __init__(self, model, datapath):
        #  model - 'score' or 'time'
        self.model = model
        self.datapath = datapath

        # self.X_stop_binary_train = self.datapath + \
        #     f"X_stop_binary_train_{self.model}.npy"
        # self.X_Nostop_binary_train = self.datapath + \
        #     f"X_Nostop_binary_train_{self.model}.npy"
        # self.X_stop_Nobinary_train = self.datapath + \
        #     f"X_stop_Nobinary_train_{self.model}.npy"
        # self.X_Nostop_Nobinary_train = self.datapath + \
        #     f"X_Nostop_Nobinary_train_{self.model}.npy"

        # self.X_stop_binary_test = self.datapath + \
        #     f"X_stop_binary_test_{self.model}.npy"
        # self.X_Nostop_binary_test = self.datapath + \
        #     f"X_Nostop_binary_test_{self.model}.npy"
        # self.X_stop_Nobinary_test = self.datapath + \
        #     f"X_stop_Nobinary_test_{self.model}.npy"
        # self.X_Nostop_Nobinary_test = self.datapath + \
        #     f"X_Nostop_Nobinary_test_{self.model}.npy"

        self.index = -1
        self.param = {}

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
            reg = SGDRegressor(alpha=self.param['alpha'], loss=self.param['loss'],
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
        vectorizer_stop_binary = CountVectorizer(
            lowercase=True, stop_words='english',
            max_df=1.0, min_df=1, max_features=self.num_features,
            binary=True, dtype=np.int8
        )

        vectorizer_Nostop_binary = CountVectorizer(
            lowercase=True, stop_words=None,
            max_df=1.0, min_df=1, max_features=self.num_features,
            binary=True, dtype=np.int8
        )

        vectorizer_stop_Nobinary = CountVectorizer(
            lowercase=True, stop_words='english',
            max_df=1.0, min_df=1, max_features=self.num_features,
            binary=False, dtype=np.int8
        )

        vectorizer_Nostop_Nobinary = CountVectorizer(
            lowercase=True, stop_words=None,
            max_df=1.0, min_df=1, max_features=self.num_features,
            binary=False, dtype=np.int8
        )

        X_stop_binary = vectorizer_stop_binary.fit_transform(lines).toarray()
        X_Nostop_binary = vectorizer_Nostop_binary.fit_transform(
            lines).toarray()
        X_stop_Nobinary = vectorizer_stop_Nobinary.fit_transform(
            lines).toarray()
        X_Nostop_Nobinary = vectorizer_Nostop_Nobinary.fit_transform(
            lines).toarray()
        print("[step] vectorizing text... DONE")

        print("[step] saving vectors...")
        total_train = self.train_count * self.train_batches
        f = open(self.X_train[0], 'wb')
        np.save(f, X_stop_binary[0:total_train, :])
        f = open(self.X_train[1], 'wb')
        np.save(f, X_Nostop_binary[0:total_train, :])
        f = open(self.X_train[2], 'wb')
        np.save(f, X_stop_Nobinary[0:total_train, :])
        f = open(self.X_train[3], 'wb')
        np.save(f, X_Nostop_Nobinary[0:total_train, :])

        f = open(self.X_test[0], 'wb')
        np.save(f, X_stop_binary[total_train:, :])
        f = open(self.X_test[1], 'wb')
        np.save(f, X_Nostop_binary[total_train:, :])
        f = open(self.X_test[2], 'wb')
        np.save(f, X_stop_Nobinary[total_train:, :])
        f = open(self.X_test[3], 'wb')
        np.save(f, X_Nostop_Nobinary[total_train:, :])

        f = open(self.Y_train, 'wb')
        np.save(f, np.log(1 + np.array(values[0:total_train]))
                if self.log_y else np.array(values[0:total_train]))
        f = open(self.Y_test, 'wb')
        np.save(f, np.array(values[total_train:]))

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
        i = -1
        best_Score = 0
        best_params = {}
        print("Tuning parameters...")
        for index, f in enumerate(self.X_train):
            print(f)

            param_grid = {
                'alpha': [0.001, 0.01, 0.05,
                          0.10, 0.20, 0.5, 1, 10],
                'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'learning_rate': ['constant', 'optimal', 'invscaling'],
            }

            grid = GridSearchCV(estimator=SGDRegressor(), param_grid=param_grid,
                                scoring='r2', verbose=1, n_jobs=-1)

            X, y = np.load(f, mmap_mode='r'), np.load(
                self.Y_train, mmap_mode='r')

            grid_result = grid.fit(X, y)
            print('Best Score for file: ', grid_result.best_score_)
            if(grid_result.best_score_ > best_Score):
                best_Score = grid_result.best_score_
                best_params = grid_result.best_params_
                i = index

        print('Best Score Overall: ', best_Score)
        print('Best Params Overall: ', best_params)
        self.index = i
        self.param = best_params
        print("Tuning parameters... DONE")
