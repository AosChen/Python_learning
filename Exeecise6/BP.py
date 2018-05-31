import numpy as np


class BP(object):

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = dict()
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None):
        w1, b1 = self.params['W1'], self.params['b1']
        w2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape
        h_output = np.maximum(0, X.dot(w1) + b1)
        scores = h_output.dot(w2) + b2
        if y is None:
            return scores
        shift_scores = scores - np.max(scores, axis=1).reshape(-1, 1)
        softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1).reshape(-1, 1)
        loss = -np.sum(np.log(softmax_output[range(N), list(y)]))
        loss /= N

        grads = {}
        dscores = softmax_output.copy()
        dscores[range(N), list(y)] -= 1
        dscores /= N
        grads['W2'] = h_output.T.dot(dscores)
        grads['b2'] = np.sum(dscores, axis=0)

        dw2 = dscores.dot(w2.T)
        dw1 = (h_output > 0) * dw2
        grads['W1'] = X.T.dot(dw1)
        grads['b1'] = np.sum(dw1, axis=0)

        return loss, grads

    def train(self, X, y, X_val, y_val, learning_rate=0.01, learning_rate_decay=0.95,
              num_iters=100, batch_size=200, verbose=False):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            idx = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[idx]
            y_batch = y[idx]
            loss, grads = self.loss(X_batch, y=y_batch)
            loss_history.append(loss)

            self.params['W2'] += - learning_rate * grads['W2']
            self.params['b2'] += - learning_rate * grads['b2']
            self.params['W1'] += - learning_rate * grads['W1']
            self.params['b1'] += - learning_rate * grads['b1']

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch)[0] == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history[-1],
            'train_acc_history': train_acc_history[-1],
            'val_acc_history': val_acc_history[-1],
        }

    def predict(self, X):
        h = np.maximum(0, X.dot(self.params['W1']) + self.params['b1'])
        scores = h.dot(self.params['W2']) + self.params['b2']
        y_pred = np.argmax(scores, axis=1)
        return y_pred

    def predict_prob(self, X):
        h = np.maximum(0, X.dot(self.params['W1']) + self.params['b1'])
        scores = h.dot(self.params['W2']) + self.params['b2']
        softmax_output = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(-1, 1)
        return softmax_output
