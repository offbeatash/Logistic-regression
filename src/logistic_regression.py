import numpy as np

class LogisticRegressionGD:
    def __init__(self, lr = 0.1, n_iters =5000, lambda_= 0):
        self.lr = lr
        self.n_iters = n_iters
        self.lambda_ = lambda_
        self.w = None
        self.b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, x, y):
        self.losses = []
        y = y.reshape(-1)

        n_samples, n_features = x.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            linear = np.dot(x,self.w) +self.b
            y_pred = self.sigmoid(linear)

            error = y_pred -y

            #log files
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_pred, epsilon, 1-epsilon)
            loss = -(1/n_samples) * np.sum( y* np.log(y_pred_clipped)+ (1-y) * np.log(1-y_pred_clipped))
            self.losses.append(loss)

            dw = (1/n_samples) * np.dot(x.T, error) + (self.lambda_ / n_samples) * self.w
            db = (1/n_samples) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_prob(self, x):
        linear = np.dot(x, self.w)  + self.b
        return self.sigmoid(linear)
    
    def predict(self, x):
        probs = self.predict_prob(x)
        return np.array([1 if i >= 0.5 else 0 for i in probs])