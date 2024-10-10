import matplotlib.pyplot as plt
import numpy as np
import util


def main(tau, train_path, eval_path):
    """Problem: Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    lwr_model = LocallyWeightedLinearRegression(tau)
    lwr_model.fit(x_train, y_train)

    # Get MSE value on the validation set
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    y_pred = lwr_model.predict(x_valid)
    mse = np.mean((y_valid - y_pred) ** 2)
    print(f"Mean Squared Error on validation set: {mse}")

    # Plot validation predictions on top of training set
    plt.figure()
    plt.scatter(x_train[:, 1], y_train, marker='x', c='b', label='Training data')
    plt.scatter(x_valid[:, 1], y_pred, marker='o', c='r', label='Validation predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('LWR: Training Data and Validation Predictions')
    plt.savefig('lwr_predictions.png')
    plt.close()
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression():
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        y_pred = np.zeros(m)

        for i in range(m):
            weights = np.exp(-np.sum((self.x - x[i]) ** 2, axis=1) / (2 * self.tau ** 2))
            W = np.diag(weights)
            theta = np.linalg.inv(self.x.T @ W @ self.x) @ self.x.T @ W @ self.y
            y_pred[i] = x[i] @ theta

        return y_pred
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau=5e-1,
         train_path='./train.csv',
         eval_path='./valid.csv')
