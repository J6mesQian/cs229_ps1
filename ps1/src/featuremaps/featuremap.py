import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.linalg.solve(X.T @ X, X.T @ y)
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        n_examples = X.shape[0]
        X_poly = np.zeros((n_examples, k + 1))
        for i in range(k + 1):
            X_poly[:, i] = X[:, 1] ** i
        return X_poly
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        n_examples = X.shape[0]
        X_sin = np.zeros((n_examples, k + 2))
        X_sin[:, 0] = 1  # Intercept term
        for i in range(1, k + 1):
            X_sin[:, i] = X[:, 1] ** i
        X_sin[:, -1] = np.sin(X[:, 1])
        return X_sin
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return X @ self.theta
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        model = LinearModel()
        if sine:
            X_train = model.create_sin(k, train_x)
            X_plot = model.create_sin(k, plot_x)
        else:
            X_train = model.create_poly(k, train_x)
            X_plot = model.create_poly(k, plot_x)
        
        model.fit(X_train, train_y)
        plot_y = model.predict(X_plot)
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Regression with Different Degrees' + (' (with Sine)' if sine else ''))
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all experiments
    '''
    # *** START CODE HERE ***
    
    # (b) Degree-3 polynomial regression
    run_exp(train_path, sine=False, ks=[3], filename='degree_3_polynomial.png')
    
    # (c) Degree-k polynomial regression
    run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='degree_k_polynomial.png')
    
    # (d) Degree-k polynomial regression with sine feature
    run_exp(train_path, sine=True, ks=[0, 1, 2, 3, 5, 10, 20], filename='degree_k_polynomial_with_sine.png')
    
    # (e) Overfitting with expressive models and small data
    run_exp(small_path, sine=False, ks=[1, 2, 5, 10, 20], filename='small_dataset_overfitting.png')
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')