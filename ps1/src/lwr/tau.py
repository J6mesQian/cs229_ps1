import matplotlib.pyplot as plt
import numpy as np
import util

from lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem: Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Load validation and test sets
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    best_tau = None
    best_mse = float('inf')
    best_model = None

    # Search tau_values for the best tau (lowest MSE on the validation set)
    for tau in tau_values:
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)
        y_pred_valid = model.predict(x_valid)
        mse = np.mean((y_valid - y_pred_valid) ** 2)
        
        print(f"Tau: {tau}, Validation MSE: {mse}")
        
        # Plot predictions for each tau
        plt.figure(figsize=(10, 6))
        plt.scatter(x_train[:, 1], y_train, marker='x', c='b', label='Training data')
        plt.scatter(x_valid[:, 1], y_pred_valid, marker='o', c='r', label='Validation predictions')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title(f'LWR: Training Data and Validation Predictions (tau = {tau})')
        plt.savefig(f'lwr_predictions_tau_{tau}.png')
        plt.close()

        if mse < best_mse:
            best_mse = mse
            best_tau = tau
            best_model = model

    print(f"\nBest tau: {best_tau}")
    print(f"Best validation MSE: {best_mse}")

    # Run on the test set to get the MSE value
    y_pred_test = best_model.predict(x_test)
    test_mse = np.mean((y_test - y_pred_test) ** 2)
    print(f"Test MSE with best tau: {test_mse}")

    # Save predictions to pred_path
    np.savetxt(pred_path, y_pred_test)

    # Plot final predictions on test set
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train[:, 1], y_train, marker='x', c='b', label='Training data')
    plt.scatter(x_test[:, 1], y_pred_test, marker='o', c='r', label='Test predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'LWR: Training Data and Test Predictions (best tau = {best_tau})')
    plt.savefig('lwr_predictions_best_tau.png')
    plt.close()
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='./train.csv',
         valid_path='./valid.csv',
         test_path='./test.csv',
         pred_path='./pred.txt')
