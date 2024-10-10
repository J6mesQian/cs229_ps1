import numpy as np

A = np.array([[1, 0], [0, 2]])
theta_0 = np.array([-1, 0.5])

def J(theta):
    return theta.T.dot(A).dot(theta)

def update_theta(theta, lr):
    """Problem: given the current value of theta and the learning rate lr,
    you should return the new value of theta obtained by running 1 iteration
    of the gradient descend algorithm.

    Args:
        theta: the current theta
        lr: the learning rate

    Returns:
        the new value of theta after 1 iteration of gradient descend
    """
    # *** START CODE HERE ***
    gradient = 2 * A.dot(theta)
    new_theta = theta - lr * gradient
    return new_theta
    # *** END CODE HERE ***

def gradient_descend(J, theta_0, lr, update_theta, epsilon=1e-50):
    """Write the gradient descend algorithm using the parameters.
    You can stop the algorithm when either:
        1. the absolute difference of J(theta^[t]) and J(theta^[t-1]) is less than epsilon or
        2. the loss function J(theta^[t]) is bigger than 1e20

    Args:
        J: the objective function
        theta_0: the initial theta
        lr: the learning rate
        update_theta: the theta update function, which you implemented above
        epsilon: we stop when the absolute loss function differences is below this value
    """
    theta = theta_0
    # *** START CODE HERE ***
    prev_loss = J(theta)
    while True:
        theta = update_theta(theta, lr)
        current_loss = J(theta)
        
        if abs(current_loss - prev_loss) < epsilon:
            break
        if current_loss > 1e20:
            break
        
        prev_loss = current_loss
    # *** END CODE HERE ***
    return theta

if __name__ == "__main__":
    theta = gradient_descend(J, theta_0, 1e-2, update_theta)
    assert np.isclose(theta[0], theta[1]), f"elements of theta {theta} is not close"
    assert all(abs(theta_i) < 1e-24 for theta_i in theta), f"elements of theta {theta} is too far from the optimal value"
    print("All sanity checks passed")

