import numpy as np

# Generate a 1D array of 5 random numbers from a uniform distribution
uniform_random_numbers = np.random.rand(5)
print(uniform_random_numbers)

# Generate a 2D array of shape (3, 4) of random numbers from a uniform distribution
uniform_random_matrix = np.random.rand(3, 4)
print(uniform_random_matrix)


# Generate a 1D array of 5 random numbers from a standard normal distribution
normal_random_numbers = np.random.randn(5)
print(normal_random_numbers)

# Generate a 2D array of shape (3, 4) of random numbers from a standard normal distribution
normal_random_matrix = np.random.randn(3, 4)
print(normal_random_matrix)

#########################################################################
# seed 0 ensures the same random numbers are generated each time
np.random.seed(0)
X = 2 * np.random.rand(10, 1)
print(X)

# output/target
y = 4 + 3 * X + np.random.randn(10, 1)
print(y)

# Add a bias term (column of ones) to the input data the sahpe becomes (10, 2)
X_b = np.c_[np.ones((10, 1)), X]
print(X_b)

# Initialize parameters (weights)
theta = np.random.randn(2, 1)
print(theta)

# Hyperparameters
learning_rate = 0.01
n_epochs = 1

def sgd(X, y, theta, learning_rate, n_epochs):
    """
    Perform Stochastic Gradient Descent (SGD) to optimize the parameters.

    Parameters:
    X (numpy.ndarray): Input features with shape (m, n+1), where m is the number of samples and n is the number of features.
                       The first column should be all ones to account for the bias term.
    y (numpy.ndarray): Target values with shape (m, 1).
    theta (numpy.ndarray): Initial parameters (weights) with shape (n+1, 1).
    learning_rate (float): Learning rate for the gradient descent updates.
    n_epochs (int): Number of epochs (iterations over the entire dataset).

    Returns:
    numpy.ndarray: Optimized parameters (weights) with shape (n+1, 1).
    """
    m = len(X)
    print(m)

    for epoch in range(n_epochs):
        print(f"############################### epoch {epoch}")
        for i in range(m):
            print(f"############# inner iteraction {i}")
            random_index = np.random.randint(m)
            print(f"### random index selected: {random_index}")
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            print(f"### xi: {xi}")
            print(f"### yi: {yi}")

            # Compute the prediction for the current sample using dot product
            prediction = xi.dot(theta)
            print(f"### current theta: {theta}")
            print(f"### predicted value using dot product is : {prediction}")

            # Compute the error for the current sample
            error = prediction - yi
            print(f"### error for this random row/point is: {error}")

            # Compute the gradient for the current sample using the transpose of xi and doc product
            gradients = 2 * xi.T.dot(error)
            print(f"### gradients/slop for this random point/row is : {gradients}")

            # update the theta using the learning rate and gradients
            theta = theta - learning_rate * gradients
            print(f"### updated theta is ", theta)
    return theta

# Train the model using SGD
theta = sgd(X_b, y, theta, learning_rate, n_epochs)

print("Trained weights:", theta)