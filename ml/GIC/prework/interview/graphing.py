"""
The standard normal distribution is a specific type of normal distribution with a mean of 0 and a standard deviation of 1. It is a continuous probability distribution that is symmetric around the mean, with its shape defined by the bell curve (also known as the Gaussian curve).

### Key Characteristics
1. **Mean (μ)**: The average value, which is 0 for the standard normal distribution.
2. **Standard Deviation (σ)**: A measure of the spread or dispersion of the distribution, which is 1 for the standard normal distribution.
3. **Symmetry**: The distribution is symmetric around the mean.
4. **Bell Curve Shape**: The probability density function (PDF) forms a bell-shaped curve.

### Probability Density Function (PDF)
The PDF of the standard normal distribution is given by:
\[ f(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}} \]
where:
- \( x \) is the variable.
- \( \pi \) is the mathematical constant Pi.
- \( e \) is the base of the natural logarithm.

### Example
Here is an example of generating and plotting the standard normal distribution using Python and Matplotlib:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data points from the standard normal distribution
data = np.random.randn(1000)

# Plot the histogram of the data
plt.hist(data, bins=30, density=True, alpha=0.6, color='g')

# Plot the standard normal distribution curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
plt.plot(x, p, 'k', linewidth=2)

title = "Standard Normal Distribution"
plt.title(title)
plt.show()
```

### Explanation
- **Data Generation**: [`np.random.randn(1000)`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fpeteryxu%2FCODE%2Fsagemaker%2Fgoogleml-crashcourse%2Fml%2FGIC%2Fprework%2Finterview%2Fsgd.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A10%2C%22character%22%3A6%7D%7D%5D%2C%227294718b-31e7-4587-884e-f5ce7e7b3ac9%22%5D "Go to definition") generates 1000 samples from the standard normal distribution.
- **Histogram**: The histogram shows the distribution of the generated data.
- **PDF Curve**: The black line represents the PDF of the standard normal distribution.

### In Your Code
In the provided code snippet:
```python
# Add a bias term (column of ones) to the input data
X_b = np.c_[np.ones((100, 1)), X]
print(X_b)
```
- The standard normal distribution is used to generate random numbers for the target values and to initialize the parameters (weights).

### Summary
- The standard normal distribution is a normal distribution with a mean of 0 and a standard deviation of 1.
- It is symmetric around the mean and forms a bell-shaped curve.
- It is commonly used in statistics and machine learning for various purposes, including generating random numbers and initializing parameters.


"""

"""
Standard deviation is a measure of the amount of variation or dispersion in a set of values. It quantifies how much the values in a dataset deviate from the mean (average) of the dataset. A low standard deviation indicates that the values are close to the mean, while a high standard deviation indicates that the values are spread out over a wider range.

"""

import numpy as np
import matplotlib.pyplot as plt

# Generate data points from the standard normal distribution
data = np.random.randn(1000)
print(data)

# Plot the histogram of the data
plt.hist(data, bins=30, density=True, alpha=0.6, color='g')

# Plot the standard normal distribution curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
plt.plot(x, p, 'k', linewidth=2)

title = "Standard Normal Distribution"
plt.title(title)
plt.show()