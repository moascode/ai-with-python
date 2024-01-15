# Probability Calculation for Gaussian and Binomial Distributions

This Python package provides functions to calculate the probability of Gaussian and Binomial distributions. The package includes two classes, `Gaussian` and `Binomial`, each with their own methods to calculate the probability.

## Installation

To install the package, you can use pip install moascode-probability

## Usage

### Gaussian Distribution

To calculate the probability of a Gaussian distribution, you need to create an instance of the `Gaussian` class and provide the mean (`mu`) and standard deviation (`sigma`) as parameters.

```python
from moascode_probability import Gaussian

# Create a Gaussian distribution with mean 0 and standard deviation 1
gaussian = Gaussian(0, 1)

# Calculate the probability of a value
probability = gaussian.pdf(2)
print("Probability:", probability)
```

### Binomial Distribution

To calculate the probability of a Binomial distribution, you need to create an instance of the `Binomial` class and provide the probability (`prob`) and size (`size`) as parameters.

```python
from moascode_probability import Binomial

# Create a Binomial distribution with probability 0.5 and size 10
binomial = Binomial(0.5, 10)

# Calculate the probability of a value
probability = binomial.pdf(5)
print("Probability:", probability)
```