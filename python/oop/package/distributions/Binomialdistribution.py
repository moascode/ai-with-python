import math
import matplotlib.pyplot as plt
from scipy.special import comb
from .Generaldistribution import Distribution

class Binomial(Distribution):
    """ Binomial distribution class for calculating and 
    visualizing a Binomial distribution.
    
    Attributes:
        mean (float) representing the mean value of the distribution
        stdev (float) representing the standard deviation of the distribution
        data_list (list of floats) a list of floats to be extracted from the data file
        p (float) representing the probability of an event occurring
        n (int) the total number of trials
    
    
    TODO: Fill out all TODOs in the functions below
            
    """
    
    #       A binomial distribution is defined by two variables: 
    #           the probability of getting a positive outcome
    #           the number of trials
    
    #       If you know these two values, you can calculate the mean and the standard deviation
    #       
    #       For example, if you flip a fair coin 25 times, p = 0.5 and n = 25
    #       You can then calculate the mean and standard deviation with the following formula:
    #           mean = p * n
    #           standard deviation = sqrt(n * p * (1 - p))
    
    #       
    
    def __init__(self, prob=.5, size=20):
        
        # TODO: store the probability of the distribution in an instance variable p
        # TODO: store the size of the distribution in an instance variable n
        self.p = prob
        self.n = size
        
        # TODO: Now that you know p and n, you can calculate the mean and standard deviation
        #       Use the calculate_mean() and calculate_stdev() methods to calculate the
        #       distribution mean and standard deviation
        #
        #       Then use the init function from the Distribution class to initialize the
        #       mean and the standard deviation of the distribution
        #
        #       Hint: You need to define the calculate_mean() and calculate_stdev() methods
        #               farther down in the code starting in line 55. 
        #               The init function can get access to these methods via the self
        #               variable.   
        Distribution.__init__(self, self.calculate_mean(), self.calculate_stdev())
    
    def calculate_mean(self):
    
        """Function to calculate the mean from p and n
        
        Args: 
            None
        
        Returns: 
            float: mean of the data set
    
        """
        
        # TODO: calculate the mean of the Binomial distribution. Store the mean
        #       via the self variable and also return the new mean value
                
        return self.p * self.n 



    def calculate_stdev(self):

        """Function to calculate the standard deviation from p and n.
        
        Args: 
            None
        
        Returns: 
            float: standard deviation of the data set
    
        """
        
        # TODO: calculate the standard deviation of the Binomial distribution. Store
        #       the result in the self standard deviation attribute. Return the value
        #       of the standard deviation.
        return math.sqrt(self.n * self.p * (1 - self.p))
        
        
        
    def replace_stats_with_data(self):
    
        """Function to calculate p and n from the data set
        
        Args: 
            None
        
        Returns: 
            float: the p value
            float: the n value
    
        """        
        
        # TODO: The read_data_file() from the Generaldistribution class can read in a data
        #       file. Because the Binomaildistribution class inherits from the Generaldistribution class,
        #       you don't need to re-write this method. However,  the method
        #       doesn't update the mean or standard deviation of
        #       a distribution. Hence you are going to write a method that calculates n, p, mean and
        #       standard deviation from a data set and then updates the n, p, mean and stdev attributes.
        #       Assume that the data is a list of zeros and ones like [0 1 0 1 1 0 1]. 
        #
        #       Write code that: 
        #           updates the n attribute of the binomial distribution
        #           updates the p value of the binomial distribution by calculating the
        #               number of positive trials divided by the total trials
        #           updates the mean attribute
        #           updates the standard deviation attribute
        #
        #       Hint: You can use the calculate_mean() and calculate_stdev() methods
        #           defined previously.
        self.n = len(self.data)
        self.p = self.data.count(1)/self.n
        self.mean = self.calculate_mean()
        self.stdev = self.calculate_stdev()
        return (self.p, self.n)
            
        
    def plot_bar(self):
        """Function to output a histogram of the instance variable data using 
        matplotlib pyplot library.
        
        Args:
            None
            
        Returns:
            None
        """
            
        # TODO: Use the matplotlib package to plot a bar chart of the data
        #       The x-axis should have the value zero or one
        #       The y-axis should have the count of results for each case
        #
        #       For example, say you have a coin where heads = 1 and tails = 0.
        #       If you flipped a coin 35 times, and the coin landed on
        #       heads 20 times and tails 15 times, the bar chart would have two bars:
        #       0 on the x-axis and 15 on the y-axis
        #       1 on the x-axis and 20 on the y-axis
        
        #       Make sure to label the chart with a title, x-axis label and y-axis label
        zero_count = self.data.count(0)
        one_count = self.data.count(1)
        
        plt.bar([0,1], [zero_count, one_count])
        plt.title('Binomial Count of Results')
        plt.xlabel('Value')
        plt.ylabel('count')
        plt.show()
        
    def pdf(self, k):
        """Probability density function calculator for the gaussian distribution.
        
        Args:
            k (float): point for calculating the probability density function
            
        
        Returns:
            float: probability density function output
        """
        
        # TODO: Calculate the probability density function for a binomial distribution
        #  For a binomial distribution with n trials and probability p, 
        #  the probability density function calculates the likelihood of getting
        #   k positive outcomes. 
        # 
        #   For example, if you flip a coin n = 60 times, with p = .5,
        #   what's the likelihood that the coin lands on heads 40 out of 60 times?
        
        return (comb(self.n, k)) * (self.p ** k) * ((1 - self.p) ** (self.n - k))         

    def plot_bar_pdf(self):

        """Function to plot the pdf of the binomial distribution
        
        Args:
            None
        
        Returns:
            list: x values for the pdf plot
            list: y values for the pdf plot
            
        """
    
        # TODO: Use a bar chart to plot the probability density function from
        # k = 0 to k = n
        
        #   Hint: You'll need to use the pdf() method defined above to calculate the
        #   density function for every value of k.
        
        #   Be sure to label the bar chart with a title, x label and y label

        #   This method should also return the x and y values used to make the chart
        #   The x and y values should be stored in separate lists
        
        x = range(self.n + 1)
        y = []
        for k in x:
            y.append(self.pdf(k))
            
        plt.bar(x, y)
        plt.xlabel('Number of Successes (k)')
        plt.ylabel('Probability')
        plt.title('Binomial PDF')
        plt.show()
        
        return x,y
                
    def __add__(self, other):
        
        """Function to add together two Binomial distributions with equal p
        
        Args:
            other (Binomial): Binomial instance
            
        Returns:
            Binomial: Binomial distribution
            
        """
        
        try:
            assert self.p == other.p, 'p values are not equal'
        except AssertionError as error:
            raise
        
        # TODO: Define addition for two binomial distributions. Assume that the
        # p values of the two distributions are the same. The formula for 
        # summing two binomial distributions with different p values is more complicated,
        # so you are only expected to implement the case for two distributions with equal p.
        
        # the try, except statement above will raise an exception if the p values are not equal
        
        # Hint: You need to instantiate a new binomial object with the correct n, p, 
        #   mean and standard deviation values. The __add__ method should return this
        #   new binomial object.
        
        #   When adding two binomial distributions, the p value remains the same
        #   The new n value is the sum of the n values of the two distributions.
        
        new_binomial = Binomial()
        new_binomial.p = self.p
        new_binomial.n = self.n + other.n
        new_binomial.calculate_mean()
        new_binomial.calculate_stdev()
        
        return new_binomial
        
        
    def __repr__(self):
    
        """Function to output the characteristics of the Binomial instance
        
        Args:
            None
        
        Returns:
            string: characteristics of the Gaussian
        
        """
        
        # TODO: Define the representation method so that the output looks like
        #       mean 5, standard deviation 4.5, p .8, n 20
        #
        #       with the values replaced by whatever the actual distributions values are
        #       The method should return a string in the expected format
    
        return "mean {}, standard deviation {:.2f}, p {:.2f} n {}".format(self.mean, self.stdev, self.p, self.n)
    
