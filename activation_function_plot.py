import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as stats

def sigmoid(x, a=1.0, b=1.0, c=0.0, d=0.0, e=1.0):
    """
    numpy Vector/matrix friendly sigmoid function
    """
    return a / (b + np.exp(-e*(x-c))) + d

def rectified_linear_unit(x, threshold=0.0, scale=1.0):
    """
    """

    return scale * stats.threshold(x, threshmin=threshold, newval=0)

def heaviside(x, threshold=0.0, newval=1.0):
    """
    """

    return newval * (x > threshold)

if __name__ == '__main__':

    x = np.linspace(-6,6,100)
    y = sigmoid(x, c=1, e=10)#heaviside(x, 0) #rectified_linear_unit(x) #
    # plt.plot(x,y, lw=2)
    # y = sigmoid(x, c=1.0, e=10, a=1, d=-0.2)
    plt.plot(x,y, lw=2, ls='-')
    plt.grid(True)
    plt.axvline(0, ls='--', color='black', lw=1)
    # plt.axhline(0, ls='--', color='black', lw=2)
    plt.xlim(-6,6)
    plt.tight_layout()
    plt.ylabel('y')
    plt.xlabel('x')
    plt.savefig("test.png", dpi=300)