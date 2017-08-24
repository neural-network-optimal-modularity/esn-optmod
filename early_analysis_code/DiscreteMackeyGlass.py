"""
Impliments the mackey-glass equations, which are nonlinear time-delay euqations that can
exhibit chaotic dynamics.

The following is the equation:

dx/dt = beta * (x_tau / (1 + (x_tau)^n) ) - gamma * x
beta, gamma, n > 0

x_tau is the value of the system at time t - tau

Author: Nathaniel Rodriguez
"""
from collections import deque
import sys
import copy

class MackeyGlassSystem(object):

    def __init__(self, initial_conditions, beta, gamma, n, tau):
        """
        tau - integer
        initial_conditions should include current and t - tau conditions
        """

        if initial_conditions == "default":
            initial_conditions = [ 0.5 for i in xrange(tau + 1)]

        # ICs for window going back tau [tau, 0]
        self.initial_conditions = deque(initial_conditions)
        self.window = copy.copy(self.initial_conditions)

        # system parameters
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.n = float(n)
        self.tau = int(tau)

        # Check tau has the correct window
        if len(self.window) != int(self.tau + 1):
            print "Error: window is the wrong size for given tau."
            print "Window: ", len(self.window), "Needed Window: ", int(self.tau + 1), " Tau", self.tau 
            sys.exit()

        # Current state
        self.x = self.window[-1]

    def __call__(self):
        """
        Return next value in system
        """

        # update sys
        self.x += (self.beta * (self.window[0] / (1. + self.window[0]**self.n) ) - self.gamma * self.x)
        # update window
        self.window.popleft()
        self.window.append(self.x)

        return self.x

    def generate_series(self, size):
        """
        Return a list with the time-series of the model from t=0
        """

        return [ self() for i in xrange(size) ]

    def reset(self):
        self.window = copy.copy(self.initial_conditions)
        self.x = self.window[-1]

    def set_window(self, new_window):
        self.window = copy.copy(new_window)
        self.x = self.window[-1]

if __name__ == '__main__':
    """
    for testing
    """
    import matplotlib.pyplot as plt 
    import numpy as np

    time = np.arange(0,1000)
    tau = 17
    ic = np.random.uniform(0.0,1.0, size=tau + 1)
    macky = MackeyGlassSystem(ic, 0.2, 0.1, 10.0, tau)
    response = macky.generate_series(len(time))
    f, ax = plt.subplots(2)
    ax[0].plot(time[300:], np.tanh(np.array(response) - 1.)[300:], color='black')
    ax[0].set_xlabel("$t$")
    ax[0].set_ylabel("$x(t)$")
    ax[1].plot(response[int(tau + 1) + 300:], response[300:-int(tau + 1)], color='black')
    ax[1].set_xlabel("$x(t)$")
    ax[1].set_ylabel("$x(t-\\tau)$")
    # plt.plot(response[int(tau + 1):], response[:-int( + 1)], color='black')
    # plt.xlabel("x(t)")
    # plt.ylabel("x(t-tau)")
    plt.savefig("glass_test.png",dpi=300)