"""Price a European or American option by a binomial tree"""
from stockOption import StockOption
import math
import numpy as np

class BinomialTreeOption(StockOption):
    
    def _setup_parameters_(self):
        self.u = 1 + self.pu #Expected value in the up state
        self.d = 1 - self.pd #Expected value in the down state
        self.qu = (math.exp((self.r-self.div)*self.dt) - self.d)/(self.u-self.d)
        self.qd = 1 - self.qu
    
    #uses 2D numpy array to store expeced returns of stock prices for all time steps. used to calculate payoff values from exercising the option at each period
    def _initialize_stock_price_(self):
        #Initialize a 2D tree at T=0
        self.STs = [np.array([self.S0])]
        
        #simulate the possible stock prices path
        for i in range(self.N):
            prev_branches = self.STs[-1]
            st = np.concatenate((prev_branches*self.u,
                                [prev_branches[-1]*self.d]))
            self.STs.append(st) #add nodes at each time step

    #creates the payoff tree as a numpy array, starting w the instrinsic option values at maturity
    def _initialize_payoffs_tree_(self):
        #The payoffs when option expires
        return np.maximum(0, (self.STs[self.N]-self.K) if self.is_call else (self.K - self.STs[self.N]))

    #private method that returns the maximum payoff values bw execrising the American option early & not at all
    def __check_early_exercise__(self, payoffs, node):
        early_ex_payoff = (self.STs[node] - self.K) if self.is_call else (self.K - self.STs[node])
        
        return np.maximum(payoffs, early_ex_payoff)

    #checks how optimal exercising the option @ every time step is
    def _traverse_tree_(self, payoffs):
        for i in reversed(range(self.N)):
            #payoffs from not exercising the option
            payoffs = (payoffs[:-1] * self.qu + payoffs[1:] * self.qd) * self.df
                                
            #Payoffs from exercising, for American options
            if not self.is_euro:
                payoffs = self.__check_early_exercise__(payoffs, i)
        return payoffs
                                 
    def __begin_tree_traversal__(self):
        payoffs = self._initialize_payoffs_tree_()
        return self._traverse_tree_(payoffs)
                                 
    def price(self):
        self._setup_parameters_()
        self._initialize_stock_price_()
        payoffs = self.__begin_tree_traversal__()

        return payoffs[0]