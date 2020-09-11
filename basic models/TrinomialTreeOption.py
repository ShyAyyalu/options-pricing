"""
binomial vs. trinomial tree:
    each node leads to two vs three in the next time step. Middle node indicates no state change
    trinomial is similar to recombining tree when >2 time steps
    trinomial accuracy > binomial when less time steps modeled
"""

from BinomialAmericanOption import BinomialTreeOption as blo
import math
import numpy as np

class TrinomialTreeOption(blo):
    
    def _setup_parameters_(self):
        #required model calculations
        self.u = math.exp(self.sigma*math.sqrt(2.*self.dt))
        self.d = 1/self.u
        self.m = 1
        self.qu = ((math.exp((self.r-self.div)
                             *self.dt/2.) - 
                    math.exp(-self.sigma*
                             math.sqrt(self.dt/2.))) / 
                   (math.exp(self.sigma * 
                             math.sqrt(self.dt/2.)) - 
                    math.exp(-self.sigma * 
                             math.sqrt(self.dt/2.))))**2
        self.qd = ((math.exp(self.sigma *
                             math.sqrt(self.dt/2.)) - 
                    math.exp((self.r-self.div) * 
                             self.dt/2.)) / 
                   (math.exp(self.sigma * 
                             math.sqrt(self.dt/2.)) - 
                    math.exp(-self.sigma * 
                             math.sqrt(self.dt/2.))))**2.
        
        self.qm = 1 - self.qu - self.qd
        
    def _initialize_stock_price_(self):
        #Intialize a 2D tree at t=0
        self.STs = [np.array([self.S0])]
        
        for i in range(self.N):
            prev_nodes = self.STs[-1]
            self.ST = np.concatenate(
                (prev_nodes*self.u, [prev_nodes[-1]*self.m,
                                     prev_nodes[-1]*self.d]))
            self.STs.append(self.ST)
            
    def _traverse_tree_(self, payoffs):
        #traverse the tree backwards
        for i in reversed(range(self.N)):
            payoffs = (payoffs[:-2] *self.qu +
                       payoffs[1:-1] * self.qm + 
                       payoffs[2:] * self.qd) * self.df
            
            if not self.is_euro:
                payoffs = self.__check_early_exercise__(payoffs,
                                                        i)
                
        return payoffs
            