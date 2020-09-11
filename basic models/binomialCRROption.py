"""
binomial euro & american both assumed that the underlying stock price 
would increase/decrease by 20% in u/d state

Cox-Ross-Rubenstein(CRR): over ST time period in risk-neutral world, 
binomial model matches mean & variance of the underlying

The following prices an option utilzing the binomial CRR option

"""

from BinomialAmericanOption import BinomialTreeOption as BTO
import math

class BinomialCRROption(BTO):
    
    #overwriting org. setup parameters to accomodate CRR model
    def _setup_parameters_(self):
        self.u = math.exp(self.sigma * math.sqrt(self.dt))
        self.d = 1.0/self.u
        self.qu = (math.exp((self.r-self.div)*self.dt) - self.d)/(self.u-self.d)
        self.qd = 1-self.qu