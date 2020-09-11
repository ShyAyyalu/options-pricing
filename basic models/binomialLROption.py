"""
utilizes binomial tree model to approximate BS solution as the number of steps increase

key differentiators:
    -nodes don't recombine @ every alternate step
    -use inversion formula to acheive better accuracy during traversal

"""

from BinomialAmericanOption import BinomialTreeOption as BTO
import math

class BinomialLROption(BTO):
    
    def _setup_parameters_(self):
        odd_N = self.N if (self.N%2 == 1) else (self.N+1)
        d1 = (math.log(self.S0/self.K)+((self.r-self.div)+(self.sigma**2)/2.)*self.T) / (self.sigma * math.sqrt(self.T))
        d2 = (math.log(self.S0/self.K) + ((self.r-self.div) - (self.sigma**2)/2.) * self.T) / (self.sigma*math.sqrt(self.T))
        pp_2_inversion = lambda z,n: .5 + math.copysign(1,z)* math.sqrt(.25- .25*math.exp(-((z/(n+1./3.+.1/(n+1)))**2.)*(n+1./6.)))
        pbar = pp_2_inversion(d1, odd_N)

        self.p = pp_2_inversion(d2, odd_N)
        self.u = 1/self.df * pbar/self.p
        self.d = (1/self.df - self.p*self.u)/(1-self.p)
        self.qu = self.p
        self.qd = 1-self.p