import numpy as np
import math
from OptionValuation import *
from European import *
from Binary import *
from scipy.optimize import root
from scipy.stats import norm
from math import log, exp, sqrt
import matplotlib.pyplot as plt


class ContingentPremium(OptionValuation):
    """ Boston Option Valuation Class

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, method='BS', nsteps=None, npaths=None, keep_hist=False):
        """ Wrapper function that calls appropriate valuation method.

        User passes parameters to calc_px, which saves them to local PriceSpec object
        and calls specific pricing function (_calc_BS,...).
        This makes significantly less docstrings to write, since user is not interfacing pricing functions,
        but a wrapper function calc_px().

        Parameters
        ----------
        method : str
                Required. Indicates a valuation method to be used: 'BS', 'LT', 'MC', 'FD'
        nsteps : int
                LT, MC, FD methods require number of times steps
        npaths : int
                MC, FD methods require number of simulation paths
        keep_hist : bool
                If True, historical information (trees, simulations, grid) are saved in self.px_spec object.

        Returns
        -------
        self : ContingentPremium

        .. sectionauthor:: Andrew Weatherly

        Notes
        -----
        A Contingent Premium option is simply an European option except that the premium is paid at the end of the contract
        instead of
        the beginning as done in a normal European option. Additionally, the premium is only paid if the asset hits the
        strike price at TTM (i.e. above for call, below for put).

        See page 598 and 599 in Hull for explanation.

        Examples
        -------
        >>> s = Stock(S0=1/97, vol=.2, q=.032)
        >>> o = ContingentPremium(ref=s, right='call', K=1/100, T=.25, rf_r=.059)
        >>> o.calc_px(method='LT', nsteps=2000, keep_hist=False).px_spec.px
        0.000995798782229753

        >>> o.calc_px(method='LT', nsteps=2000, keep_hist=False)
        ContingentPremium
        K: 0.01
        T: 0.25
        _right: call
        _signCP: 1
        frf_r: 0
        px_spec: PriceSpec
          LT_specs:
            a: 1.0000033750056954
            d: 0.9977664301601515
            df_T: 0.9853582483752771
            df_dt: 0.9999926250271952
            dt: 0.000125
            p: 0.5001956568255778
            u: 1.0022385698419318
          keep_hist: false
          method: LT
          nsteps: 2000
          px: 0.000995798782229753
          sub_method: Binomial Tree
        ref: Stock
          S0: 0.010309278350515464
          curr: -
          desc: -
          q: 0.032
          tkr: -
          vol: 0.2
        rf_r: 0.059
        seed0: -
        <BLANKLINE>

        >>> s = Stock(S0=45, vol=.3, q=.02)
        >>> o = ContingentPremium(ref=s, right='call', K=52, T=3, rf_r=.05)
        >>> o.calc_px(method='LT', nsteps=10, keep_hist=False).px_spec.px
        25.921951519642672
        >>> o.calc_px(method='LT', nsteps=10, keep_hist=False)
        ContingentPremium
        K: 52
        T: 3
        _right: call
        _signCP: 1
        frf_r: 0
        px_spec: PriceSpec
          LT_specs:
            a: 1.0090406217738679
            d: 0.8484732107801852
            df_T: 0.8607079764250578
            df_dt: 0.9851119396030626
            dt: 0.3
            p: 0.4863993185207596
            u: 1.1785875939211838
          keep_hist: false
          method: LT
          nsteps: 10
          px: 25.921951519642672
          sub_method: Binomial Tree
        ref: Stock
          S0: 45
          curr: -
          desc: -
          q: 0.02
          tkr: -
          vol: 0.3
        rf_r: 0.05
        seed0: -
        <BLANKLINE>

        >>> s = Stock(S0=100, vol=.4)
        >>> o = ContingentPremium(ref=s, right='put', K=100, T=1, rf_r=.08)
        >>> o.calc_px(method='LT', nsteps=5, keep_hist=False).px_spec.px
        26.877929027736258
        >>> o.calc_px(method='LT', nsteps=5, keep_hist=False)
        ContingentPremium
        K: 100
        T: 1
        _right: put
        _signCP: -1
        frf_r: 0
        px_spec: PriceSpec
          LT_specs:
            a: 1.016128685406095
            d: 0.8362016906807812
            df_T: 0.9231163463866358
            df_dt: 0.9841273200552851
            dt: 0.2
            p: 0.5002390255615027
            u: 1.1958837337268056
          keep_hist: false
          method: LT
          nsteps: 5
          px: 26.877929027736258
          sub_method: Binomial Tree
        ref: Stock
          S0: 100
          curr: -
          desc: -
          q: 0
          tkr: -
          vol: 0.4
        rf_r: 0.08
        seed0: -
        <BLANKLINE>

        """
        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Boston

        .. sectionauthor:: Andrew Weatherly

        References
        -------
        http://business.missouri.edu/stansfieldjj/457/PPT/Chpt019.ppt - Slide 4
        http://www.risklatte.com/Articles/QuantitativeFinance/QF50.php

        http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2contingent.pdf -
        This has verifiable example. Note that they actually calculated the example incorrectly. They had a d_1 value of
        .4771 when it was actually supposed to be .422092. You can check this on your own and recalculate the option price
        that they give. It should be roughly .00095 instead of .01146
        """

        #Verify Input
        assert self.right in ['call', 'put'], 'right must be "call" or "put" '
        assert self.ref.vol > 0, 'vol must be >=0'
        assert self.K > 0, 'K must be > 0'
        assert self.T > 0, 'T must be > 0'
        assert self.ref.S0 >= 0, 'S must be >= 0'
        assert self.rf_r >= 0, 'r must be >= 0'

        keep_hist = getattr(self.px_spec, 'keep_hist', False)
        n = getattr(self.px_spec, 'nsteps', 3)
        _ = self.LT_specs(n)
        if self.ref.q is not None:
            vanilla = European(ref=Stock(S0=self.ref.S0, vol=self.ref.vol, q=self.ref.q), right=self.right,
                           K=self.K, rf_r=self.rf_r, T=self.T).calc_px(method='LT', nsteps=n, keep_hist=False).px_spec.px
        else:
            vanilla = European(ref=Stock(S0=self.ref.S0, vol=self.ref.vol), right=self.right,
                           K=self.K, rf_r=self.rf_r, T=self.T).calc_px(method='LT', nsteps=n, keep_hist=False).px_spec.px

        def binary(Q):
            #  Calculate d1 and d2
            d1 = ((log(self.ref.S0 / self.K)) + ((self.rf_r - self.ref.q + self.ref.vol ** 2 / 2) * self.T)) / (
                self.ref.vol * sqrt(self.T))
            d2 = d1 - (self.ref.vol * sqrt(self.T))
            # Calculate the discount
            discount = Q * exp(-self.rf_r * self.T)
            # Compute the put and call price
            px_call = discount * norm.cdf(d2)
            px_put = discount * norm.cdf(-d2)
            px = px_call if self.signCP == 1 else px_put if self.signCP == -1 else None
            return px - vanilla

        option_price = root(binary, vanilla, method='hybr') #finds the binary price that we need
        option_price = option_price.x
        self.px_spec.add(px=float(Util.demote(option_price)), method='LT', sub_method='Binomial Tree',
                        LT_specs=_)

        return self


if __name__ == "__main__":
    import doctest
    doctest.testmod()

s = Stock(S0=50, vol=.2, q=.01)
o = [0] * 21
strike = [40] * 21
for i in range(0, 21):
    strike[i] += i
    o[i] = ContingentPremium(ref=s, right='call', K=strike[i], T=1, rf_r=.05).calc_px(method='LT', nsteps=100).px_spec.px

plt.plot(strike, o, label='Changing Strike')
plt.xlabel('Strike Price')
plt.ylabel("Option Price")
plt.legend(loc='best')
plt.title("Changing Strike Price")
plt.show()
