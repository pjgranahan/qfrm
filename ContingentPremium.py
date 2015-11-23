import numpy as np
import math
from OptionValuation import *
from European import *
from Binary import *
from scipy.optimize import root
from scipy.stats import norm
from math import log, exp, sqrt

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
        >>> s = Stock(S0=50, vol=.3)
        >>> o = ContingentPremium(ref=s, right='put', K=52, T=2, rf_r=.05)
        >>> o.calc_px(method='LT', nsteps=2, keep_hist=True).px_spec.px
        8.209653750647185

        >>> o.px_spec.ref_tree
        ((50.000000000000014,), (37.0409110340859, 67.49294037880017), (27.440581804701324, 50.00000000000001, 91.10594001952546))

        >>> o.calc_px(method='LT', nsteps=2, keep_hist=False)
        Boston
        K: 52
        T: 2
        _right: put
        _signCP: -1
        frf_r: 0
        px_spec: PriceSpec
          LT_specs:
            a: 1.0512710963760241
            d: 0.7408182206817179
            df_T: 0.9048374180359595
            df_dt: 0.951229424500714
            dt: 1.0
            p: 0.5097408651817704
            u: 1.3498588075760032
          keep_hist: false
          method: LT
          nsteps: 2
          px: 8.209653750647185
          sub_method: Binomial Tree
        ref: Stock
          S0: 50
          curr: -
          desc: -
          q: 0
          tkr: -
          vol: 0.3
        rf_r: 0.05
        seed0: -
        <BLANKLINE>

        >>> s = Stock(S0=45, vol=.3, q=.02)
        >>> o = ContingentPremium(ref=s, right='call', K=52, T=3, rf_r=.05)
        >>> o.calc_px(method='LT', nsteps=10, keep_hist=True).px_spec.px
        9.272539685915113
        >>> o.calc_px(method='LT', nsteps=10, keep_hist=False)
        Boston
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
          px: 9.272539685915113
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
        >>> o.calc_px(method='LT', nsteps=5, keep_hist=True).px_spec.px
        14.256042662176432
        >>> o.calc_px(method='LT', nsteps=5, keep_hist=False)
        Boston
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
          px: 14.256042662176432
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

        http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2contingent.pdf - This has verifiable example
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

        print(vanilla)
        option_price = root(binary, vanilla, method='hybr')
        option_price = option_price.x
        print(option_price)
        nsteps = n
        par = _
        S = np.zeros((nsteps + 1, nsteps + 1))
        Val = np.zeros((nsteps + 1, nsteps + 1))
        S[0, 0] = self.ref.S0
        if self.right == 'put':
            for i in range(1, nsteps + 1):
                for j in range(0, nsteps + 1):
                    if j <= i:
                        S[i, j] = self.ref.S0 * (par['u'] ** j) * (par['d'] ** (i - j))
                        if i == nsteps and S[i, j] < self.K:
                            Val[i, j] = self.K - S[i, j] + option_price
            for i in range(nsteps - 1, -1, -1):
                for j in range(nsteps + 1, -1, -1):
                    if j <= i:
                        Val[i, j] = round(par['df_dt'] * (par['p'] * Val[i + 1, j + 1] + (1 - par['p']) * Val[i + 1, j]), 4)
        else:
            for i in range(1, nsteps + 1):
                for j in range(0, nsteps + 1):
                    if j <= i:
                        S[i, j] = self.ref.S0 * (par['u'] ** j) * (par['d'] ** (i - j))
                        if i == nsteps and S[i, j] > self.K:
                            Val[i, j] = S[i, j] - self.K - option_price
            for i in range(nsteps - 1, -1, -1):
                for j in range(nsteps + 1, -1, -1):
                    if j <= i:
                        Val[i, j] = round(par['df_dt'] * (par['p'] * Val[i + 1, j + 1] + (1 - par['p']) * Val[i + 1, j]), 4)

        O = Val[0, 0]
        self.px_spec.add(px=float(Util.demote(O)), method='LT', sub_method='Binomial Tree',
                        LT_specs=_)

        # self.px_spec = PriceSpec(px=float(Util.demote(O)), method='LT', sub_method='binomial tree; Hull Ch.13',
        #                 LT_specs=_, ref_tree = S_tree if save_tree else None, opt_tree = O_tree if save_tree else None)
        return self

"""
if __name__ == "__main__":
    import doctest
    doctest.testmod()
"""
s = Stock(S0=100, vol=.2, q=.05)
o = ContingentPremium(ref=s, right='call', K=100, T=.25, rf_r=.1)
#print(o.calc_px(method='LT', nsteps=200, keep_hist=True).px_spec.px)
print(o.calc_px(method='LT', nsteps=1999))

