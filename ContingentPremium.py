import numpy as np
import math
from OptionValuation import *
from European import *
import scipy.optimize
import scipy.stats
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
        A Contingent Premium option is simply an European option except that the premium is paid at the end of the
        contract instead of
        the beginning as done in a normal European option. Additionally, the premium is only paid if the asset hits the
        strike price at TTM (i.e. above for call, below for put).

        See page 598 and 599 in Hull for explanation.

        Examples
        -------
        >>> s = Stock(S0=1/97, vol=.2, q=.032)
        >>> o = ContingentPremium(ref=s, right='call', K=1/100, T=.25, rf_r=.059)
        >>> o.calc_px(method='LT', nsteps=2000, keep_hist=False).px_spec.px
        0.000995798782229753

        >>> o.calc_px(method='LT', nsteps=2000, keep_hist=False)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        ContingentPremium...px: 0.000995799...


        >>> s = Stock(S0=45, vol=.3, q=.02)
        >>> o = ContingentPremium(ref=s, right='call', K=52, T=3, rf_r=.05)
        >>> o.calc_px(method='LT', nsteps=10, keep_hist=False).px_spec.px
        25.921951519642672
        >>> o.calc_px(method='LT', nsteps=10, keep_hist=False)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        ContingentPremium...px: 25.92195152...


        >>> s = Stock(S0=100, vol=.4)
        >>> o = ContingentPremium(ref=s, right='put', K=100, T=1, rf_r=.08)
        >>> o.calc_px(method='LT', nsteps=5, keep_hist=False).px_spec.px
        26.877929027736258
        >>> o.calc_px(method='LT', nsteps=5, keep_hist=False)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        ContingentPremium...px: 26.877929028...

        >>> s = Stock(S0=50, vol=.2, q=.01)
        >>> strike = range(40, 61)
        >>> o = [ContingentPremium(ref=s, right='call', K=strike[i], T=1, rf_r=.05).calc_px(method='LT', nsteps=100)\
        .px_spec.px for i in range(0, 21)]
        >>> plt.plot(strike, o, label='Changing Strike') # doctest: +ELLIPSIS
        [<matplotlib.lines.Line2D object at...
        >>> plt.xlabel('Strike Price') # doctest: +ELLIPSIS
        <matplotlib.text.Text object at...
        >>> plt.ylabel("Option Price") # doctest: +ELLIPSIS
        <matplotlib.text.Text object at...
        >>> plt.legend(loc='best') # doctest: +ELLIPSIS
        <matplotlib.legend.Legend object at...
        >>> plt.title("Changing Strike Price") # doctest: +ELLIPSIS
        <matplotlib.text.Text object at...
        >>> plt.show()
        """
        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_BS(self):
        """Internal function for option valuation.  Black Scholes Closed Form Solution

        Returns
        -------
        self: ContingentPremium

        .. sectionauthor:: Andrew Weatherly


        """

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: ContingentPremium

        .. sectionauthor:: Andrew Weatherly

        References
        -------
        http://business.missouri.edu/stansfieldjj/457/PPT/Chpt019.ppt - Slide 4
        http://www.risklatte.com/Articles/QuantitativeFinance/QF50.php

        http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2contingent.pdf -
        This has verifiable example. Note that they actually calculated the example incorrectly. They had a d_1 value of
        .4771 when it was actually supposed to be .422092. You can check this on your own and recalculate the option
        price that they give. It should be roughly .00095 instead of .01146
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
                           K=self.K, rf_r=self.rf_r, T=self.T).calc_px(method='LT', nsteps=n, keep_hist=False)\
                .px_spec.px
        else:
            vanilla = European(ref=Stock(S0=self.ref.S0, vol=self.ref.vol), right=self.right,
                           K=self.K, rf_r=self.rf_r, T=self.T).calc_px(method='LT', nsteps=n, keep_hist=False)\
                .px_spec.px

        def binary(Q):
            #  Calculate d1 and d2
            d1 = ((math.log(self.ref.S0 / self.K)) + ((self.rf_r - self.ref.q + self.ref.vol ** 2 / 2) * self.T)) / (
                self.ref.vol * math.sqrt(self.T))
            d2 = d1 - (self.ref.vol * math.sqrt(self.T))
            # Calculate the discount
            discount = Q * math.exp(-self.rf_r * self.T)
            # Compute the put and call price
            px_call = discount * scipy.stats.norm.cdf(d2)
            px_put = discount * scipy.stats.norm.cdf(-d2)
            px = px_call if self.signCP == 1 else px_put if self.signCP == -1 else None
            return px - vanilla

        option_price = scipy.optimize.root(binary, vanilla, method='hybr') #finds the binary price that we need
        option_price = option_price.x
        self.px_spec.add(px=float(Util.demote(option_price)), method='LT', sub_method='Binomial Tree',
                        LT_specs=_)

        return self

    def _calc_MC(self):
        """Internal function for option valuation.  Monte Carlo Simulation Numerical Method

        Returns
        -------
        self: ContingentPremium

        .. sectionauthor:: Andrew Weatherly


        """
        n = getattr(self.px_spec, 'nsteps', 3)
        npaths = getattr(self.px_spec, 'npaths', 3)
        dt = self.T / n
        df = math.exp(-self.rf_r * dt)
        S = self.ref.S0 * math.exp(math.cumsum(scipy.stats.normal((self.rf_r - 0.5 * self.ref.vol ** 2) * dt,
            self.ref.vol * math.sqrt(dt), (n + 1, npaths)), axis=0))
        S[0] = self.ref.S0
        payout = np.maximum(self.signCP * (S - K), 0); v = np.copy(payout)  # terminal payouts
        for i in range(n - 1, -1, -1):
            v[i] = v[i + 1] * df
        vanilla = np.mean(v[0])

        def binary(Q):
            #  Calculate d1 and d2
            d1 = ((math.log(self.ref.S0 / self.K)) + ((self.rf_r - self.ref.q + self.ref.vol ** 2 / 2) * self.T)) / (
                self.ref.vol * math.sqrt(self.T))
            d2 = d1 - (self.ref.vol * math.sqrt(self.T))
            # Calculate the discount
            discount = Q * math.exp(-self.rf_r * self.T)
            # Compute the put and call price
            px_call = discount * scipy.stats.norm.cdf(d2)
            px_put = discount * scipy.stats.norm.cdf(-d2)
            px = px_call if self.signCP == 1 else px_put if self.signCP == -1 else None
            return px - vanilla

        option_price = scipy.optimize.root(binary, vanilla, method='hybr') #finds the binary price that we need
        option_price = option_price.x
        self.px_spec.add(px=float(Util.demote(option_price)), method='MC', sub_method='Monte Carlo Simulation')
        return self

    def _calc_FD(self):
        """Internal function for option valuation.  Finite Difference Numerical Method

        Returns
        -------
        self: ContingentPremium

        .. sectionauthor:: Andrew Weatherly


        """
        return self


if __name__ == "__main__":
    import doctest
    doctest.testmod()

"""
s = Stock(S0=50, vol=.2, q=.01)
o = [0] * 21
strike = range(40, 61)
o = [ContingentPremium(ref=s, right='call', K=strike[i], T=1, rf_r=.05).calc_px(method='LT', nsteps=100)\
        .px_spec.px for i in range(0, 21)]

plt.plot(strike, o, label='Changing Strike')
plt.xlabel('Strike Price')
plt.ylabel("Option Price")
plt.legend(loc='best')
plt.title("Changing Strike Price")
plt.show()
"""
