import numpy as np
import math
import scipy.optimize
import matplotlib.pyplot as plt

try: from qfrm.OptionValuation import *  # production:  if qfrm package is installed
except:   from OptionValuation import *  # development: if not installed and running from source

try: from qfrm.European import *  # production:  if qfrm package is installed
except:   from European import *  # development: if not installed and running from source


class ContingentPremium(OptionValuation):
    """ Contingent Premium Option Valuation Class

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, Seed=0, method='BS', nsteps=None, npaths=None, keep_hist=False):
        """ Wrapper function that calls appropriate valuation method.

        All parameters of ``calc_px`` are saved to local ``px_spec`` variable of class ``PriceSpec`` before
        specific pricing method (``_calc_BS()``,...) is called.
        An alternative to price calculation method ``.calc_px(method='BS',...).px_spec.px``
        is calculating price via a shorter method wrapper ``.pxBS(...)``.
        The same works for all methods (BS, LT, MC, FD).

        Parameters
        ----------
        Seed : int
                Required for the MC method in which random paths are generated (choose a single seed to reproduce
                results)
        method : str
                Required. Indicates a valuation method to be used:
                ``BS``: Black-Scholes Merton calculation
                ``LT``: Lattice tree (such as binary tree)
                ``MC``: Monte Carlo simulation methods
                ``FD``: finite differencing methods
        nsteps : int
                LT, MC, FD methods require number of times steps
        npaths : int
                MC, FD methods require number of simulation paths
        keep_hist : bool
                If ``True``, historical information (trees, simulations, grid) are saved in ``self.px_spec`` object.

        Returns
        -------
        self : ContingentPremium
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object).


        Notes
        -----
        A Contingent Premium option is simply an European option except that the premium is paid at the end of the
        contract instead of
        the beginning as done in a normal European option. Additionally, the premium is only paid if the asset hits the
        strike price at TTM (i.e. above for call, below for put).

        See page 598 and 599 in Hull for explanation.

        Examples
        -------
        >>> from qfrm import *
        >>> s = Stock(S0=1/97, vol=.2, q=.032)
        >>> o = ContingentPremium(ref=s, right='call', K=1/100, T=.25, rf_r=.059)
        >>> o.pxLT(nsteps=100)
        0.000997259

        >>> o.calc_px(method='LT', nsteps=100)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        ContingentPremium...px: 0.000997259...

        >>> s = Stock(S0=1/97, vol=.2, q=.032)
        >>> o = ContingentPremium(ref=s, right='call', K=1/100, T=.25, rf_r=.059)
        >>> o.pxMC(nsteps=100, npaths=100, Seed=0)
        0.000682276
        >>> o.calc_px(method='MC', nsteps=100, npaths=100, Seed=0)  # doctest: +ELLIPSIS
        ContingentPremium...px: 0.000682276...

        >>> s = Stock(S0=45, vol=.3, q=.02)
        >>> o = ContingentPremium(ref=s, right='call', K=52, T=3, rf_r=.05)
        >>> o.pxLT(nsteps=100)
        25.365713103
        >>> o.calc_px(method='LT', nsteps=100)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        ContingentPremium...px: 25.365713103...


        >>> s = Stock(S0=100, vol=.4)
        >>> o = ContingentPremium(ref=s, right='put', K=100, T=1, rf_r=.08)
        >>> o.pxMC(nsteps=100, npaths=100)
        33.079676917
        >>> o.calc_px(method='MC', nsteps=100, npaths=100)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        ContingentPremium...px: 33.079676917...

        >>> s = Stock(S0=50, vol=.2, q=.01)
        >>> strike = range(40, 61)
        >>> o = [ContingentPremium(ref=s, right='call', K=strike[i], T=1, rf_r=.05).pxLT(nsteps=100) for i in range(0, 21)]
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

        :Authors:
            Andrew Weatherly
        """
        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_BS(self):
        """Internal function for option valuation.  Black Scholes Closed Form Solution

        Returns
        -------
        self: ContingentPremium

        :Authors:
            Andrew Weatherly
        """
        d2 = (math.log(self.ref.S0 / self.K) + (self.rf_r - self.ref.q - .5 * self.ref.vol ** 2) * self.T) / (
            self.ref.vol * math.sqrt(self.T))
        d1 = d2 + self.ref.vol * math.sqrt(self.T)
        Q = self.ref.S0 * math.exp((self.rf_r - self.ref.q) * self.T) * Util.norm_cdf(d1) / \
            Util.norm_cdf(d2) - self.K

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: ContingentPremium

        :Authors:
            Andrew Weatherly

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
                           K=self.K, rf_r=self.rf_r, T=self.T).pxLT(nsteps=n, keep_hist=False)
        else:
            vanilla = European(ref=Stock(S0=self.ref.S0, vol=self.ref.vol), right=self.right,
                           K=self.K, rf_r=self.rf_r, T=self.T).pxLT(nsteps=n, keep_hist=False)

        def binary(Q):
            #  Calculate d1 and d2
            d1 = ((math.log(self.ref.S0 / self.K)) + ((self.rf_r - self.ref.q + self.ref.vol ** 2 / 2) * self.T)) / (
                self.ref.vol * math.sqrt(self.T))
            d2 = d1 - (self.ref.vol * math.sqrt(self.T))
            # Calculate the discount
            discount = Q * math.exp(-self.rf_r * self.T)
            # Compute the put and call price
            px_call = discount * Util.norm_cdf(d2)
            px_put = discount * Util.norm_cdf(-d2)
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

        :Authors:
            Andrew Weatherly <amw13@rice.edu>

        References
        ----------

        http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2contingent.pdf -
        This has verifiable example. Note that they actually calculated the example incorrectly. They had a d_1 value of
        .4771 when it was actually supposed to be .422092. You can check this on your own and recalculate the option
        price that they give. It should be roughly .00095 instead of .01146

        """
        np.random.seed(getattr(self.px_spec, 'Seed', 3))
        n = getattr(self.px_spec, 'nsteps', 3)
        npaths = getattr(self.px_spec, 'npaths', 3)

        dt = self.T / n
        df = math.exp(-(self.rf_r - self.ref.q) * dt)
        St = self.ref.S0 * np.exp(np.cumsum(np.random.normal((self.rf_r - self.ref.q - 0.5 * self.ref.vol ** 2) * dt,
                                                             self.ref.vol * math.sqrt(dt), (n + 1, npaths)), axis=0))
        St[0] = self.ref.S0
        payout = np.maximum(self.signCP * (St - self.K), 0)
        v = np.copy(payout)
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
            px_call = discount * Util.norm_cdf(d2)
            px_put = discount * Util.norm_cdf(-d2)
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

        :Authors:
            Andrew Weatherly


        """
        return self


