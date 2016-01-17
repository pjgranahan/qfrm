import math

import numpy as np
import scipy.optimize

try: from qfrm.European import *  # production:  if qfrm package is installed
except:   from European import *  # development: if not installed and running from source


class ContingentPremium(European):
    """ Contingent Premium Option Valuation Class

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, **kwargs):
        """ Wrapper function that calls appropriate valuation method.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments (``method``, ``nsteps``, ``npaths``, ``keep_hist``, ``rng_seed``, ...)
            are passed to the parent. See ``European.calc_px()`` for details.

        Returns
        -------
        self : ContingentPremium
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object).


        Notes
        -----
        A Contingent Premium option is simply a European option with a premium paid at expiry instead of at initiation.
        The premium is only paid if the asset hits the strike price at TTM (i.e. above for call, below for put).

        *References:*

        - See `OFOD <http://www-2.rotman.utoronto.ca/~hull/ofod/index.html>`_, J.C.Hull, 9ed, 2014, pp.598-599
        - Exotic Options, Ch.19, Slide 4, `missouri.edu <http://business.missouri.edu/stansfieldjj/457/PPT/Chpt019.ppt>`_
        - Pay Later Option – A very simple Structured Product `Team Latte, 2007 <http://www.risklatte.com/Articles/QuantitativeFinance/QF50.php>`_
        - Contingent Premium Options (Lecture 2, MFE5010 at NUS), `Lim Tiong Wee, 2001 <http://1drv.ms/1YZEDwg>`_
        Last link has has an incorrectly computed example. They had a d_1 value of .4771
        when it was actually supposed to be .422092. You can check this on your own and recalculate the option
        price that they give. It should be roughly .00095 instead of .01146

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
        >>> o.pxMC(nsteps=100, npaths=100, rng_seed=3)
        0.000682276
        >>> o.calc_px(method='MC', nsteps=100, npaths=100, rng_seed=3)  # doctest: +ELLIPSIS
        ContingentPremium...px: 0.000682276...

        >>> s = Stock(S0=45, vol=.3, q=.02)
        >>> o = ContingentPremium(ref=s, right='call', K=52, T=3, rf_r=.05)
        >>> o.pxLT(nsteps=100)
        25.365713103
        >>> o.calc_px(method='LT', nsteps=100)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        ContingentPremium...px: 25.365713103...


        >>> s = Stock(S0=100, vol=.4)
        >>> o = ContingentPremium(ref=s, right='put', K=100, T=1, rf_r=.08)
        >>> o.pxMC(nsteps=100, npaths=100, rng_seed=3)
        33.079676917
        >>> o.calc_px(method='MC', nsteps=100, npaths=100, rng_seed=3)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
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
        self.save2px_spec(**kwargs)
        return getattr(self, '_calc_' + self.px_spec.method.upper())()
        # self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        # return getattr(self, '_calc_' + method.upper())()

    def _calc_BS(self):
        """Internal function for option valuation.  See ``calc_px()`` for complete documentation.

        :Authors:
            Andrew Weatherly
        """
        d2 = (math.log(self.ref.S0 / self.K) + (self.rf_r - self.ref.q - .5 * self.ref.vol ** 2) * self.T) / (
            self.ref.vol * math.sqrt(self.T))
        d1 = d2 + self.ref.vol * math.sqrt(self.T)
        Q = self.ref.S0 * math.exp((self.rf_r - self.ref.q) * self.T) * Util.norm_cdf(d1) / \
            Util.norm_cdf(d2) - self.K

    def _calc_LT(self):
        """ Internal function for option valuation. See ``calc_px()`` for complete documentation.

        :Authors:
            Andrew Weatherly
        """

        #Verify Input
        assert self.right in ['call', 'put'], 'right must be "call" or "put" '
        assert self.ref.vol > 0, 'vol must be >=0'
        assert self.K > 0, 'K must be > 0'
        assert self.T > 0, 'T must be > 0'
        assert self.ref.S0 >= 0, 'S must be >= 0'
        assert self.rf_r >= 0, 'r must be >= 0'

        keep_hist = self.px_spec.keep_hist # getattr(self.px_spec, 'keep_hist', False)
        n = self.px_spec.nsteps # getattr(self.px_spec, 'nsteps', 3)
        _ = self._LT_specs()

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
        """Internal function for option valuation. See ``calc_px()`` for complete documentation.

         Monte Carlo Simulation Numerical Method

        :Authors:
            Andrew Weatherly
        """
        np.random.seed(self.px_spec.rng_seed)
        n = self.px_spec.nsteps
        npaths = self.px_spec.npaths

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


