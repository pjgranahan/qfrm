import numpy.random as rnd

try: from qfrm.European import *  # production:  if qfrm package is installed
except:   from European import *  # development: if not installed and running from source

try:  from qfrm.American import *  # production:  if qfrm package is installed
except:    from American import *  # development: if not installed and running from source


class Quanto(European):
    """
    `Quanto <https://en.wikipedia.org/wiki/Quanto>`_ exotic option class.

    A quanto is a type of derivative in which the underlying is denominated in one currency, but the instrument itself
    is settled in another currency at some rate.

    Quanto options have both the strike price and underlier denominated in the foreign currency. At exercise, the value
    of the option is calculated as the option's intrinsic value in the foreign currency, which is then converted to the
    domestic currency at the fixed exchange rate.
    """

    def calc_px(self, vol_ex=0.0, corr=0.0, deg=5, **kwargs):
        """ Calculates the value of a plain vanilla Quanto option.

        Parameters
        ----------
        corr : float
                LT. The correlation between the asset and the exchange rate.
        vol_ex : float
                LT. Volatility of the exchange rate.
        deg: int
                Degrees in LSM MC method.
        kwargs : dict
            Keyword arguments (``method``, ``nsteps``, ``npaths``, ``keep_hist``, ``rng_seed``, ...)
            are passed to the parent. See ``European.calc_px()`` for details.

        Returns
        -------
        self : Quanto
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object)

        Notes
        -----

        *References:*

        - Options, Futures, and Other Derivatives, `OFOD <http://www-2.rotman.utoronto.ca/~hull/ofod/index.html>`_, J.C.Hull, 9ed, 2014, pp.699-702
        - Quanto Options - Guide and Spreadsheet. `Excel tool. Samir Khan <http://investexcel.net/quanto-options/>`_

        Examples
        --------

        **LT**

        Example #1: Calculate the price of a Quanto option.

        >>> s = Stock(S0=1200, vol=.25, q=0.015)
        >>> o = Quanto(ref=s, right='call', K=1200, T=2, rf_r=.03, frf_r=0.05)
        >>> o.pxLT(nsteps=10, vol_ex=0.12, corr=0.2, keep_hist=True)
        176.296999017

        Example #2: Show (recombining) binomial tree of progression of underlying stock prices.

        >>> o.px_spec.ref_tree  # doctest: +ELLIPSIS
        ((1199.9999999999995,), ... 3670.601501648697))

        Example #3 (verifiable from Hull [1] ch.30, ex.30.5 (p.701-702)): Calculate the price of a Quanto option.
        Uncomment to run (number of paths required is too high for doctests)

        # >>> s = Stock(S0=1200, vol=.25, q=0.015)
        # >>> o = Quanto(ref=s, right='call', K=1200, T=2, rf_r=.03, frf_r=0.05)
        # >>> o.pxLT(nsteps=100, vol_ex=0.12, corr=0.2, keep_hist=True)
        # 179.826073643

        Example #4 (verifiable from Hull [1] ch.30, problem.30.9.b (p.704)): Calculate the price of a Quanto option.
        Uncomment to run (number of paths required is too high for doctests)

        # >>> s = Stock(S0=400, vol=.2, q=0.03)
        # >>> o = Quanto(ref=s, right='call', K=400, T=2, rf_r=.06, frf_r=0.04)
        # >>> o.pxLT(nsteps=100, vol_ex=0.06, correlation=0.4)
        # 57.50700503

        Example #5 (plot): Convergence

        >>> s = Stock(S0=400, vol=.2, q=0.03)
        >>> o = Quanto(ref=s, right='call', K=400, T=2, rf_r=.06, frf_r=0.04)
        >>> o.plot_px_convergence()


        **MC**

        Next example (see OFOD J.C.Hull, Ch.30, Ex.30.5, p.701-702) yields price close to GBP 180

        >>> s = Stock(S0=1200, vol=.25, q=0.015)
        >>> o = Quanto(ref=s, right='call', K=1200, T=2, rf_r=.03, frf_r=0.05)
        >>> o.pxMC(nsteps=100, npaths=5000, vol_ex=0.12, corr=0.2, rng_seed=1)
        179.885465636

        Next example (see OFOD J.C.Hull, Ch.30, Problem 30.9b, p.704) yields price close to GBP180
        Calculate the price of a Quanto option. This example comes from Hull ch.30, problem.30.9.b (p.704)
        For an accurate result, use nsteps=100, npaths=4000

        >>> s = Stock(S0=400, vol=.2, q=0.03)
        >>> o = Quanto(ref=s, right='call', K=400, T=2, rf_r=.06, frf_r=0.04)
        >>> o.pxMC(nsteps=100, npaths=4000, vol_ex=0.06, corr=0.4, rng_seed=1)
        57.363490259

        Example of option price (MC method) with increasing time
        For an accurate result, use ``nsteps=100``, ``npaths=5000``

        >>> from pandas import Series
        >>> Ts = range(1,21)   # expiries (in years)
        >>> O = Series([o.update(T=t).pxMC(nsteps=100, npaths=1000, vol_ex=0.12, corr=0.2, rng_seed=1) for t in Ts], Ts)
        >>> O.plot(grid=1, title='Quanto MC price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()


        :Authors:
            Patrick Granahan,
            Runmin Zhang <z.runmin@gmail.com>
        """
        self.save2px_spec(vol_ex=vol_ex, corr=corr, deg=deg, **kwargs)
        return getattr(self, '_calc_' + self.px_spec.method.upper())()

    def _calc_BS(self):
        """ Internal function for option valuation.        """

        return self

    def _calc_LT(self):
        """ Internal function for option valuation. See ``calc_px()`` for full documentation.

        :Author:
            Patrick Granahan
        """

        # Get provided parameters
        _ = self.px_spec;   n, vol_ex, corr, keep_hist = _.nsteps, _.vol_ex, _.corr, _.keep_hist
        _ = self.ref;       S0, vol, q = _.S0, _.vol, _.q
        _ = self;           K, T, right, rf_r, frf_r, net_r = _.K, _.T, _.right, _.rf_r, _.frf_r, _.net_r

        # Compute the foreign numeraire dividend yield  TODO: this calculation can be extracted to a class method
        growth_rate_of_underlying = (corr * vol * vol_ex)
        domestic_numeraire = rf_r - q
        foreign_numeraire = domestic_numeraire + growth_rate_of_underlying
        foreign_numeraire_dividend_yield = frf_r - foreign_numeraire

        # Once we have the foreign numeraire dividend yield calculated,
        # we can price the Quanto option using an American option with specific parameters
        stock = Stock(S0=S0, vol=vol, q=foreign_numeraire_dividend_yield)
        american_option = American(ref=stock, right=right, K=K, T=T, rf_r=frf_r)

        # Then we take the price spec from the American option
        self.px_spec = american_option.calc_px(method='LT', nsteps=n, keep_hist=keep_hist).px_spec

        return self

    def _calc_MC(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Runmin Zhang <z.runmin@gmail.com>
        """

        # Verify the input
        try: deg = self.px_spec.deg
        except TypeError: deg = 5

        _ = self.px_spec;   n, m, vol_ex, corr, keep_hist, rng_seed = _.nsteps, _.npaths, _.vol_ex, _.corr, _.keep_hist, _.rng_seed
        _ = self.ref;       S0, vol, q = _.S0, _.vol, _.q
        _ = self;           K, T, right, rf_r, frf_r, net_r, sCP = _.K, _.T, _.right, _.rf_r, _.frf_r, _.net_r, _.signCP

        # Compute the foreign numeraire dividend yield
        growth_rate_of_underlying   = (corr * vol * vol_ex)
        domestic_numeraire          = rf_r - q
        foreign_numeraire           = domestic_numeraire + growth_rate_of_underlying
        foreign_numeraire_dividend_yield = frf_r - foreign_numeraire

        # Once we have the foreign numeraire dividend yield calculated,
        # Follow the LT method. We can price the Quanto option using an American option with specific parameters.

        dt = T / n
        df = np.exp(-frf_r * dt)

        rnd.seed(rng_seed)
        S = S0 * np.exp(np.cumsum(rnd.normal(
                (frf_r-foreign_numeraire_dividend_yield- 0.5 * vol ** 2) * dt, vol * np.sqrt(dt), (n + 1, m)), axis=0));
        S[0] = S0
        payout = np.maximum(sCP * (S - K), 0); v = np.copy(payout)  # terminal payouts

        for i in range(n - 1, 0, -1):    # American Option Valuation by Backwards Induction
            rg   = np.polyfit(S[i], v[i + 1] * df, deg)      # fit 5th degree polynomial to PV of current inner values
            C    = np.polyval(rg, S[i])              # continuation values.
            v[i] = np.where(payout[i] > C, payout[i], v[i + 1] * df)  # exercise decision
        v[0] = v[1] * df
        self.px_spec.add(px=float(np.mean(v[0])))
        return self

    def _calc_FD(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Quanto

        .. sectionauthor::

        Note
        ----

        """

        return self
