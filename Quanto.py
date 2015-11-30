import numpy.random as rnd
from OptionValuation import *
from American import *


class Quanto(OptionValuation):
    """ Quanto option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, method='BS', nsteps=None, npaths=None, keep_hist=False, vol_ex=0.0, correlation=0.0, seed=1,
                deg=5):
        """ Wrapper function that calls appropriate valuation method.

        User passes parameters to calc_px, which saves them to local PriceSpec object
        and calls specific pricing function (_calc_BS,...).
        This makes significantly less docstrings to write, since user is not interfacing pricing functions,
        but a wrapper function calc_px().

        Calculates the value of a plain vanilla Quanto option.

        Parameters
        ----------
        correlation : float
                LT. The correlation between the asset and the exchange rate.
        vol_ex : float
                LT. Volatility of the exchange rate.
        method : str
                Required. Indicates a valuation method to be used: 'BS', 'LT', 'MC', 'FD'
        nsteps : int
                LT, MC, FD methods require number of times steps
        npaths : int
                MC, FD methods require number of simulation paths
        keep_hist : bool
                If True, historical information (trees, simulations, grid) are saved in self.px_spec object.
        seed: int
                MC random seed
        deg: int
                Degrees in LSM MC method.

        Returns
        -------
        self : Quanto

        Notes
        -----

        Examples
        --------

        LT Examples
        -----------

        Example #1: Calculate the price of a Quanto option.

        >>> s = Stock(S0=1200, vol=.25, q=0.015)
        >>> o = Quanto(ref=s, right='call', K=1200, T=2, rf_r=.03, frf_r=0.05)
        >>> o.pxLT(nsteps=10, vol_ex=0.12, correlation=0.2, keep_hist=True)
        176.29699901666874

        Example #2: Access the tree

        >>> o.px_spec.ref_tree  # doctest: +ELLIPSIS
        ((1199.9999999999995,), ... 3670.601501648697))

        Example #3 (verifiable from Hull ch.30, ex.30.5 (p.701-702)): Calculate the price of a Quanto option.
        Uncomment to run (number of paths required is too high for doctests)

        # >>> s = Stock(S0=1200, vol=.25, q=0.015)
        # >>> o = Quanto(ref=s, right='call', K=1200, T=2, rf_r=.03, frf_r=0.05)
        # >>> o.pxLT(nsteps=100, vol_ex=0.12, correlation=0.2, keep_hist=True)
        179.82607364328157

        Example #4 (verifiable from Hull ch.30, problem.30.9.b (p.704)): Calculate the price of a Quanto option.
        Uncomment to run (number of paths required is too high for doctests)

        # >>> s = Stock(S0=400, vol=.2, q=0.03)
        # >>> o = Quanto(ref=s, right='call', K=400, T=2, rf_r=.06, frf_r=0.04)
        # >>> o.pxLT(nsteps=100, vol_ex=0.06, correlation=0.4)
        57.50700503047851

        Example #5 (plot): Convergence

        >>> s = Stock(S0=400, vol=.2, q=0.03)
        >>> o = Quanto(ref=s, right='call', K=400, T=2, rf_r=.06, frf_r=0.04)
        >>> o.plot_px_convergence()


        MC Examples
        -----------
        Calculate the price of a Quanto option using MC method. This example comes from Hull ch.30, ex.30.5 (p.701-702)
        >>> s = Stock(S0=1200, vol=.25, q=0.015)
        >>> o = Quanto(ref=s, right='call', K=1200, T=2, rf_r=.03, frf_r=0.05)
        >>> print(o.calc_px(method='MC', nsteps=100, npaths=5000,vol_ex=0.12, correlation=0.2).px_spec.px)
        179.88546563590577

        Calculate the price of a Quanto option. This example comes from Hull ch.30, problem.30.9.b (p.704)
        >>> s = Stock(S0=400, vol=.2, q=0.03)
        >>> o = Quanto(ref=s, right='call', K=400, T=2, rf_r=.06, frf_r=0.04)
        >>> o.calc_px(method='MC', nsteps=100,npaths=4000, vol_ex=0.06, correlation=0.4).px_spec.px
        57.363490258590126

        Example of option price convergence (MC method) with increasing paths
        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([o.update(T=t).calc_px(method='MC', nsteps=100, npaths=5000, vol_ex=0.12, correlation=0.2)\
        .px_spec.px for t in expiries], expiries)
        >>> O.plot(grid=1, title='MC Method: Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()


        :Authors:
            Patrick Granahan,
            Runmin Zhang
        """
        return super().calc_px(method=method, nsteps=nsteps,
                               npaths=npaths, keep_hist=keep_hist, vol_ex=vol_ex, correlation=correlation, seed=1,
                               deg=deg)

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Quanto

        :Author:
            Patrick Granahan
        """

        # Get provided parameters
        vol_ex = getattr(self.px_spec, 'vol_ex')
        correlation = getattr(self.px_spec, 'correlation')
        keep_hist = getattr(self.px_spec, 'keep_hist')
        n = getattr(self.px_spec, 'nsteps')

        # Compute the foreign numeraire dividend yield
        growth_rate_of_underlying = (correlation * self.ref.vol * vol_ex)
        domestic_numeraire = self.rf_r - self.ref.q
        foreign_numeraire = domestic_numeraire + growth_rate_of_underlying
        foreign_numeraire_dividend_yield = self.frf_r - foreign_numeraire

        # Once we have the foreign numeraire dividend yield calculated,
        # we can price the Quanto option using an American option with specific parameters
        stock = Stock(S0=self.ref.S0, vol=self.ref.vol, q=foreign_numeraire_dividend_yield)
        american_option = American(ref=stock, right=self.right, K=self.K, T=self.T, rf_r=self.frf_r)

        # Then we take the price spec from the American option
        self.px_spec = american_option.calc_px(method='LT', nsteps=n, keep_hist=keep_hist).px_spec

        return self

    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Quanto

        .. sectionauthor::

        Note
        ----


        """

        return self

    def _calc_MC(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Quanto

        .. sectionauthor:: Runmin Zhang

        Note
        ----
        [1] http://unicreditanduniversities.eu/uploads/assets/QuantoAdjustments_in_the_Presence_of_SV_Giese.pdf
        """


        # Verify the input
        try: deg = self.px_spec.deg
        except TypeError: deg = 5

        vol_ex = getattr(self.px_spec, 'vol_ex')  # Volatility of the exchange rate
        correlation = getattr(self.px_spec, 'correlation')  # Correlation between asset and exchange rate
        n_steps = getattr(self.px_spec, 'nsteps', 3) # # of steps
        n_paths = getattr(self.px_spec, 'npaths', 5000) # of paths in MC simulation
        _ = self


        # Compute the foreign numeraire dividend yield
        growth_rate_of_underlying = (correlation * self.ref.vol * vol_ex)
        domestic_numeraire = self.rf_r - self.ref.q
        foreign_numeraire = domestic_numeraire + growth_rate_of_underlying
        foreign_numeraire_dividend_yield = self.frf_r - foreign_numeraire

        # Once we have the foreign numeraire dividend yield calculated,
        # Follow the LT method. We can price the Quanto option using an American option with specific parameters.

        dt = _.T / n_steps; df = np.exp(-_.frf_r * dt)
        signCP = 1 if _.right.lower()[0] == 'c' else -1

        rnd.seed(_.px_spec.seed)
        S = _.ref.S0 * np.exp\
            (np.cumsum(rnd.normal((_.frf_r-foreign_numeraire_dividend_yield- 0.5 * _.ref.vol ** 2) * dt,\
                                  _.ref.vol * np.sqrt(dt), (n_steps + 1, n_paths)), axis=0)); S[0] = _.ref.S0
        payout = np.maximum(signCP * (S - _.K), 0); v = np.copy(payout)  # terminal payouts

        for i in range(n_steps - 1, 0, -1):    # American Option Valuation by Backwards Induction
            rg = np.polyfit(S[i], v[i + 1] * df, deg)      # fit 5th degree polynomial to PV of current inner values
            C = np.polyval(rg, S[i])              # continuation values.
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
