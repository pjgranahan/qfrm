from qfrm import *


class Quanto(OptionValuation):
    """ Quanto option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, method='BS', nsteps=None, npaths=None, keep_hist=False, vol_ex=0.0, correlation=0.0):
        """ Wrapper function that calls appropriate valuation method.

        User passes parameters to calc_px, which saves them to local PriceSpec object
        and calls specific pricing function (_calc_BS,...).
        This makes significantly less docstrings to write, since user is not interfacing pricing functions,
        but a wrapper function calc_px().

        Calculates the value of a plain vanilla Quanto option.

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
        self : Quanto

        .. sectionauthor:: Patrick Granahan

        Notes
        -----

        Examples
        -------

        >>> s = Stock(S0=1200, vol=.25, q=0.015)
        >>> o = Quanto(ref=s, right='call', K=1200, T=2, rf_r=.03, frf_r=0.05, desc='179.83, Hull p.701-702 ex.30.5')

        >>> o.calc_px(method='LT', nsteps=100, vol_ex=0.12, correlation=0.2, keep_hist=True).px_spec.px
        179.83

        >>> o.px_spec.ref_tree
        ((50.000000000000014,),
         (37.0409110340859, 67.49294037880017),
         (27.440581804701324, 50.00000000000001, 91.10594001952546))

        >>> o.calc_px(method='LT', nsteps=100, keep_hist=False)
        American
        K: 52
        T: 2
        _right: put
        _signCP: -1
        desc: 7.42840, Hull p.288
        frf_r: 0
        px_spec: qfrm.PriceSpec
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
          px: 7.42840190270483
          sub_method: binomial tree; Hull Ch.13
        ref: qfrm.Stock
          S0: 50
          curr: null
          desc: null
          q: 0
          tkr: null
          vol: 0.3
        rf_r: 0.05
        seed0: null

        """
        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist,
                                 vol_ex=vol_ex, correlation=correlation)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Quanto

        .. sectionauthor:: Patrick Granahan

        """

        # Explicit imports

        # Define parameters
        T = self.T  # Time to expiry (years)
        S0 = self.ref.S0  # Exchange rate (domestic/foreign)
        K = self.K  # Delivery Price (domestic)
        rf_r = self.rf_r  # Interest rate for Domestic Currency (% / year)
        frf_r = self.frf_r  # Interest rate for Foreign Currency (% / year)
        vol = self.ref.vol  # Volatility of Underlying (% / year)
        vol_ex = getattr(self.px_spec, 'vol_ex')  # Volatility of the exchange rate
        q = self.ref.q  # Dividend Yield of Underlying
        correlation = getattr(self.px_spec, 'correlation')  # Correlation of Asset to Domestic Currency

        # Get additional provided parameters
        keep_hist = getattr(self.px_spec, 'keep_hist', False)
        n = getattr(self.px_spec, 'nsteps', 3)
        _ = self.LT_specs(n)

        # Compute the foreign numeraire dividend yield
        growth_rate_of_underlying = (correlation * vol * vol_ex)
        domestic_numeraire = rf_r - q
        foreign_numeraire = domestic_numeraire + growth_rate_of_underlying
        foreign_numeraire_dividend_yield = frf_r - foreign_numeraire

        stock = Stock(S0=S0, vol=vol, q=foreign_numeraire_dividend_yield)
        from American import American
        american_option = American(ref=stock, right=self.right, K=K, T=T, rf_r=frf_r)

        self.px_spec = american_option.calc_px(method='LT', nsteps=n, keep_hist=keep_hist).px_spec

        return self
        # # Compute final nodes
        # S = S0 * _['d'] ** arange(n, -1, -1) * _['u'] ** arange(0, n + 1)  # terminal stock prices
        # O = maximum(self.signCP * (S - K), 0)  # terminal option payouts
        #
        # # Initialize tree structure
        # S_tree = (tuple([float(s) for s in S]),)  # use tuples of floats (instead of numpy.float)
        # O_tree = (tuple([float(o) for o in O]),)
        #
        #
        # for i in range(n, 0, -1):
        #     O = _['df_dt'] * ((1 - _['p']) * O[:i] + (_['p']) * O[1:])  # prior option prices (@time step=i-1)
        #     S = _['d'] * S[1:i + 1]  # prior stock prices (@time step=i-1)
        #     Payout = maximum(self.signCP * (S - K), 0)  # payout at time step i-1 (moving backward in time)
        #     O = maximum(O, Payout)
        #
        #     # Build tree
        #     S_tree = (tuple([float(s) for s in S]),) + S_tree
        #     O_tree = (tuple([float(o) for o in O]),) + O_tree
        #
        #
        # # Record price specs
        # self.px_spec.add(px=float(Util.demote(O)), method='LT', sub_method='binomial tree; Hull Ch.13', LT_specs=_)
        # if keep_hist:
        #     self.px_spec.add(ref_tree=S_tree, opt_tree=O_tree)
        #
        # return self

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

        .. sectionauthor::

        Note
        ----

        """
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
