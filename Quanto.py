from qfrm import *


class Quanto(OptionValuation):
    """ Quanto option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, method='BS', nsteps=None, npaths=None, keep_hist=False, dom_for_ex=0.0, for_dom_spot=0.0,
                vol_exchange=0.0, correlation=0.0):
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
        self : Quanto

        .. sectionauthor:: Patrick Granahan

        Notes
        -----

        Examples
        -------

        """
        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist,
                                 dom_for_ex=dom_for_ex, for_dom_spot=for_dom_spot, vol_exchange=vol_exchange,
                                 correlation=correlation)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Quanto

        .. sectionauthor:: Patrick Granahan

        """

        # Explicit imports
        from numpy import arange, maximum

        # Define parameters
        T = self.T  # Time to expiry (years)
        S0 = self.ref.S  # Underlying Asset in Foreign Currency
        K = self.K  # Delivery Price in Foreign Currency
        rf_r = self.rf_r  # Interest rate for Domestic Currency
        frf_r = self.frf_r  # Interest rate for Foreign Currency
        dom_for_ex = getattr(self.px_spec, 'dom_for_ex')  # Exchange Rate for Domestic/Foreign Currency
        for_dom_spot = getattr(self.px_spec, 'for_dom_spot')  # Spot Rate for Foreign/Domestic Currency
        vol = self.ref.vol  # Volatility of Underlying
        vol_ex = getattr(self.px_spec, 'vol_exchange')  # Volatility of Domestic Currency
        dividend = self.ref.q  # Dividend Yield of Underlying
        correlation = getattr(self.px_spec, 'correlation')  # Correlation of Asset to Domestic Currency

        # Get additional provided parameters
        keep_hist = getattr(self.px_spec, 'keep_hist', False)
        n = getattr(self.px_spec, 'nsteps', 3)
        _ = self.LT_specs(n)

        S = S0 * _['d'] ** arange(n, -1, -1) * _['u'] ** arange(0, n + 1)  # terminal stock prices
        O = maximum(self.signCP * (S - K), 0)  # terminal option payouts
        # tree = ((S, O),)
        S_tree = (tuple([float(s) for s in S]),)  # use tuples of floats (instead of numpy.float)
        O_tree = (tuple([float(o) for o in O]),)
        # tree = ([float(s) for s in S], [float(o) for o in O],)

        for i in range(n, 0, -1):
            O = _['df_dt'] * ((1 - _['p']) * O[:i] + (_['p']) * O[1:])  # prior option prices (@time step=i-1)
            S = _['d'] * S[1:i + 1]  # prior stock prices (@time step=i-1)
            Payout = maximum(self.signCP * (S - K), 0)  # payout at time step i-1 (moving backward in time)
            O = maximum(O, Payout)
            # tree = tree + ((S, O),)
            S_tree = (tuple([float(s) for s in S]),) + S_tree
            O_tree = (tuple([float(o) for o in O]),) + O_tree
            # tree = tree + ([float(s) for s in S], [float(o) for o in O],)

        # Record price specs
        self.px_spec.add(px=float(Util.demote(O)), method='LT', sub_method='binomial tree; Hull Ch.13', LT_specs=_)
        if keep_hist:
            self.px_spec.add(ref_tree=S_tree, opt_tree=O_tree)

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
