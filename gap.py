from qfrm import *

class Gap(OptionValuation):
    """ Gap option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, K2=None, method='BS', nsteps=None, npaths=None, keep_hist=False):
        """ Wrapper function that calls appropriate valuation method.

        User passes parameters to calc_px, which saves them to local PriceSpec object
        and calls specific pricing function (_calc_BS,...).
        This makes significantly less docstrings to write, since user is not interfacing pricing functions,
        but a wrapper function calc_px().

        Parameters
        ----------
        K2 : float
                The trigger price.
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
        self : Gap

        .. sectionauthor:: Yen-fei Chen

        Notes
        -----

        Examples
        -------

        >>> s = Stock(S0=500000, vol=.2)
        >>> o = Gap(ref=s, right='put', K=400000, T=1, rf_r=.05, desc='Hull p.601 Example 26.1')
        >>> print(o.calc_px(K2=350000, method='BS').px_spec.px)
        1895.6889443965902

        >>> s = Stock(S0=50, vol=.2)
        >>> o = Gap(ref=s, right='call', K=57, T=1, rf_r=.09)
        >>> print(o.calc_px(K2=50, method='BS').px_spec.px)
        2.266910325361735

        >>> s = Stock(S0=50, vol=.2)
        >>> o = Gap(ref=s, right='put', K=57, T=1, rf_r=.09)
        >>> print(o.calc_px(K2=50, method='BS').px_spec.px)
        4.360987885821741

        """
        self.K2 = float(K2)
        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Gap

        .. sectionauthor:: Oleg Melnikov

        """
        return self

    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Gap

        .. sectionauthor:: Yen-fei Chen

        Note
        ----

        """
        from scipy.stats import norm
        from math import sqrt, exp, log

        _ = self
        d1 = (log(_.ref.S0 / _.K2) + (_.rf_r - _.ref.q + _.ref.vol ** 2 / 2.) * _.T)/(_.ref.vol * sqrt(_.T))
        d2 = d1 - _.ref.vol * sqrt(_.T)

        # if calc of both prices is cheap, do both and include them into Price object.
        # Price.px should always point to the price of interest to the user
        # Save values as basic data types (int, floats, str), instead of numpy.array
        px_call = float(_.ref.S0 * exp(-_.ref.q * _.T) * norm.cdf(d1) - _.K * exp(-_.rf_r * _.T) * norm.cdf(d2))
        px_put = float(- _.ref.S0 * exp(-_.ref.q * _.T) * norm.cdf(-d1) + _.K * exp(-_.rf_r * _.T) * norm.cdf(-d2))
        px = px_call if _.signCP == 1 else px_put if _.signCP == -1 else None

        self.px_spec.add(px=px, sub_method='standard; Hull p.335', px_call=px_call, px_put=px_put, d1=d1, d2=d2)

        return self

    def _calc_MC(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Gap

        .. sectionauthor:: Oleg Melnikov

        Note
        ----

        """
        return self

    def _calc_FD(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Gap

        .. sectionauthor:: Oleg Melnikov

        Note
        ----

        """

        return self
