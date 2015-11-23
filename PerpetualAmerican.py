from OptionValuation import *
import numpy as np

class PerpetualAmerican(OptionValuation):
    """ perpetual American option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, method='BS', nsteps=None, npaths=None, keep_hist=False):
        """ Wrapper function that calls appropriate valuation method.

        User passes parameters to calc_px, which saves them to local PriceSpec object
        and calls specific pricing function


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
        self : PerpetualAmerican

        .. sectionauthor:: Tianyi Yao

        Notes
        -----
        In finance, a perpetual American option is a special type of American option which deos not have a
        maturity date.

        Formula reference: Hull P.599 Perpetual American

        Examples
        -------

        Use the Black-Scholes model to price a perpetual American option

        Verification of examples:
        All the examples below can be verified by this online tools: http://www.coggit.com/freetools
        R package: PerpetualBS.R

        >>> s = Stock(S0=50, vol=.3, q=0.01)
        >>> o = PerpetualAmerican(ref=s, right='call', T=1, K=50, rf_r=0.08, desc='call @37.19 put @8.68 example from Internet')

        >>> print(o.calc_px(method='BS').px_spec.px) #print the option price with the above specification
        37.190676833752335

        >>> print(o.calc_px(method="BS").px_spec)  #display the specification of the perpetual American option
        PriceSpec
        keep_hist: false
        method: BS
        px: 37.190676833752335
        <BLANKLINE>

        >>> print(o.calc_px(method='BS'))  #display more properties of the option
        PerpetualAmerican.PerpetualAmerican
        K: 50
        T: 1
        _right: call
        _signCP: 1
        desc: call @37.19 put @8.68 example from Internet
        frf_r: 0
        px_spec: PriceSpec
          keep_hist: false
          method: BS
          px: 37.190676833752335
        ref: Stock
          S0: 50
          curr: -
          desc: -
          q: 0.01
          tkr: -
          vol: 0.3
        rf_r: 0.08
        seed0: -
        <BLANKLINE>

        Change the option to a put
        >>> print(o.update(right='put').calc_px().px_spec.px)
        8.67627928986901

        >>> print(o.update(right='put').calc_px()) #display full specification
        PerpetualAmerican.PerpetualAmerican
        K: 50
        T: 1
        _right: put
        _signCP: -1
        desc: call @37.19 put @8.68 example from Internet
        frf_r: 0
        px_spec: PriceSpec
          keep_hist: false
          method: BS
          px: 8.67627928986901
        ref: Stock
          S0: 50
          curr: -
          desc: -
          q: 0.01
          tkr: -
          vol: 0.3
        rf_r: 0.08
        seed0: -
        <BLANKLINE>

        Another example with different dividend and risk free interest rate
        >>> s = Stock(S0=50, vol=.3, q=0.02)
        >>> o = PerpetualAmerican(ref=s, right='call', T=1, K=50, rf_r=0.05, desc='call @27.47 put @13.43 example from Internet')
        >>> print(o.calc_px(method='BS').px_spec.px)
        27.465595636754223

        Change the option to a put
        >>> print(o.update(right='put').calc_px().px_spec.px)
        13.427262534976805

        >>> print(o.update(right='put').calc_px())
        PerpetualAmerican.PerpetualAmerican
        K: 50
        T: 1
        _right: put
        _signCP: -1
        desc: call @27.47 put @13.43 example from Internet
        frf_r: 0
        px_spec: PriceSpec
          keep_hist: false
          method: BS
          px: 13.427262534976805
        ref: Stock
          S0: 50
          curr: -
          desc: -
          q: 0.02
          tkr: -
          vol: 0.3
        rf_r: 0.05
        seed0: -
        <BLANKLINE>

        # Example of option price development (BS method) with increasing maturities (This would give a horizontal line\
        because this perpetual American option does not have an expiry)
        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([o.update(T=t).calc_px(method='BS').px_spec.px for t in expiries], expiries)
        >>> O.plot(grid=1, title='Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()

        """
        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        return getattr(self, '_calc_' + method.upper())()




    def _calc_BS(self):

        """ Internal function for option valuation.

        Returns
        -------
        self: PerpetualAmerican

        .. sectionauthor:: Tianyi Yao

        Note
        ----

        """

        #Get parameters
        _ = self

        #Make sure no time to maturity is specified as this option has no expiry
        try:
            _.T == None
        except TypeError:
            _.T = None

        #Check the validity of value of dividend
        assert _.ref.q > 0, 'q should be >0, q=0 will give an alpha1 of infinity'




        #Compute parameters and barrier threshold (same notation as Hull P.599)
        w = _.rf_r - _.ref.q - ((_.ref.vol ** 2) / 2.)
        alpha1 = (-w + np.sqrt((w ** 2) + 2 * (_.ref.vol ** 2) * _.rf_r)) / (_.ref.vol ** 2)
        H1 = _.K * (alpha1 / (alpha1 - 1))
        alpha2 = (w + np.sqrt((w ** 2) + 2 * (_.ref.vol ** 2) * _.rf_r)) / (_.ref.vol ** 2)
        H2 = _.K * (alpha2 / (alpha2 + 1))

        #price the perpetual American call option
        if _.signCP == 1:
            if _.ref.S0 < H1:
                out = (_.K / (alpha1 - 1)) * ((((alpha1 - 1) / alpha1) * (_.ref.S0 / _.K)) ** alpha1)
            elif _.ref.S0 > H1:
                out = _.ref.S0 - _.K
            else:
                print('The option cannot be priced due to unknown threshold condition')
        #price the perpetual American put option
        else:
            if _.ref.S0 > H2:
                out = (_.K / (alpha2 + 1)) * ((((alpha2 + 1) / alpha2) * (_.ref.S0 / _.K)) ** ( -alpha2 ))
            elif _.ref.S0 < H2:
                out = _.K - _.ref.S0
            else:
                print('The option cannot be priced due to unknown threshold condition')

        self.px_spec.add(px=float(out))



        return self

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: PerpetualAmerican

        .. sectionauthor::

        """

        return self

    def _calc_MC(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: PerpetualAmerican

        .. sectionauthor::

        Note
        ----

        """
        return self

    def _calc_FD(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: PerpetualAmerican

        .. sectionauthor::

        Note
        ----

        """

        return self



