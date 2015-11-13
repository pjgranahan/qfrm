from qfrm import *

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

        Examples
        -------

        >>> s = Stock(S0=50, vol=.3)
        >>> o = American(ref=s, right='put', K=52, T=2, rf_r=.05, desc='7.42840, Hull p.288')

        >>> o.calc_px(method='LT', nsteps=2, keep_hist=True).px_spec.px
        7.42840190270483

        >>> o.px_spec.ref_tree
        ((50.000000000000014,),
         (37.0409110340859, 67.49294037880017),
         (27.440581804701324, 50.00000000000001, 91.10594001952546))

        >>> o.calc_px(method='LT', nsteps=2, keep_hist=False)
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
        _ = self

        try:
            _.T == None
        except TypeError:
            _.T = None

        assert _.ref.q > 0, 'q should be >0, q=0 will give an alpha1 of infinity'



        from math import sqrt

        w = _.rf_r - _.ref.q - ((_.ref.vol ** 2) / 2.)
        alpha1 = (-w + sqrt((w ** 2) + 2 * (_.ref.vol ** 2) * _.rf_r)) / (_.ref.vol ** 2)
        H1 = _.K * (alpha1 / (alpha1 - 1))
        alpha2 = (w + sqrt((w ** 2) + 2 * (_.ref.vol ** 2) * _.rf_r)) / (_.ref.vol ** 2)
        H2 = _.K * (alpha2 / (alpha2 + 1))

        #price the perpetual American option
        if _.signCP == 1:
            if _.ref.S0 < H1:
                return (_.K / (alpha1 - 1)) * ((((alpha1 - 1) / alpha1) * (_.ref.S0 / _.K)) ** alpha1)
            elif _.ref.S0 > H1:
                return _.ref.S0 - _.K
            else:
                print('The option cannot be priced')
        else:
            if _.ref.S0 > H2:
                return (_.K / (alpha2 + 1)) * ((((alpha2 + 1) / alpha2) * (_.ref.S0 / _.K)) ** ( -alpha2 ))
            elif _.ref.S0 < H2:
                return _.K - _.ref.S0
            else:
                print('The option cannot be priced ')
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


s = Stock(S0=50, vol=0.3, q=0.01)
o = PerpetualAmerican(ref=s, right='call', T=1, K=50, rf_r=0.08)
print(o.calc_px(method='BS'))  # save interim results to self.px_spec. Equivalent to repr(o)

