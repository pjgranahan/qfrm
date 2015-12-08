import numpy as np

try: from qfrm.OptionValuation import *  # production:  if qfrm package is installed
except:   from OptionValuation import *  # development: if not installed and running from source


class PerpetualAmerican(OptionValuation):
    """ perpetual American option class.

    Inherits all methods and properties of ``OptionValuation`` class.
    """

    def calc_px(self, method='BS', nsteps=None, npaths=None, keep_hist=False):
        """ Wrapper function that calls appropriate valuation method.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments (``method``, ``nsteps``, ``npaths``, ``keep_hist``, ``rng_seed``, ...)
            are passed to the parent. See ``European.calc_px()`` for details.


        Returns
        ----------
        self : PerpetualAmerican
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object).



        Notes
        ---------
        In finance, a perpetual American option is a special type of American option which does not have a
        maturity date.

        Please note that this perpetual American option has no expiry data. Hence, the parameter ``T`` is not used.

        [1] Formula reference: Hull P.599 Perpetual American

        Examples
        ------------

        Use the Black-Scholes model to price a perpetual American option

        Verification of examples:
        `All the examples below can be verified by this online tools: <http://www.coggit.com/freetools>`_




        This examples below can be verified with
        `Coggit's Perpetual Option Pricing tool <http://www.coggit.com/freetools>`_

        >>> s = Stock(S0=50, vol=.3, q=0.01)
        >>> o = PerpetualAmerican(ref=s, right='call', T=1, K=50, rf_r=0.08, \
        desc='call @37.19, put @8.68. See www.coggit.com/freetools')
        >>> o.pxBS()
        37.190676834

        >>> o.calc_px(method="BS").px_spec  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px: 37.190676834...

        >>> o.calc_px(method='BS')  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PerpetualAmerican...px: 37.190676834...


        `Change the option to a put, can verified by this online tools: <http://www.coggit.com/freetools>`_

        >>> o.update(right='put').calc_px().px_spec.px # doctest: +ELLIPSIS
        8.676279289...

        >>> o.update(right='put').calc_px() # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PerpetualAmerican...px: 8.67627929...


        Another example with different dividend and risk free interest rate
        `This examples below can be verified by this online tools: <http://www.coggit.com/freetools>`_

        >>> s = Stock(S0=50, vol=.3, q=0.02)
        >>> o = PerpetualAmerican(ref=s, right='call', T=1, K=50, rf_r=0.05, \
        desc='call @27.47, put @13.43. See www.coggit.com/freetools')
        >>> o.pxBS()
        27.465595637

        `Change the option to a put, can be verified by this online tools: <http://www.coggit.com/freetools>`_

        >>> o.update(right='put').calc_px().px_spec.px# doctest: +ELLIPSIS
        13.427262534...

        >>> o.update(right='put').calc_px()# doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PerpetualAmerican...px: 13.427262535...




        # Example of option price development (BS method) with increasing maturities \
        (This would give a horizontal line because this perpetual American option does not have an expiry)

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([o.update(T=t).calc_px(method='BS').px_spec.px for t in expiries], expiries)
        >>> O.plot(grid=1, title='Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()

        :Authors:
            Tianyi Yao <ty13@rice.edu>

        """

        return super().calc_px(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)




    def _calc_BS(self):

        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Tianyi Yao <ty13@rice.edu>
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

        See ``calc_px()`` for complete documentation.

        :Authors:

        """

        return self

    def _calc_MC(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:

        """
        return self

    def _calc_FD(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:

        """

        return self



