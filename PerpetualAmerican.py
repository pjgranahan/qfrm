import numpy as np

try: from qfrm.European import *  # production:  if qfrm package is installed
except:   from European import *  # development: if not installed and running from source


class PerpetualAmerican(European):
    """ Perpetual American option class.
    A perpetual American is an American option without determined maturity date.
    """

    def calc_px(self, **kwargs):
        """ Wrapper function that calls appropriate valuation method.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments (``method``, ``nsteps``, ``npaths``, ``keep_hist``, ``rng_seed``, ...)
            are passed to the parent. See ``European.calc_px()`` for details.
            Parameter ``T`` (years to expiry) is not used, since perpetual American does not expire.

        Returns
        ----------
        self : PerpetualAmerican
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object).


        Notes
        ---------

        *References:*

        - Examples can be verified with `Coggit Free tool: <http://www.coggit.com/freetools>`_
        - See p.599 in Options, Futures and Other Derivatives, `OFOD <http://www-2.rotman.utoronto.ca/~hull/ofod/index.html>`_, John C. Hull, 9ed, 2014, ISBN `0133456315 <http://amzn.com/0133456315>`_

        Examples
        ------------

        **BS**
        Notice the ignored expiry parameter ``T``.

        >>> s = Stock(S0=50, vol=.3, q=0.01)
        >>> o = PerpetualAmerican(ref=s, right='call', K=50, rf_r=0.08, desc='CoggIt.com: call @37.19, put @8.68')
        >>> o.pxBS()
        37.190676834

        >>> o.calc_px(method="BS").px_spec  # doctest: +ELLIPSIS
        PriceSpec...px: 37.190676834...

        >>> o.calc_px(method='BS')  # doctest: +ELLIPSIS
        PerpetualAmerican...px: 37.190676834...

        Change the option's right from call to put.

        >>> o.update(right='put').calc_px().px_spec.px  # doctest: +ELLIPSIS
        8.676279289...

        >>> o.update(right='put').calc_px()  # doctest: +ELLIPSIS
        PerpetualAmerican...px: 8.67627929...

        Another example with different dividend and risk free interest rate.

        >>> s = Stock(S0=50, vol=.3, q=0.02)
        >>> o = PerpetualAmerican(ref=s, right='call', K=50, rf_r=0.05, desc='CoggIt.com: call @27.47, put @13.43')
        >>> o.pxBS()
        27.465595637

        Change to a put.

        >>> o.update(right='put').pxBS()  # doctest: +ELLIPSIS
        13.427262535

        >>> o.update(right='put').calc_px()  # doctest: +ELLIPSIS
        PerpetualAmerican...px: 13.427262535...


        Next example shows sensitivity of perpetual American option's price to changes in volatility and strike.

        >>> from pandas import DataFrame
        >>> Ks = [30 + 4 * i for i in range(11)];   # a range of strikes
        >>> Ts = tuple(range(1, 101)) # a range of expiries
        >>> vols = [.05 + .025 * i for i in range(11)];   # a range of strikes
        >>> def px(vol, K):
        ...     s = Stock(S0=50, vol=vol, q=0.02)
        ...     return PerpetualAmerican(ref=s, right='call', K=K, rf_r=.05).pxBS()
        >>> px_grid = [[px(vol=vol, K=K) for vol in vols] for K in Ks]
        >>> DataFrame(px_grid, columns=Ks).plot(grid=1, title='BS strike vs vol at varying strikes, for ' + o.specs)  # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at 0x...>


        :Authors:
            Tianyi Yao <ty13@rice.edu>
        """
        # Check the validity of value of dividend
        assert self.ref.q > 0, 'q should be >0, q=0 will give an alpha1 of infinity'

        self.T = float('inf')
        self.save2px_spec(**kwargs)
        return getattr(self, '_calc_' + self.px_spec.method.upper())()

    def _calc_BS(self):
        """ Internal function for option valuation.        See ``calc_px()`` for complete documentation.

        :Authors:
            Tianyi Yao <ty13@rice.edu>
        """
        _ = self.ref;       S0, vol, q = _.S0, _.vol, _.q
        _ = self;           T, K, rf_r, net_r, sCP = _.T, _.K, _.rf_r, _.net_r, _.signCP

        # Compute parameters and barrier threshold (same notation as Hull P.599)
        w = rf_r - q - ((vol ** 2) / 2.)
        alpha1 = (-w + np.sqrt((w ** 2) + 2 * (vol ** 2) * rf_r)) / (vol ** 2)
        H1 = K * (alpha1 / (alpha1 - 1))
        alpha2 = (w + np.sqrt((w ** 2) + 2 * (vol ** 2) * rf_r)) / (vol ** 2)
        H2 = K * (alpha2 / (alpha2 + 1))

        # price the perpetual American call option
        if sCP == 1:
            if S0 < H1:
                out = (K / (alpha1 - 1)) * ((((alpha1 - 1) / alpha1) * (S0 / K)) ** alpha1)
            elif S0 > H1:
                out = S0 - K
            else:
                print('The option cannot be priced due to unknown threshold condition')
        # price the perpetual American put option
        else:
            if S0 > H2:
                out = (K / (alpha2 + 1)) * ((((alpha2 + 1) / alpha2) * (S0 / K)) ** ( -alpha2 ))
            elif S0 < H2:
                out = K - S0
            else:
                print('The option cannot be priced due to unknown threshold condition')

        self.px_spec.add(px=float(out))
        return self

    def _calc_LT(self):
        """ Internal function for option valuation.        See ``calc_px()`` for complete documentation.        """
        return self

    def _calc_MC(self):
        """ Internal function for option valuation.        See ``calc_px()`` for complete documentation.        """
        return self

    def _calc_FD(self):
        """ Internal function for option valuation.        See ``calc_px()`` for complete documentation.        """
        return self



