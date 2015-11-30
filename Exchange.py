from scipy import stats
import numpy as np
import math
from OptionValuation import *

class Exchange(OptionValuation):
    """ Exchange option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, method='BS', nsteps=None, npaths=None, keep_hist=False, cor=0.1):
        """ Wrapper function that calls appropriate valuation method.

        User passes parameters to calc_px, which saves them to local PriceSpec object
        and calls specific pricing function (``_calc_BS``,...).
        This makes significantly less docstrings to write, since user is not interfacing pricing functions,
        but a wrapper function ``calc_px()``.

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
        cor: float, between 0 and 1
                Required. This specifies the correlation between the two assets of interest.

        Returns
        -------
        Exchange
            Returned object contains specifications and calculated price in embedded PriceSpec object.

        Examples
        --------
        BS Examples
        ---------------

        >>> s = Stock(S0=(100,100), vol=(0.15,0.20), q=(0.04,0.05))
        >>> o = Exchange(ref=s, right='call', K=40, T=1, rf_r=.1, \
        desc='px @4.578 page 4 http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L3exchange.pdf')
        >>> o.calc_px(method='BS',cor=0.75).px_spec.px # doctest: +ELLIPSIS
        4.578049200...

        >>> o.calc_px(method='BS', cor=0.75).px_spec # save interim results to self.px_spec. Equivalent to repr(o)
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px: 4.5780492...

        >>> (o.px_spec.px, o.px_spec.d1, o.px_spec.d2, o.px_spec.method)  # alternative attribute access
        (4.578049200203779, -0.009449111825230689, -0.14173667737846024, 'BS')

        >>> Exchange(clone=o).pxBS(cor=0.75)
        4.5780492

        Example of option price development (BS method) with increasing maturities

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([o.update(T=t).calc_px(method='BS', cor=0.75).px_spec.px for t in expiries], expiries)
        >>> O.plot(grid=1, title='Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()


        :Authors:
            Tianyi Yao <ty13@rice.edu>
        """
        self.cor = cor
        return super().calc_px(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist, cor=cor)

    def _calc_BS(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Tianyi Yao <ty13@rice.edu>
        """

        #extract parameters
        _ = self

        S0 = _.ref.S0
        S0_1 = S0[0] #spot price of asset 1
        S0_2 = S0[1] #spot price of asset 2
        vol = _.ref.vol
        vol_1 = vol[0] #volatility of asset 1
        vol_2 = vol[1] #volatility of asset 2
        q = _.ref.q
        q_1 = q[0] #annualized dividend yield of asset 1
        q_2 = q[1] #annualized dividend yield of asset 2
        cor = _.cor #correlation coefficient between the two assets
        T = _.T


        #compute necessary parameters
        vol_a = (vol_1 ** 2) + (vol_2 ** 2) -2 * cor * vol_1 * vol_2
        d1 = (np.log(S0_2 / S0_1) + ((q_1 - q_2 + (vol_a / 2)) * T)) / (np.sqrt(vol_a) * np.sqrt(T))
        d2 = d1 - np.sqrt(vol_a) * np.sqrt(T)

        px = (S0_2 * np.exp(-q_2 * T) * stats.norm.cdf(d1) - S0_1 * np.exp(-q_1 * T) * stats.norm.cdf(d2))

        self.px_spec.add(px=float(px), sub_method=None, d1=d1, d2=d2)

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
