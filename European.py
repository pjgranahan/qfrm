from scipy import stats
import warnings
import numpy as np
import math
from OptionValuation import *


class European(OptionValuation):
    """ European option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, method='BS', nsteps=None, npaths=None, keep_hist=False):
        """ Wrapper function that calls appropriate valuation method.

        User passes parameters to calc_px, which saves them to local ``PriceSpec`` object
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
                If ``True``, historical information (trees, simulations, grid) are saved in ``self.px_spec`` object.

        Returns
        -------
        European
            Returned object contains specifications and calculated price in embedded ``PriceSpec`` object.

        Examples
        --------

        >>> s = Stock(S0=42, vol=.20)
        >>> o = European(ref=s, right='put', K=40, T=.5, rf_r=.1, desc='call @0.81, put @4.76, Hull p.339')
        >>> o.calc_px(method='BS').px_spec   # save interim results to self.px_spec. Equivalent to repr(o)
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px: 0.808599373...

        >>> (o.px_spec.px, o.px_spec.d1, o.px_spec.d2, o.px_spec.method)  # alternative attribute access
        (0.8085993729000922, 0.7692626281060315, 0.627841271868722, 'BS')

        >>> o.update(right='call').pxBS()  # change option object to a put
        4.759422393

        >>> European(clone=o, K=41, desc='Ex. copy params; new strike.').pxLT()
        4.227003911

        >>> s = Stock(S0=810, vol=.2, q=.02)
        >>> o = European(ref=s, right='call', K=800, T=.5, rf_r=.05, desc='53.39, Hull p.291')
        >>> o.pxLT(nsteps=3)  # option price from a 3-step tree (that's 2 time intervals)
        59.867529938

        >>> o.pxLT(nsteps=3, keep_hist=True)  # option price from a 3-step tree (that's 2 time intervals)
        59.867529938

        >>> o.px_spec.ref_tree  # prints reference tree  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        ((810.0,), (746.491768087...878.911232579...), (687.962913360...810.0, 953.685129326...),
         (634.023026633...746.491768087...878.911232579...1034.8204598880159))

        >>> o.calc_px(method='LT', nsteps=2, keep_hist=True).px_spec.opt_tree
        ((53.39471637496134,), (5.062315192620067, 100.66143225703827), (0.0, 10.0, 189.3362341097378))

        >>> o.calc_px(method='LT', nsteps=2)   # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        European...px: 53.394716375...


        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """

        return super().calc_px(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)

    def _calc_BS(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """

        _ = self
        d1 = (math.log(_.ref.S0 / _.K) + (_.rf_r + _.ref.vol ** 2 / 2.) * _.T)/(_.ref.vol * math.sqrt(_.T))
        d2 = d1 - _.ref.vol * math.sqrt(_.T)

        # if calc of both prices is cheap, do both and include them into Price object.
        # Price.px should always point to the price of interest to the user
        # Save values as basic data types (int, floats, str), instead of np.array
        px_call = float(_.ref.S0 * math.exp(-_.ref.q * _.T) * stats.norm.cdf(d1)
                        - _.K * math.exp(-_.rf_r * _.T) * stats.norm.cdf(d2))
        px_put = float(- _.ref.S0 * math.exp(-_.ref.q * _.T) * stats.norm.cdf(-d1)
                       + _.K * math.exp(-_.rf_r * _.T) * stats.norm.cdf(-d2))
        px = px_call if _.signCP == 1 else px_put if _.signCP == -1 else None

        self.px_spec.add(px=px, sub_method='standard; Hull p.335', px_call=px_call, px_put=px_put, d1=d1, d2=d2)

        return self

    def _calc_LT(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """

        n = getattr(self.px_spec, 'nsteps', 3)
        _ = self.LT_specs(n)

        S = self.ref.S0 * _['d'] ** np.arange(n, -1, -1) * _['u'] ** np.arange(0, n + 1)
        O = np.maximum(self.signCP * (S - self.K), 0)          # terminal option payouts
        S_tree, O_tree = None, None

        if getattr(self.px_spec, 'keep_hist', False):
            S_tree = (tuple([float(s) for s in S]),)
            O_tree = (tuple([float(o) for o in O]),)

            for i in range(n, 0, -1):
                O = _['df_dt'] * ((1 - _['p']) * O[:i] + ( _['p']) * O[1:])  #prior option prices (@time step=i-1)
                S = _['d'] * S[1:i+1]                   # prior stock prices (@time step=i-1)

                S_tree = (tuple([float(s) for s in S]),) + S_tree
                O_tree = (tuple([float(o) for o in O]),) + O_tree

            out = O_tree[0][0]
        else:
            csl = np.insert(np.cumsum(np.log(np.arange(n) + 1)), 0, 0)         # logs avoid overflow & truncation
            tmp = csl[n] - csl - csl[::-1] + np.log(_['p']) * np.arange(n + 1) \
                  + np.log(1 - _['p']) * np.arange(n + 1)[::-1]
            out = (_['df_T'] * sum(np.exp(tmp) * tuple(O)))

        self.px_spec.add(px=float(out), sub_method='binomial tree; Hull Ch.135',
                         LT_specs=_, ref_tree=S_tree, opt_tree=O_tree)

        return self

    def _calc_MC(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """
        return self

    def _calc_FD(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """
        return self

