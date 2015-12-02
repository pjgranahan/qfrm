import numpy as np
import matplotlib.pyplot as plt

try: from qfrm.OptionValuation import *  # production:  if qfrm package is installed
except:   from OptionValuation import *  # development: if not installed and running from source

try: from qfrm.European import *  # production:  if qfrm package is installed
except:   from European import *  # development: if not installed and running from source


class American(OptionValuation):
    """ American option class.

    Inherits all methods and properties of ``OptionValuation`` class.
    """

    def calc_px(self, method='BS', nsteps=None, npaths=None, keep_hist=False):
        """ Wrapper function that calls appropriate valuation method.

        All parameters of ``calc_px`` are saved to local ``px_spec`` variable of class ``PriceSpec`` before
        specific pricing method (``_calc_BS()``,...) is called.
        An alternative to price calculation method ``.calc_px(method='BS',...).px_spec.px``
        is calculating price via a shorter method wrapper ``.pxBS(...)``.
        The same works for all methods (BS, LT, MC, FD).


        Parameters
        ----------
        method : str
                Required. Indicates a valuation method to be used:
                ``BS``: Black-Scholes Merton calculation
                ``LT``: Lattice tree (such as binary tree)
                ``MC``: Monte Carlo simulation methods
                ``FD``: finite differencing methods
        nsteps : int
                LT, MC, FD methods require number of times steps
        npaths : int
                MC, FD methods require number of simulation paths
        keep_hist : bool
                If ``True``, historical information (trees, simulations, grid) are saved in ``self.px_spec`` object.

        Returns
        -------
        self : American
            Returned object contains specifications and calculated price in embedded ``PriceSpec`` object.

        Examples
        --------
        >>> s = Stock(S0=50, vol=.3)
        >>> American(ref=s, right='put', K=52, T=2, rf_r=.05, desc='7.42840, See J.C.Hull p.288').pxLT(nsteps=2)
        7.428401903

        >>> o = American(ref=s, right='put', K=52, T=2, rf_r=.05, desc='7.42840, See J.C.Hull p.288')
        >>> o.calc_px(method='LT', nsteps=2, keep_hist=True).px_spec.px
        7.42840190270483

        >>> o.px_spec.ref_tree  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        ((50.000...), (37.040911034...67.49294037880017), (27.440581804...50.000...91.10594001952546))

        >>> o.calc_px(method='LT', nsteps=2, keep_hist=False)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        American...px: 7.428401903...

        >>> s = Stock(S0=30, vol=.3)
        >>> o = American(ref=s, right='call', K=30, T=1., rf_r=.08)
        >>> o.calc_px(method='BS')  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        American...px: 4.713393764...

        >>> print(o.px_spec)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px: 4.713393764...

        >>> t = Stock(S0=40, vol=.2)
        >>> z = American(ref=t, right='put', K=35, T=.5833, rf_r=.0488, desc='Example From Hull and White 2001')
        >>> z.calc_px(method='BS')   # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        American...px: 0.432627059...

        >>> print(z.px_spec)   # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px: 0.432627059...

        >>> p = Stock(S0=50, vol=.25, q=.02)
        >>> v = American(ref=p, right='call', K=40, T=2, rf_r=.05)
        >>> print(v.calc_px(method='BS'))  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        American...px: 11.337850838...

        >>> print(v.px_spec)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px: 11.337850838...

        :Authors:
            Oleg Melnikov <xisreal@gmail.com>, Andrew Weatherly
        """
        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_LT(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """
        # from numpy import arange, maximum, log, exp, sqrt

        keep_hist = getattr(self.px_spec, 'keep_hist', False)
        n = getattr(self.px_spec, 'nsteps', 3)
        _ = self.LT_specs(n)

        S = self.ref.S0 * _['d'] ** np.arange(n, -1, -1) * _['u'] ** np.arange(0, n + 1)  # terminal stock prices
        O = np.maximum(self.signCP * (S - self.K), 0)          # terminal option payouts
        # tree = ((S, O),)
        S_tree = (tuple([float(s) for s in S]),)  # use tuples of floats (instead of numpy.float)
        O_tree = (tuple([float(o) for o in O]),)
        # tree = ([float(s) for s in S], [float(o) for o in O],)

        for i in range(n, 0, -1):
            O = _['df_dt'] * ((1 - _['p']) * O[:i] + ( _['p']) * O[1:])  #prior option prices (@time step=i-1)
            S = _['d'] * S[1:i+1]                   # prior stock prices (@time step=i-1)
            Payout = np.maximum(self.signCP * (S - self.K), 0)   # payout at time step i-1 (moving backward in time)
            O = np.maximum(O, Payout)
            # tree = tree + ((S, O),)
            S_tree = (tuple([float(s) for s in S]),) + S_tree
            O_tree = (tuple([float(o) for o in O]),) + O_tree
            # tree = tree + ([float(s) for s in S], [float(o) for o in O],)

        self.px_spec.add(px=float(Util.demote(O)), method='LT', sub_method='binomial tree; Hull Ch.13',
                        LT_specs=_, ref_tree = S_tree if keep_hist else None, opt_tree = O_tree if keep_hist else None)

        return self

    def _calc_BS(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        The ``_calc_BS()`` function is called through ``calc_PX()`` and uses the Black Scholes Merton
        differential equation to price the American option. Due to the optimal stopping problem,
        this is technically impossible, so the methods below are
        approximations that have been developed by financial computation scientists.


        Notes
        -----
        Important that if you plan on giving a dividend paying stock that it is semi-annual percentage dividends.
        This is currently the only type of dividends that the BSM can accept.

        :Formulae:
        - `The Closed-form Solution for Pricing American Options <http://aeconf.com/articles/may2007/aef080111.pdf>`_
        - `Black's approximation <https://en.wikipedia.org/wiki/Black%27s_approximation>`_ (dividend call)
        -
            `Closed-Form American Call Option Pricing by Roll-Geske-Whaley, 2008
            <http://www.bus.lsu.edu/academics/finance/faculty/dchance/Instructional/TN98-01.pdf>`_

        *Verifiable Example:*
        See `The Use of Control Variate Technique in Option-Pricing by Hull & White 2001
        <https://www.researchgate.net/publication/46543317_The_Use_of_Control_Variate_Technique_in_Option-Pricing>`_.
        2nd example in list, on p.246; the very bottom right number b/c we use control variate for n = 100

        :Authors:
            Andrew Weatherly
        """
        #Verify Input
        assert self.right in ['call', 'put'], 'right must be "call" or "put" '
        assert self.ref.vol > 0, 'vol must be >=0'
        assert self.K > 0, 'K must be > 0'
        assert self.T > 0, 'T must be > 0'
        assert self.ref.S0 >= 0, 'S must be >= 0'
        assert self.rf_r >= 0, 'r must be >= 0'

        #Imports
        from math import exp
        from numpy import linspace

        if self.right == 'call' and self.ref.q != 0:
            # Black's approximations outlined on pg. 346
            # Dividend paying stocks assume semi-annual payments
            if self.T > .5:
                dividend_val1 = sum([self.ref.q * self.ref.S0 * exp(-self.rf_r * i) for i in linspace(.5, self.T - .5,
                                    self.T * 2 - .5)])
                dividend_val2 = sum([self.ref.q * self.ref.S0 * exp(-self.rf_r * i) for i in linspace(.5, self.T - 1,
                                    self.T * 2 - 1)])
            else:
                dividend_val1 = 0
                dividend_val2 = 0
            first_val = European(
                    ref=Stock(S0=self.ref.S0 - dividend_val1, vol=self.ref.vol, q=self.ref.q), right=self.right,
                    K=self.K, rf_r=self.rf_r, T=self.T).calc_px(method='BS').px_spec.px
            second_val = European(
                    ref=Stock(S0=self.ref.S0 - dividend_val2, vol=self.ref.vol, q=self.ref.q),
                    right=self.right, K=self.K, rf_r=self.rf_r, T=self.T - .5).calc_px(method='BS').px_spec.px
            self.px_spec.add(px=float(max([first_val, second_val])), method='BSM', sub_method='Black\'s Approximation')
        elif self.right == 'call':
            # American call is worth the same as European call if there are no dividends. This is by definition.
            # Check first line of http://www.bus.lsu.edu/academics/finance/faculty/dchance/Instructional/TN98-01.pdf
            # paper as evidence
            self.px_spec.add(px=float(European(ref=Stock(S0=self.ref.S0, vol=self.ref.vol), right=self.right, K=self.K,
                                               rf_r=self.rf_r, T=self.T).calc_px(method='BS').px_spec.px),
                             method='European BSM')
        elif self.ref.q != 0:
            # I wasn't able to find a good approximation for American Put BSM w/ dividends so I'm using 200 and 201
            # time step LT and taking the average. This is effectively
            # the Antithetic Variable technique found on pg. 476 due
            # to the oscillating nature of binomial tree
            f_a = (American(ref=Stock(S0=self.ref.S0, vol=self.ref.vol, q=self.ref.q), right=self.right,
                            K=self.K, rf_r=self.rf_r, T=self.T).calc_px(method='LT', nsteps=200).px_spec.px
                   + American(ref=Stock(S0=self.ref.S0, vol=self.ref.vol, q=self.ref.q), right=self.right, K=self.K,
                              rf_r=self.rf_r, T=self.T).calc_px(method='LT', nsteps=201).px_spec.px) / 2
            self.px_spec.add(px=float(f_a), method='BSM', sub_method='Antithetic Variable')
        else:
            # Control Variate technique outlined on pg.463
            f_a = American(ref=Stock(S0=self.ref.S0, vol=self.ref.vol), right=self.right,
                           K=self.K, rf_r=self.rf_r, T=self.T).calc_px(method='LT', nsteps=100).px_spec.px
            f_bsm = European(ref=Stock(S0=self.ref.S0, vol=self.ref.vol), right=self.right,
                             K=self.K, rf_r=self.rf_r, T=self.T).calc_px(method='BS').px_spec.px
            f_e = European(ref=Stock(S0=self.ref.S0, vol=self.ref.vol), right=self.right,
                           K=self.K, rf_r=self.rf_r, T=self.T).calc_px(method='LT', nsteps=100).px_spec.px
            self.px_spec.add(px=float(f_a + (f_bsm - f_e)), method='BSM', sub_method='Control Variate')
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

