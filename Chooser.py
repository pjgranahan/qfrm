import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from OptionValuation import *


class Chooser(OptionValuation):
    """ Chooser option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, tau=None, method='BS', nsteps=None, npaths=None, keep_hist=False):
        """ Wrapper function that calls appropriate valuation method.

        User passes parameters to calc_px, which saves them to local PriceSpec object
        and calls specific pricing function (_calc_BS,...).
        This makes significantly less docstrings to write, since user is not interfacing pricing functions,
        but a wrapper function calc_px().

        Parameters
        ------------------------------------------------
        tau : float
                Time to choose whether this option is a call or put.
        method : str
                Required. Indicates a valuation method to be used: 'BS', 'LT', 'MC', 'FD'
        nsteps : int
                LT, MC, FD methods require number of times steps
        npaths : int
                MC, FD methods require number of simulation paths
        keep_hist : bool
                If True, historical information (trees, simulations, grid) are saved in self.px_spec object.

        Returns
        -------------------------------------
        self : Chooser

        Notes
        --------------------------------------
        An option contract that allows the holder to decide whether it is a call or put prior to
        the expiration date. Chooser options usually have the same exercise price and expiration
        date regardless of what decision the holder ultimately makes.

        Examples
        --------------------------------------

        BS Examples
        --------------------------------------
        EXOTIC OPTIONS: A CHOOSER OPTION AND ITS PRICING by Raimonda Martinkkute-Kauliene (Dec 2012)
        https://www.dropbox.com/s/r9lvi0uzdehwlm4/101-330-1-PB%20%284%29.pdf?dl=0
        >>> s = Stock(S0=50, vol=0.2, q=0.05)
        >>> o = Chooser(ref=s, right='put', K=50, T=1, rf_r=.1, desc= 'Exotic options paper page 297 Table 2 time 0.5')
        >>> o.pxBS(tau=6/12)
        6.5878963235321955

        >>> s = Stock(S0=50, vol=0.2, q=0.05)
        >>> o = Chooser(ref=s, right='put', K=50, T=1, rf_r=.1, desc= 'Exotic options paper page 297 Table 2 time 1.00')
        >>> o.pxBS(tau=12/12)
        7.6213022738289808

        >>> s = Stock(S0=50, vol=0.25, q=0.08)
        >>> o = Chooser(ref=s, right='put', K=50, T=.5, rf_r=.08)
        >>> o.pxBS(tau=3/12)
        5.7775783438734258

        LT Examples
        ---------------------------------------------
        >>> s = Stock(S0=50, vol=0.2, q=0.05)
        >>> o = Chooser(ref=s, right='put', K=50, T=1, rf_r=.1, desc= 'Exotic options paper page 297 Table 2 time 0.5')
        >>> o.pxLT(tau=3/12, nsteps=2)
        6.755605274510829

        >>> o.calc_px(tau=3/12, method='LT', nsteps=2, keep_hist=True).px_spec.ref_tree
        ((50.0,), (43.40617226972924, 57.595495508445445), (37.68191582218824, 49.99999999999999, 66.3448220572672))

        >>> o.calc_px(tau=3/12, method='LT', nsteps=2, keep_hist=True).px_spec
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px: 6.755605275...

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> o = Series([o.update(T=t).pxLT(tau=3/12, nsteps=2) for t in expiries], expiries)
        >>> o.plot(grid=1, title='Price vs expiry (in years)')# doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()

        See Also
        -----------------------------------------------
        Hull, John C.,Options, Futures and Other Derivatives, 9ed, 2014. Prentice Hall. ISBN 978-0-13-345631-8.
        http://www-2.rotman.utoronto.ca/~hull/ofod/index.html

        Huang Espen G., Option Pricing Formulas, 2ed.
        http://down.cenet.org.cn/upfile/10/20083212958160.pdf

        Wee, Lim Tiong, MFE5010 Exotic Options,Notes for Lecture 4 Chooser option.
        http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L4chooser.pdf

        Humphreys, Natalia A., ACTS 4302 Principles of Actuarial Models: Financial Economics.
        Lesson 14: All-or-nothing, Gap, Exchange and Chooser Options.

        Implementing Binomial Trees:   http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1341181

        :Authors:
            Thawda Aung
            Yen-fei Chen <yensfly@gmail.com>

        """
        self.tau = float(tau)
        return super().calc_px(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)

    def _calc_BS(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Thawda Aung
        """
        from scipy.stats import norm
        from math import sqrt, exp, log

        _ = self

        d2 = (log(_.ref.S0/_.K) + ((_.rf_r - _.ref.q  - _.ref.vol**2/2)*_.T) ) / ( _.ref.vol * sqrt(_.T))
        d1 =  d2 + _.ref.vol * sqrt(_.T)

        d2n = (log(_.ref.S0/_.K) + (_.rf_r - _.ref.q) * _.T - _.ref.vol**2 * _.tau /2) / ( _.ref.vol * sqrt(_.tau))
        d1n = d2n + _.ref.vol * sqrt(_.tau)

        px = _.ref.S0 * exp(-_.ref.q * _.T) * norm.cdf(d1) - _.K* exp(-_.rf_r * _.T ) * norm.cdf(d2) +\
             _.K* exp(-_.rf_r * _.T ) * norm.cdf(-d2n)  - _.ref.S0* exp(-_.ref.q * _.T) * norm.cdf(-d1n)
        self.px_spec.add(px=px, d1=d1, d2=d2)

        return self

    def _calc_LT(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Yen-fei Chen <yensfly@gmail.com>
        """
        from numpy import cumsum, log, arange, insert, exp, sum, maximum

        n = getattr(self.px_spec, 'nsteps', 3)
        _ = self.LT_specs(n)

        S = self.ref.S0 * _['d'] ** arange(n, -1, -1) * _['u'] ** arange(0, n + 1)
        O = maximum(maximum((S - self.K), 0), maximum(-1*(S - self.K), 0))
        S_tree, O_tree = None, None

        if getattr(self.px_spec, 'keep_hist', False):
            S_tree = (tuple([float(s) for s in S]),)
            O_tree = (tuple([float(o) for o in O]),)

            for i in range(n, 0, -1):
                O = _['df_dt'] * ((1 - _['p']) * O[:i] + (_['p']) * O[1:])  # prior option prices (@time step=i-1)
                S = _['d'] * S[1:i + 1]  # prior stock prices (@time step=i-1)

                S_tree = (tuple([float(s) for s in S]),) + S_tree
                O_tree = (tuple([float(o) for o in O]),) + O_tree

            out = O_tree[0][0]
        else:
            csl = insert(cumsum(log(arange(n) + 1)), 0, 0)  # logs avoid overflow & truncation
            tmp = csl[n] - csl - csl[::-1] + log(_['p']) * arange(n + 1) + log(1 - _['p']) * arange(n + 1)[::-1]
            out = (_['df_T'] * sum(exp(tmp) * tuple(O)))

        self.px_spec.add(px=float(out), sub_method='binomial tree; Hull Ch.135',
                         LT_specs=_, ref_tree=S_tree, opt_tree=O_tree)

        return self

    def _calc_MC(self, nsteps=3, npaths=4, keep_hist=False):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """
        return self

    def _calc_FD(self, nsteps=3, npaths=4, keep_hist=False):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """
        return self
