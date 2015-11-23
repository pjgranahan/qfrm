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
        ----------
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
        -------
        self : Chooser

        .. sectionauthor:: thawda, Yen-fei Chen

        Notes
        -----
        An option contract that allows the holder to decide whether it is a call or put prior to
        the expiration date. Chooser options usually have the same exercise price and expiration
        date regardless of what decision the holder ultimately makes.

        Examples
        -------
         Examples
        -------
        BS Examples

        >>> s = Stock(S0=50, vol=0.2, q=0.05)
        >>> o = Chooser(ref=s, right='put', K=50, T=1, rf_r=.1, desc= 'Exotic options paper page 9 Table 2 time 05')
        >>> print(o.calc_px(tau=6/12, method='BS').px_spec.px)
        6.58789632353
        EXOTIC OPTIONS: A CHOOSER OPTION AND ITS PRICING by Raimonda Martinkkute-Kauliene (Dec 2012)
        https://www.dropbox.com/s/r9lvi0uzdehwlm4/101-330-1-PB%20%284%29.pdf?dl=0

        >>> s = Stock(S0=50, vol=0.25, q=0.08)
        >>> o = Chooser(ref=s, right='put', K=50, T=.5, rf_r=.08)
        >>> print(o.calc_px(tau=3/12, method='BS').px_spec.px)
        5.77757834387

        >>> s = Stock(S0=50, vol=0.2, q=0.02)
        >>> o = Chooser(ref=s, right='call', K=50, T=9/12, rf_r=.06)
        >>> print(o.calc_px(tau=3/12, method='BS').px_spec.px)
        5.42052870833





        >>> o.calc_px(tau=3/12, method='LT', nsteps=5, keep_hist=True).px_spec.px
        7.109866570176281

        >>> o.px_spec.ref_tree
        ((50.00000000000001,),
        (46.19936548599171, 54.11329730833717),
        (42.687627426164845, 50.0, 58.5649789116098),
        (39.442826023824665, 46.19936548599171, 54.11329730833716, 63.38288231400874),
        (36.44467070550122, 42.687627426164845, 50.0, 58.56497891160979, 68.597135098346),
        (33.67441323880132, 39.442826023824665, 46.19936548599171, 54.11329730833716, 63.38288231400873, 74.24034332153934))

        >>> o.calc_px(tau=3/12, method='LT', nsteps=2, keep_hist=False)
        Chooser
        K: 50
        T: 0.5
        _right: put
        _signCP: -1
        frf_r: 0
        px_spec: qfrm.PriceSpec
        LT_specs:
            a: 1.0
            d: 0.8824969025845953
            df_T: 0.9607894391523232
            df_dt: 0.9801986733067553
            dt: 0.25
            p: 0.4687906266262439
            u: 1.1331484530668263
        keep_hist: false
        method: LT
        nsteps: 2
        px: 5.9971272680133465
        sub_method: binomial tree; Hull Ch.135
        ref: qfrm.Stock
        S0: 50
        curr: null
        desc: null
        q: 0.08
        tkr: null
        vol: 0.25
        rf_r: 0.08
        seed0: null
        tau: 0.25

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> o = Series([o.update(T=t).calc_px(method='LT', nsteps=5).px_spec.px for t in expiries], expiries)
        >>> o.plot(grid=1, title='Price vs expiry (in years)')

        See Also
        --------
        Hull, John C.,Options, Futures and Other Derivatives, 9ed, 2014. Prentice Hall. ISBN 978-0-13-345631-8.
        http://www-2.rotman.utoronto.ca/~hull/ofod/index.html

        Huang Espen G., Option Pricing Formulas, 2ed.
        http://down.cenet.org.cn/upfile/10/20083212958160.pdf

        Wee, Lim Tiong, MFE5010 Exotic Options,Notes for Lecture 4 Chooser option.
        http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L4chooser.pdf

        Humphreys, Natalia A., ACTS 4302 Principles of Actuarial Models: Financial Economics.
        Lesson 14: All-or-nothing, Gap, Exchange and Chooser Options.


        """
        self.tau = float(tau)
        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Chooser BS method
        Please see the examples above
        .. section author:: Thawda Aung, Yen-fei Chen

        References :
        Hull, John C.,Options, Futures and Other Derivatives, 9ed, 2014. Prentice Hall. ISBN 978-0-13-345631-8.
        http://www-2.rotman.utoronto.ca/~hull/ofod/index.html

        Huang Espen G., Option Pricing Formulas, 2ed. http://down.cenet.org.cn/upfile/10/20083212958160.pdf

        Wee, Lim Tiong, MFE5010 Exotic Options,Notes for Lecture 4 Chooser option.
        http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L4chooser.pdf

        Humphreys, Natalia A., ACTS 4302 Principles of Actuarial Models: Financial Economics. Lesson 14: All-or-nothing,
         Gap, Exchange and Chooser Options.

        """
        from scipy.stats import norm
        from math import sqrt, exp, log

        _ = self


        d2 = (log(_.ref.S0/_.K) + ((_.rf_r - _.ref.q  - _.ref.vol**2/2)*_.T) ) / ( _.ref.vol * sqrt(_.T))
        d1 =  d2 + _.ref.vol * sqrt(_.T)

        d2n = (log(_.ref.S0/_.K) + (_.rf_r - _.ref.q) * _.T - _.ref.vol**2 * _.tau /2) / ( _.ref.vol * sqrt(_.tau))
        d1n = d2n + _.ref.vol * sqrt(_.tau)

        px = _.ref.S0 * exp(-_.ref.q * _.T) * norm.cdf(d1) - _.K* exp(-_.rf_r * _.T ) * norm.cdf(d2) + _.K* exp(-_.rf_r * _.T ) * norm.cdf(-d2n)  - _.ref.S0* exp(-_.ref.q * _.T) * norm.cdf(-d1n)
        self.px_spec.add(px=px, d1=d1, d2=d2)

        return self

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Chooser

        .. sectionauthor:: Yen-fei Chen

        .. note::
        Implementing Binomial Trees:   http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1341181

        """
        from numpy import cumsum, log, arange, insert, exp, sqrt, sum, maximum

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

        Returns
        -------
        self: Chooser

        .. sectionauthor::

        Notes
        -----


        """
        return self

    def _calc_FD(self, nsteps=3, npaths=4, keep_hist=False):
        """ Internal function for option valuation.

        Returns
        -------
        self: Chooser

        .. sectionauthor::

        """
        return self
