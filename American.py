from OptionValuation import *
from European import *

class American(OptionValuation):
    """ American option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, method='BS', nsteps=None, npaths=None, keep_hist=False):
        """ Wrapper function that calls appropriate valuation method.

        User passes parameters to calc_px, which saves them to local PriceSpec object
        and calls specific pricing function (_calc_BS,...).
        This makes significantly less docstrings to write, since user is not interfacing pricing functions,
        but a wrapper function calc_px().

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
        self : American

        .. sectionauthor:: Oleg Melnikov

        Notes
        -----

        Examples
        -------

        >>> s = Stock(S0=50, vol=.3)
        >>> o = American(ref=s, right='put', K=52, T=2, rf_r=.05, desc='7.42840, Hull p.288')
        >>> o.calc_px(method='LT', nsteps=2, keep_hist=True).px_spec.px
        7.42840190270483

        >>> o.px_spec.ref_tree
        ((50.000000000000014,), (37.0409110340859, 67.49294037880017), (27.440581804701324, 50.00000000000001, 91.10594001952546))

        >>> o.calc_px(method='LT', nsteps=2, keep_hist=False)
        American
        K: 52
        T: 2
        _right: put
        _signcp: -1
        desc: 7.42840, Hull p.288
        frf_r: 0
        px_spec: OptionValuation.PriceSpec
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
        ref: OptionValuation.Stock
          S0: 50
          curr: null
          desc: null
          q: 0
          tkr: null
          vol: 0.3
        rf_r: 0.05
        seed0: null
        <BLANKLINE>
        """
        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: American

        .. sectionauthor:: Oleg Melnikov

        """
        from numpy import arange, maximum, log, exp, sqrt

        keep_hist = getattr(self.px_spec, 'keep_hist', False)
        n = getattr(self.px_spec, 'nsteps', 3)
        _ = self.LT_specs(n)

        S = self.ref.S0 * _['d'] ** arange(n, -1, -1) * _['u'] ** arange(0, n + 1)  # terminal stock prices
        O = maximum(self.signCP * (S - self.K), 0)          # terminal option payouts
        # tree = ((S, O),)
        S_tree = (tuple([float(s) for s in S]),)  # use tuples of floats (instead of numpy.float)
        O_tree = (tuple([float(o) for o in O]),)
        # tree = ([float(s) for s in S], [float(o) for o in O],)

        for i in range(n, 0, -1):
            O = _['df_dt'] * ((1 - _['p']) * O[:i] + ( _['p']) * O[1:])  #prior option prices (@time step=i-1)
            S = _['d'] * S[1:i+1]                   # prior stock prices (@time step=i-1)
            Payout = maximum(self.signCP * (S - self.K), 0)   # payout at time step i-1 (moving backward in time)
            O = maximum(O, Payout)
            # tree = tree + ((S, O),)
            S_tree = (tuple([float(s) for s in S]),) + S_tree
            O_tree = (tuple([float(o) for o in O]),) + O_tree
            # tree = tree + ([float(s) for s in S], [float(o) for o in O],)

        self.px_spec.add(px=float(Util.demote(O)), method='LT', sub_method='binomial tree; Hull Ch.13',
                        LT_specs=_, ref_tree = S_tree if keep_hist else None, opt_tree = O_tree if keep_hist else None)

        # self.px_spec = PriceSpec(px=float(Util.demote(O)), method='LT', sub_method='binomial tree; Hull Ch.13',
        #                 LT_specs=_, ref_tree = S_tree if save_tree else None, opt_tree = O_tree if save_tree else None)
        return self

    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: American

        .. sectionauthor:: Andrew Weatherly

        Note
        ----

        Formulae:
        http://aeconf.com/articles/may2007/aef080111.pdf (put)
        https://en.wikipedia.org/wiki/Black%27s_approximation (dividend call)
        http://www.bus.lsu.edu/academics/finance/faculty/dchance/Instructional/TN98-01.pdf (non-dividend call)

        Verifiable Example from Hull & White 2001
        http://efinance.org.cn/cn/FEshuo/230301%20%20%20%20%20The%20Use%20of%20the%20Control%20Variate%20Technique%20in%20Option%20Pricing,%20pp.%20237-251.pdf
        Scroll to page 246 in the pdf and and look at the very bottom right number b/c we use control variate for n = 100

        Examples
        -------

        >>> s = Stock(S0=30, vol=.3)
        >>> o = American(ref=s, right='call', K=30, T=1., rf_r=.08, desc='Example from Internet')
        >>> o.calc_px(method='BS')
        American
        K: 30
        T: 1.0
        _right: call
        _signcp: 1
        desc: Example from Internet
        frf_r: 0
        px_spec: OptionValuation.PriceSpec
          keep_hist: false
          method: European BSM
          px: 4.71339376436789
        ref: OptionValuation.Stock
          S0: 30
          curr: null
          desc: null
          q: 0
          tkr: null
          vol: 0.3
        rf_r: 0.08
        seed0: null
        <BLANKLINE>
        >>> print(o.px_spec)
        OptionValuation.PriceSpec
        keep_hist: false
        method: European BSM
        px: 4.71339376436789
        <BLANKLINE>
        >>> t = Stock(S0=40, vol=.2)
        >>> z = American(ref=t, right='put', K=35, T=.5833, rf_r=.0488)
        >>> z.calc_px(method='BS')
        American
        K: 35
        T: 0.5833
        _right: put
        _signcp: -1
        frf_r: 0
        px_spec: OptionValuation.PriceSpec
          keep_hist: false
          method: BSM
          px: 0.43262705935537815
          sub_method: Control Variate
        ref: OptionValuation.Stock
          S0: 40
          curr: null
          desc: null
          q: 0
          tkr: null
          vol: 0.2
        rf_r: 0.0488
        seed0: null
        <BLANKLINE>
        >>> print(z.px_spec)
        OptionValuation.PriceSpec
        keep_hist: false
        method: BSM
        px: 0.43262705935537815
        sub_method: Control Variate
        <BLANKLINE>
        """

        from math import exp
        from numpy import linspace
        if self.right == 'call' and self.ref.q != 0:
            #Black's approximations outlined on pg. 346
            #Dividend paying stocks assume semi-annual payments
            if self.T > .5:
                dividend_val1 = sum([self.ref.q * self.ref.S0 * exp(-self.rf_r * i) for i in linspace(.5, self.T - .5,
                                    self.T * 2 - .5)])
                dividend_val2 = sum([self.ref.q * self.ref.S0 * exp(-self.rf_r * i) for i in linspace(.5, self.T - 1,
                                    self.T * 2 - 1)])
            else:
                dividend_val1 = 0
                dividend_val2 = 0
            first_val = European(ref=Stock(S0=self.ref.S0 - dividend_val1, vol=self.ref.vol, q=self.ref.q), right=self.right,
                                 K=self.K, rf_r=self.rf_r, T=self.T).calc_px(method='BS').px_spec.px
            second_val = European(ref=Stock(S0=self.ref.S0 - dividend_val2, vol=self.ref.vol, q=self.ref.q),
                                  right=self.right, K=self.K, rf_r=self.rf_r, T=self.T - .5).calc_px(method='BS').px_spec.px
            self.px_spec.add(px=float(max([first_val, second_val])), method='BSM', sub_method='Black\'s Approximation')
        elif self.right == 'call':
            #American call is worth the same as European call if there are no dividends. This is by definition.
            #Check first line of the http://www.bus.lsu.edu/academics/finance/faculty/dchance/Instructional/TN98-01.pdf
            #paper as evidence
            self.px_spec.add(px=float(European(ref=Stock(S0=self.ref.S0, vol=self.ref.vol), right=self.right, K=self.K,
                                               rf_r=self.rf_r, T=self.T).calc_px(method='BS').px_spec.px),
                             method='European BSM')
        elif self.ref.q != 0:
            # I wasn't able to find a good approximation for American Put BSM w/ dividends so I'm using 200 and 201
            # time step LT and taking the average. This is effectively the Antithetic Variable technique found on pg. 476 due
            # to the oscillating nature of binomial tree
            f_a = (American(ref=Stock(S0=self.ref.S0, vol=self.ref.vol, q=self.ref.q), right=self.right,
                            K=self.K, rf_r=self.rf_r, T=self.T).calc_px(method='LT', nsteps=200).px_spec.px
                   + American(ref=Stock(S0=self.ref.S0, vol=self.ref.vol, q=self.ref.q), right=self.right, K=self.K,
                              rf_r=self.rf_r, T=self.T).calc_px(method='LT', nsteps=201).px_spec.px) / 2
            self.px_spec.add(px=float(f_a), method='BSM', sub_method='Antithetic Variable')
        else:
            #Control Variate technique outlined on pg.463
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

        Returns
        -------
        self: American

        .. sectionauthor:: Oleg Melnikov

        Note
        ----

        """
        return self

    def _calc_FD(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: American

        .. sectionauthor:: Oleg Melnikov

        Note
        ----

        """

        return self

if __name__ == "__main__":
    import doctest
    doctest.testmod()

"""
s = Stock(S0=50, vol=.3, q=.02)
o = American(ref=s, right='put', K=50, T=1., rf_r=.1, desc='Example from Internet')
o.calc_px(method='BS')
print(o.px_spec)

t = Stock(S0=40, vol=.2)
z = American(ref=t, right='put', K=35, T=.5833, rf_r=.0488)
print(z.calc_px(method='BS'))
print(z.px_spec)
"""
