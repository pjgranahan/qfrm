import numpy as np
from OptionValuation import *
import matplotlib.pyplot as plt

class Gap(OptionValuation):
    """ Gap option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def __init__(self, on = None, *args, **kwargs):

        """ Constructor for Gap option class

        Passes additional arguments to OptionValuation class

        Parameters
        ----------
        on : Numeric Vector
                A vector of number of steps to be used in binomial tree averaging, vector of positive intergers
        dir : string
                'in' or 'out'
        *args, **kwargs: varies
                arguments required by the constructor of OptionValuation class


        Returns
        -------
        self : Gap

        .. sectionauthor:: Thawda Aung

       """

        self.on = on

        super().__init__(*args,**kwargs)

    def calc_px(self, K2=None, method='BS', nsteps=None, npaths=None, keep_hist=False):
        """ Wrapper function that calls appropriate valuation method.

        User passes parameters to calc_px, which saves them to local PriceSpec object
        and calls specific pricing function (_calc_BS,...).
        This makes significantly less docstrings to write, since user is not interfacing pricing functions,
        but a wrapper function calc_px().

        Parameters
        ----------
        K2 : float
                The trigger price.
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
        self : Gap

        .. sectionauthor:: Yen-fei Chen

        Notes
        -----
        A gap option has a strike price, K1 , and a trigger price, K2 . The trigger price
        determines whether or not the gap option will have a nonzero payoff. The strike price
        determines the amount of the nonzero payoff. The strike price may be greater than or
        less than the trigger price.

        Examples
        --------

        >>> s = Stock(S0=500000, vol=.2)
        >>> o = Gap(ref=s, right='put', K=400000, T=1, rf_r=.05, desc='Hull p.601 Example 26.1')
        >>> o.calc_px(K2=350000, method='BS').px_spec.px
        1895.6889443965902

        >>> s = Stock(S0=50, vol=.2)
        >>> o = Gap(ref=s, right='call', K=57, T=1, rf_r=.09)
        >>> o.calc_px(K2=50, method='BS').px_spec.px
        2.266910325361735

        >>> o.calc_px(K2=50, method='BS').px_spec
        PriceSpec
        d1: 0.5499999999999999
        d2: 0.3499999999999999
        keep_hist: false
        method: BS
        px: 2.266910325361735
        px_call: 2.266910325361735
        px_put: 4.360987885821741
        sub_method: standard; Hull p.335
        <BLANKLINE>

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([o.update(T=t).calc_px(K2=50, method='BS').px_spec.px for t in expiries], expiries)
        >>> O.plot(grid=1, title='Price vs expiry (in years)')
        >>> import matplotlib.pyplot as plt
        >>> plt.show()
        <matplotlib.axes._subplots.AxesSubplot object at ...>

        LT Examples..

        This might take about 1-4 mins depending on your CPU

        >>> s = Stock(S0=500000, vol=.2,  q = 0)
        >>> o = Gap(ref=s, right='put', K=400000, T=1, rf_r=.05, on = (90000,)*23, desc = 'HULL p. 601 Exp 26.1')
        >>> o.calc_px(K2=350000, nsteps = 22, method='LT').px_spec.px
        1895.8012967929049


        >>> s = Stock(S0=50, vol=.2,  q = 0)
        >>> o = Gap(ref=s, right='call', K=57, T=1, rf_r=.09, on = (90000,)*23)
        >>> o.calc_px(K2=50, nsteps = 22, method='LT').px_spec.px
        2.2749024276146068

        >>> s = Stock(S0=50, vol=.2,  q = 0)
        >>> o = Gap(ref=s, right='put', K=57, T=1, rf_r=.09, on = (90000,)*23)
        >>> o.calc_px(K2=50, nsteps = 22, method='LT').px_spec.px
        4.3689799979566706

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> o = Series([o.update(T=t).calc_px(method='LT', nsteps=5).px_spec.px for t in expiries], expiries)
        >>> o.plot(grid=1, title='Price vs expiry (in years)')

        See Also
        --------
        [1] http://www.actuarialbookstore.com/samples/3MFE-BRE-12FSM%20Sample%20_4-12-12.pdf
        [2] https://www.ma.utexas.edu/users/mcudina/Lecture14_3_4_5.pdf



        """
        self.K2 = float(K2)
        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Gap

        .. sectionauthor:: Yen-fei Chen

        Note
        ----

        """
        from scipy.stats import norm
        from math import sqrt, exp, log

        _ = self
        d1 = (log(_.ref.S0 / _.K2) + (_.rf_r - _.ref.q + _.ref.vol ** 2 / 2.) * _.T)/(_.ref.vol * sqrt(_.T))
        d2 = d1 - _.ref.vol * sqrt(_.T)

        # if calc of both prices is cheap, do both and include them into Price object.
        # Price.px should always point to the price of interest to the user
        # Save values as basic data types (int, floats, str), instead of numpy.array
        px_call = float(_.ref.S0 * exp(-_.ref.q * _.T) * norm.cdf(d1) - _.K * exp(-_.rf_r * _.T) * norm.cdf(d2))
        px_put = float(- _.ref.S0 * exp(-_.ref.q * _.T) * norm.cdf(-d1) + _.K * exp(-_.rf_r * _.T) * norm.cdf(-d2))
        px = px_call if _.signCP == 1 else px_put if _.signCP == -1 else None

        self.px_spec.add(px=px, sub_method='standard; Hull p.335', px_call=px_call, px_put=px_put, d1=d1, d2=d2)

        return self

    def _calc_LT(self):
        """ Internal function for option valuation.
        A binomial tree pricer of Gap options that takes the average results for given step sizes in NSteps.
        Large step sizes should be used for optimal accuracy but may take a minute or so.

        Returns
        -------
        self: Gap
        :param
                on : Numeric Vector
                A vector of number of steps to be used in binomial tree averaging, vector of positive intergers

        .. sectionauthor:: Thawda Aung

        References :
        Hull, John C., Options, Futures and Other Derivatives, 9ed, 2014. Prentice Hall. ISBN 978-0-13-345631-8. http://www-2.rotman.utoronto.ca/~hull/ofod/index.html.
        Humphreys, Natalia. University of Dallas.

        LT Examples..

        This might take about 1-4 mins depending on your CPU

        >>> s = Stock(S0=500000, vol=.2,  q = 0)
        >>> o = Gap(ref=s, right='put', K=400000, T=1, rf_r=.05, on = (90000,)*23, desc = 'HULL p. 601 Exp 26.1')
        >>> print(o.calc_px(K2=350000, nsteps = 22, method='LT').px_spec.px)
        1895.80129679


        >>> s = Stock(S0=50, vol=.2,  q = 0)
        >>> o = Gap(ref=s, right='call', K=57, T=1, rf_r=.09, on = (90000,)*23)
        >>> print(o.calc_px(K2=50, nsteps = 22, method='LT').px_spec.px)
        2.27490242761

        >>> s = Stock(S0=50, vol=.2,  q = 0)
        >>> o = Gap(ref=s, right='put', K=57, T=1, rf_r=.09, on = (90000,)*23)
        >>> print(o.calc_px(K2=50, nsteps = 22, method='LT').px_spec.px)
        4.36897999796


        >>> s = Stock(S0=500000, vol=.2)
        >>> o = Gap(ref=s, right='put', K=400000, T=1, rf_r=.05)
        >>> print(o.calc_px(K2=350000, method='LT').px_spec.px)
        """
        from scipy.stats import norm
        from math import sqrt, exp , log
        import numpy as np

        n = getattr(self.px_spec ,'nsteps', 5)
        assert len(self.on) > n , 'nsteps must be less than the vector on'
        _ = self
        para = self.LT_specs(n)
        vol = _.ref.vol
        ttm = _.T
        on = self.on
        r = _.rf_r
        q = _.ref.q
        S0 = _.ref.S0
        sign = _.signCP
        K2 = _.K2
        K = _.K
        px = np.zeros(n)
        for i in range(n):
            u1 = exp(vol * sqrt(ttm/ on[i]))
            d1 = 1/u1
            p1 = (exp( (r-q) * (ttm / on[i])) - d1 ) / (u1 - d1)
            leng = on[i]
            S = [S0 * d1**(leng - j ) * u1**(j) for j in np.arange(0 , on[i]+1)]
            O = np.zeros(len(S))
            for m in range(len(S)):
                if(sign * (S[m] - K2) > 0 ):
                    O[m] = sign* (S[m] - K)
                else:
                    O[m] = 0
            csl = np.cumsum([np.log(i) for i in np.arange(1,on[i] + 1)])
            a = np.array(0)
            a = np.insert(csl , 0 , 0 )
            csl = a
            temp = [ csl[on[i]] - csl[j] - csl[ (leng - j) ] +
                     log(p1 ) * (j) + log( 1 - p1 ) * (leng - j) for j in np.arange(0 , on[i] +1)]
            px[i] = exp(r * -ttm) * sum([exp(temp[j]) * O[j]  for j in np.arange(0,len(temp))])
            # tmp = [ csl[on[i] + 1] - csl -1 for i  ]
        Px = np.mean(px)
        self.px_spec.add(px=Px, sub_method='binomial_tree; Hull p.335',
                         L_Tspecs=para, ref_tree = O, opt_tree = O )
        return self

    def _calc_MC(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Gap

        .. sectionauthor:: Oleg Melnikov

        Note
        ----

        """
        return self

    def _calc_FD(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Gap

        .. sectionauthor:: Oleg Melnikov

        Note
        ----

        """

        return self
