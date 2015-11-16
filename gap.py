from qfrm import *

class Gap(OptionValuation):
    """ Gap option class.

    Inherits all methods and properties of OptionValuation class.
    """
    def __init__(self, on = None, *args, **kwargs):

        """ Constructor for Barrier class

        Passes additional arguments to OptionValuation class

        Parameters
        ----------
        H : int
                The barrier used to price the barrier option
        knock : string
                'down' or 'up'
        dir : string
                'in' or 'out'
        *args, **kwargs: varies
                arguments required by the constructor of OptionValuation class


        Returns
        -------
        self : Barrier

        .. sectionauthor:: Thawda Aung

       """

        self.on = (498,499,500,501,502)

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
        nsteps : vector ( a Vector of number of steps to be used in binomial tree averaging) (has to be positive numbers)
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

        Examples
        -------

        >>> s = Stock(S0=500000, vol=.2)
        >>> o = Gap(ref=s, right='put', K=400000, T=1, rf_r=.05, desc='Hull p.601 Example 26.1')
        >>> print(o.calc_px(K2=350000, method='BS').px_spec.px)
        1895.6889443965902

        >>> s = Stock(S0=50, vol=.2)
        >>> o = Gap(ref=s, right='call', K=57, T=1, rf_r=.09)
        >>> print(o.calc_px(K2=50, method='BS').px_spec.px)
        2.266910325361735

        >>> s = Stock(S0=50, vol=.2)
        >>> o = Gap(ref=s, right='put', K=57, T=1, rf_r=.09)
        >>> print(o.calc_px(K2=50, method='BS').px_spec.px)
        4.360987885821741

        """
        self.K2 = float(K2)
        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Gap

        .. sectionauthor:: Thawda Aung


        >>>s = Stock(S0=500000, vol=.2)
        >>>o = Gap(ref=s, right='put', K=400000, T=1, rf_r=.05)
        >>>print(o.calc_px(K2=350000, method='LT').px_spec.px)
        """
        from scipy.stats import norm
        from math import sqrt, exp , log
        import numpy as np

        n = getattr(self.px_spec ,'nsteps', 10)
        assert len(self.on) > n , 'nsteps must be less than the vector on'
        _ = self
        para = self.LT_specs(n)
        on = (498,499,500,501,502)
        vol = _.ref.vol
        ttm = _.T
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
        d1 = (log(_.ref.S0 / _.K2) + (_.rf_r + _.ref.vol ** 2 / 2.) * _.T)/(_.ref.vol * sqrt(_.T))
        d2 = d1 - _.ref.vol * sqrt(_.T)

        # if calc of both prices is cheap, do both and include them into Price object.
        # Price.px should always point to the price of interest to the user
        # Save values as basic data types (int, floats, str), instead of numpy.array
        px_call = float(_.ref.S0 * exp(-_.ref.q * _.T) * norm.cdf(d1) - _.K * exp(-_.rf_r * _.T) * norm.cdf(d2))
        px_put = float(- _.ref.S0 * exp(-_.ref.q * _.T) * norm.cdf(-d1) + _.K * exp(-_.rf_r * _.T) * norm.cdf(-d2))
        px = px_call if _.signCP == 1 else px_put if _.signCP == -1 else None

        self.px_spec.add(px=px, sub_method='standard; Hull p.335', px_call=px_call, px_put=px_put, d1=d1, d2=d2)

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

s = Stock(S0=500000, vol=.2)
o = Gap(ref=s, right='put', K=400000, T=1, rf_r=.05)
print(o.calc_px(K2=350000, method='LT').px_spec.px)

# s = Stock(S0=500000, vol=.1) 
#
# o = Gap(ref=s, right='put', K=400000, T=1, rf_r=.05, desc='') 
# print(o.calc_px(K2=50, method='LT').px_spec.px)