__author__ = 'MengyanXie'


from qfrm import *

class Lookback(OptionValuation):
    """ Lookback option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def __init__(self,q=0.0,*args,**kwargs):


        super().__init__(*args,**kwargs)
        self.q = q




    def calc_px(self, method='BS', nsteps=None, npaths=None, keep_hist=False, Sfl = 50.0):
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
        Sfl : float
                Asset floating price.
                If call option, Sfl is minimum asset price achieved to date.(If the look back has
                just been originated, Smin = S0.)
                If put option, Sfl is maximum asset price achieved to date. (If the look back has just been originated,
                Smax = S0.)


        Returns
        -------
        self : Lookback

        .. sectionauthor:: Mengyan Xie

        Notes
        -----

        Verification of Example: http://investexcel.net/asian-options-excel/

        Examples

        >>> s = Stock(S0=50, vol=.4)
        >>> o = Lookback(q=.0,ref=s, right='call', K=50, T=0.25, rf_r=.1, desc='Example from Internet')
        >>> print(o.calc_px(method = 'BS', Sfl = 50.0).px_spec.px)

        8.037120139607019

        >>> print(o.calc_px(method = 'BS', Sfl = 50.0))

        Lookback
        K: 50
        T: 0.25
        _right: call
        _signCP: 1
        desc: Example from Internet
        frf_r: 0
        px_spec: qfrm.PriceSpec
        Sfl: 50.0
        keep_hist: false
        method: BS
        px: 8.037120139607019
        sub_method: Look back, Hull Ch.26
        q: 0.0
        ref: qfrm.Stock
        S0: 50
        curr: null
        desc: null
        q: 0
        tkr: null
        vol: 0.4
        rf_r: 0.1
        seed0: null

        >>> s = Stock(S0=50, vol=.4)
        >>> o = Lookback(q=.0,ref=s, right='call', K=50, T=0.25, rf_r=.1, desc='Example from Internet')
        >>> print(o.calc_px(method = 'BS', Sfl = 50.0).px_spec.px)

        Lookback
        K: 50
        T: 0.25
        _right: put
        _signCP: -1
        desc: Example from Internet
        frf_r: 0
        px_spec: qfrm.PriceSpec
        Sfl: 50.0
        keep_hist: false
        method: BS
        px: 7.79021925989035
        sub_method: Look back, Hull Ch.26
        q: 0.0
        ref: qfrm.Stock
        S0: 50
        curr: null
        desc: null
        q: 0
        tkr: null
        vol: 0.4
        rf_r: 0.1
        seed0: null

        >>> print(o.px_spec)
        qfrm.PriceSpec
        Sfl: 50.0
        keep_hist: false
        method: BS
        px: 7.79021925989035
        sub_method: Look back, Hull Ch.26

        -------

       """

        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist, Sfl = Sfl)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Look back

        .. sectionauthor::

        """
        return self

    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Look back

        .. sectionauthor:: Mengyan Xie

        Note
        ----
        Formular: https://en.wikipedia.org/wiki/Lookback_option
        """

        # Verify input
        try:
            right   =   self.right.lower()
            S       =   float(self.ref.S0)
            Sfl     =   float(self.px_spec.Sfl)
            T       =   float(self.T)
            vol     =   float(self.ref.vol)
            r       =   float(self.rf_r)
            q       =   float(self.q)
            signCP  =   self.signCP


        except:
            print('right must be String. S, Sfl, T, vol, r, q must be floats or be able to be coerced to float')
            return False

        assert right in ['call','put'], 'right must be "call" or "put" '
        assert S >= 0, 'S must be >= 0'
        assert Sfl > 0, 'Sfl must be > 0'
        assert T > 0, 'T must be > 0'
        assert vol > 0, 'vol must be >=0'
        assert r >= 0, 'r must be >= 0'
        assert q >= 0, 'q must be >= 0'

        # Imports
        from math import exp, log, sqrt
        from scipy.stats import norm

        # Parameters for Value Calculation (see link in docstring)


        S_new = S / Sfl if right == 'call' else Sfl / S

        a1 = (log(S_new) + (signCP * (r - q) + vol ** 2 / 2) * T) / (vol * sqrt(T))
        a2 = a1 - vol * sqrt(T)
        a3 = (log(S_new) + signCP * (-r + q + vol ** 2 / 2) * T) / (vol * sqrt(T))
        Y1 = signCP * (-2 * (r - q - vol ** 2 / 2) * log(S_new)) / (vol ** 2)

        c = S * exp(-q * T) * norm.cdf(a1) - S * exp(-q * T) * (vol ** 2) * norm.cdf(-a1) / (2 * (r - q)) - Sfl * exp(-r * T) * (norm.cdf(a2) - vol ** 2 * exp(Y1) * norm.cdf(-a3) / (2 * (r - q)))
        p = Sfl * exp(-r * T) * (norm.cdf(a1) - vol ** 2 * exp(Y1) * norm.cdf(-a3) / (2 * (r - q))) + S * exp(-q *T) * (vol ** 2) * norm.cdf(-a2) / (2 * (r - q)) - S * exp(-q * T) * norm.cdf(a2)


        # Calculate the value of the option using the BS Equation
        if right == 'call':
            self.px_spec.add(px=float(c), method='BS', sub_method='Look back, Hull Ch.26')

        else:
            self.px_spec.add(px=float(p), method='BS', sub_method='Look back, Hull Ch.26')
        return self

    def _calc_MC(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Look back

        .. sectionauthor::

        Note
        ----

        """
        return self

    def _calc_FD(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Look back

        .. sectionauthor::

        Note
        ----

        """

        return self


