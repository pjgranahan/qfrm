__author__ = 'scottmorgan'


from qfrm import *

class Asian(OptionValuation):
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

        .. sectionauthor:: Scott Morgan

        Notes
        -----

        Verification of First Example: http://investexcel.net/asian-options-excel/

        Examples
        -------

        >>> s = Stock(S0=30, vol=.3, q = .02)
        >>> o = Asian(ref=s, right='call', K=29, T=1., rf_r=.08, desc='Example from Internet')
        >>> o.calc_px()
        >>> print(o.px_spec)

            qfrm.PriceSpec
            keep_hist: false
            method: BSM
            px: 2.777361112923389
            sub_method: Geometric

        >>> s = Stock(S0=30, vol=.3, q = .02)
        >>> o = Asian(ref=s, right='put', K=29, T=1., rf_r=.08, desc='Example from Internet')
        >>> o.calc_px()
        >>> print(o.px_spec)

            qfrm.PriceSpec
            keep_hist: false
            method: BSM
            px: 1.2240784465431602
            sub_method: Geometric

        >>> s = Stock(S0=30, vol=.3, q = .02)
        >>> o = Asian(ref=s, right='put', K=30., T=1., rf_r=.08, desc='Example from Internet')
        >>> o.calc_px()
        >>> print(o.px_spec)

            qfrm.PriceSpec
            keep_hist: false
            method: BSM
            px: 1.6341047993229445
            sub_method: Geometric

        >>> s = Stock(S0=20, vol=.3, q = .00)
        >>> o = Asian(ref=s, right='put', K=21., T=1., rf_r=.08, desc='Example from Internet')
        >>> o.calc_px()
        >>> print(o.px_spec)

            qfrm.PriceSpec
            keep_hist: false
            method: BSM
            px: 1.489497403315955
            sub_method: Geometric

        >>> s = Stock(S0=20, vol=.3, q = .00)
        >>> o = Asian(ref=s, right='put', K=21., T=2., rf_r=.08, desc='Example from Internet')
        >>> o.calc_px()
        >>> print(o.px_spec)

            qfrm.PriceSpec
            keep_hist: false
            method: BSM
            px: 1.6162118076748948
            sub_method: Geometric




       """

        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Asian

        .. sectionauthor::

        """
        return self

    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Asian

        .. sectionauthor:: Scott Morgan

        Note
        ----

        Formulae: http://homepage.ntu.edu.tw/~jryanwang/course/Financial%20Computation%20or%20Financial%20Engineering%20(graduate%20level)/FE_Ch10%20Asian%20Options.pdf

        """

        # Verify input
        try:
            right   =   self.right.lower()
            S       =   float(self.ref.S0)
            K       =   float(self.K)
            T       =   float(self.T)
            vol     =   float(self.ref.vol)
            r       =   float(self.rf_r)
            q       =   float(self.ref.q)


        except:
            print('right must be String. S, K, T, vol, r, q must be floats or be able to be coerced to float')
            return False

        assert right in ['call','put'], 'right must be "call" or "put" '
        assert vol > 0, 'vol must be >=0'
        assert K > 0, 'K must be > 0'
        assert T > 0, 'T must be > 0'
        assert S >= 0, 'S must be >= 0'
        assert r >= 0, 'r must be >= 0'
        assert q >= 0, 'q must be >= 0'

        # Imports
        from math import exp
        from math import log
        from math import sqrt
        from scipy.stats import norm

        # Parameters for Value Calculation (see link in docstring)
        a = .5 * (r - q - (vol**2) / 6.)
        vola = vol / sqrt(3.)
        d1 = (log(S * exp(a * T) / K) + (vola**2) * .5 * T) / (vola * sqrt(T))
        d2 = d1 - vola * sqrt(T)

        # Calculate the value of the option using the BS Equation
        if right == 'call':
            px = S * exp((a - r) * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
            self.px_spec.add(px=float(px), method='BSM', sub_method='Geometric')

        else:
            px = K * exp(-r * T) * norm.cdf(-d2) - S * exp((a - r) * T) * norm.cdf(-d1)
            self.px_spec.add(px=float(px), method='BSM', sub_method='Geometric')
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
