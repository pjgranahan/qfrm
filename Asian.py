__author__ = 'scottmorgan'


def pxBS(right,S,K,T,vol,r,q=0):

    """
    pxBS computes the price of a GEOMETRIC Asian option given the parameters.


    :param S: Underlying price
    :type S: float

    :param K: Strike price
    :type K: float

    :param vol: volatility
    :type vol: float

    :param r: risk free rate, continuously compounded, annualized
    :type r: float

    :param q: dividend yield of the underlying, continuously compounded, annualized
    :type q: float

    :return: value of the  GEOMETRIC Asian option
    :rtype: float

    :Example:

    >>> pxBS('call',30.,29.,1.,.3,.08,.02) #returns 2.77736111292
    >>> pxBS('put',30.,29.,1.,.3,.08,.02) #returns 1.22407844654


    .. see also::

        Note: must copy and paste full link into browser

        Formulae: http://homepage.ntu.edu.tw/~jryanwang/course/Financial%20Computation%20or%20Financial%20Engineering%20(graduate%20level)/FE_Ch10%20Asian%20Options.pdf
        Verification of Examples: http://investexcel.net/asian-options-excel/
    """

    # Verify input
    try:
        right   =   right.lower()
        S       =   float(S)
        K       =   float(K)
        T       =   float(T)
        vol     =   float(vol)
        r       =   float(r)
        q       =   float(q)

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
        return S * exp((a - r) * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        return K * exp(-r * T) * norm.cdf(-d2) - S * exp((a - r) * T) * norm.cdf(-d1)

from qfrm import *

class Asian(OptionValuation):
    """ American option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def __init__(self,q=0.0,*args,**kwargs):


        super().__init__(*args,**kwargs)
        self.q = q




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
        self: American

        .. sectionauthor:: Oleg Melnikov

        Note
        ----

        """

        # Verify input
        try:
            right   =   self.right.lower()
            S       =   float(self.ref.S0)
            K       =   float(self.K)
            T       =   float(self.T)
            vol     =   float(self.ref.vol)
            r       =   float(self.rf_r)
            q       =   float(self.q)


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


s = Stock(S0=30, vol=.3)
o = Asian(q=.02,ref=s, right='call', K=29, T=1., rf_r=.08, desc='7.42840, Hull p.288')
print(o.right.__class__)
o.calc_px()
print(o.px_spec)

