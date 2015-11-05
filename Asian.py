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
