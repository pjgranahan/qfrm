def Gap_BS(right, S0, K1, K2, T, r, vol, q=0):

    """
    Price of a gap option estimated by Black-Scholes model.

    :param right: 'call'/'put' option right
    :type right: str
    :param S0: underlying price
    :type S0: float
    :param K1: the first strike price
    :type K1: float
    :param K2: the second strike price
    :type K2: float
    :param T: time to maturity, in years, positive number.
    :type T: float
    :param r: risk free rate, continuously compounded, annualized
    :type r: float
    :param q: dividend yield of the underlying, continuously compounded, annualized
    :type q: float
    :param vol: volatility
    :type vol: float

    :return: value of the gap option
    :rtype: float

    :Example:
    >>> print(Gap_BS('put', S0=500000, K1=400000, K2=350000, T=1, r=0.05, vol=0.2, q=0))
    1895.6889444

    .. see also::
        Hull p.601
    """
    from numpy import log, sqrt, exp
    from scipy.stats import norm

    assert right in ['call','put'], 'Option right must be "call" or "put" '
    assert S0 >= 0, 'S0 must be >= 0'
    assert K1 >= 0, 'K1 must be >= 0'
    assert T > 0, 'T must be > 0'
    assert vol > 0, 'vol must be >0'
    assert r >= 0, 'r must be >= 0'
    assert q >= 0, 'q must be >= 0'

    d1 = (log(S0/K2)+(r-q+vol**2/2)*T)/(vol*sqrt(T))
    d2 = d1 - vol*sqrt(T)
    if right=='call':
        return S0*exp(-q*T)*norm.cdf(d1) - K1*exp(-r*T)*norm.cdf(d2)
    else:
        return K1*exp(-r*T)*norm.cdf(-d2) - S0*exp(-q*T)*norm.cdf(-d1)


