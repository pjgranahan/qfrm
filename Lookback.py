__author__ = 'Mengyan Xie'

def Lookback_BS(right, S, Sfl, T, vol, r, q=0):

    """
    Lookback_BS computes the price of a Lookback option given the parameters.

    :param S: Underlying price
    :type S: float

    param Sfl: Minimum asset price achieved to date for call option. Maximum asset price achieved to date for put option.
    :type Sfl: float

    :param T: Time to maturity, in years, possitive number.
    :type T: float

    :param vol: volatility
    :type vol: float

    :param r: risk free rate, continuously compounded, annualized
    :type r: float

    :param q: dividend yield of the underlying, continuously compounded, annualized
    :type q: float

    :return: value of the  GEOMETRIC Asian option
    :rtype: float

    :Example:
    >>> Lookback_BS('call', 50.0, 50.0, 0.25, 0.4, 0.1, 0) #returns 8.03712013961
    >>> Lookback_BS('put', 50.0, 50.0, 0.25, 0.4, 0.1, 0) #returns 7.79021925989

    .. see also::
        Note: must copy and paste full link into browser
        Formular: https://en.wikipedia.org/wiki/Lookback_option
    """

    assert right in ['call','put'], 'right must be "call" or "put" '
    assert S >= 0, 'S must be >= 0'
    assert Sfl >= 0, 'Sfl must be >= 0'
    assert T > 0, 'T must be > 0'
    assert vol > 0, 'vol must be >=0'
    assert r >= 0, 'r must be >= 0'
    assert q >= 0, 'q must be >= 0'

    # Imports
    from math import exp, log, sqrt
    from scipy.stats import norm

    # Parameters for Value Calculation
    signCP = 1 if right == 'call' else -1

    S_new = S / Sfl if right == 'call' else Sfl / S

    a1 = (log(S_new) + (signCP * (r - q) + vol ** 2 / 2) * T) / (vol * sqrt(T))
    a2 = a1 - vol * sqrt(T)
    a3 = (log(S_new) + signCP * (-r + q + vol ** 2 / 2) * T) / (vol * sqrt(T)) 
    Y1 = signCP * (-2 * (r - q - vol ** 2 / 2) * log(S_new)) / (vol ** 2)

    c = S * exp(-q * T) * norm.cdf(a1) - S * exp(-q * T) * (vol ** 2) * norm.cdf(-a1) / (2 * (r - q)) - Sfl * exp(-r * T) * (norm.cdf(a2) - vol ** 2 * exp(Y1) * norm.cdf(-a3) / (2 * (r - q)))  
    p = Sfl * exp(-r * T) * (norm.cdf(a1) - vol ** 2 * exp(Y1) * norm.cdf(-a3) / (2 * (r - q))) + S * exp(-q *T) * (vol ** 2) * norm.cdf(-a2) / (2 * (r - q)) - S * exp(-q * T) * norm.cdf(a2)

    return c if right == 'call' else p

print(Lookback_BS('put', 50.0, 50.0, 0.25, 0.4, 0.1, 0)) 
print(Lookback_BS('call', 50.0, 50.0, 0.25, 0.4, 0.1, 0)) 









