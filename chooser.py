__author__ = 'thawda'
def chooser(S , K , T , vol , rfr = 0.01 , q = 0 , right = "call" , tau = 0):
    """

    :param S: Underlying stock price
    :type S : float
    :param K: Stike price
    :type K : float
    :param T: Time to maturity of call option, measured in years
    :type T : float
    :param vol: Volatility of the underlying stock
    :type vol : float
    :param rfr: Risk free rate
    :type rfr : float

    :param q: Dividend yield of the underlying (CC)
    :type q : float

    :param right: call or put
    :type right : String

    :param tau: Time to maturity of put option, measured in years
    :type tau : float
    :return:

    Reference : Hull, John C.,Options, Futures and Other Derivatives, 9ed, 2014. Prentice Hall. ISBN 978-0-13-345631-8. http://www-2.rotman.utoronto.ca/~hull/ofod/index.html

                Huang Espen G., Option Pricing Formulas, 2ed. http://down.cenet.org.cn/upfile/10/20083212958160.pdf

                Wee, Lim Tiong, MFE5010 Exotic Options,Notes for Lecture 4 Chooser option. http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L4chooser.pdf

                Humphreys, Natalia A., ACTS 4302 Principles of Actuarial Models: Financial Economics. Lesson 14: All-or-nothing, Gap, Exchange and Chooser Options.
    """
    try:
        right   =   right.lower()
        S       =   float(S)
        K       =   float(K)
        T       =   float(T)
        vol     =   float(vol)
        rfr       =   float(rfr)
        q       =   float(q)
        tau    = float(tau)

    except:
        print('Right has to be string and everything else will be treated as float')
        return False

    assert right in ['call','put'], 'Make sure the right to be the "call" or "put" '
    assert vol > 0, 'Vol must be >=0'
    assert K > 0, 'K must be > 0'
    assert T > 0, 'T must be > 0'
    assert S >= 0, 'S must be >= 0'
    assert rfr >= 0, 'rfr must be >= 0'
    assert q >= 0, 'q must be >= 0'
    from numpy import sqrt , log, exp
    from scipy.stats import norm
    d2 = (log(S/K) + ((rfr - q - vol**2/2)*T) ) / ( vol * sqrt(T))
    d1 =  d2 + vol * sqrt( T)

    d2n = (log(S/K) + (rfr - q ) * T - vol**2 * tau /2) / ( vol * sqrt(tau))
    d1n = d2n + vol * sqrt(tau)

    price = S * exp(-q * T) * norm.cdf(d1) - K* exp(-rfr * T ) * norm.cdf(d2) + K* exp(-rfr * T ) * norm.cdf(-d2n)  - S* exp(-q * T) * norm.cdf(-d1n)
    return(price)
# p=chooser( S=50 , K=50 , vol = 0.2 , rfr = 0.06 , q = 0.02 , T = 9/12 , tau = 3/12 )
# print(p)