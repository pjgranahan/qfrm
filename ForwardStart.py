__author__ = 'Runmin Zhang'

class ForwardStart(OptionValuation):
    """ ForwardStart option class

    Inherits all methods and properties of Optionvalueation class.
    """

    def __init__(self):
        super.__init__(self,T1=0.)


    def pxBS(right='call', S=50, T=1.0, T1=0.5, vol=.2,r=0.02, q=0):
        """ pricing forward start options using Black-Scholes model.


        :param right: call or put
        :type right: str
        :param S: current (last) stock price of an underlying
        :type S: float
        :param K: option strike price
        :type K: float
        :param T: time to expiry (in years)
        :type T: float
        :param T1: time that the option starts
        :type T1:  floa
        :param vol: volatility of an underlying (as a ratio, not %)
        :type vol: float
        :param r: interest rate (matching expiry, T), as a ration, not as %
        :type r: float
        :param q: dividend yield
        :type q: float

        :return option price
        :rtype  float

        :Example:
        >>> pxBS() #Default
        4.4580186392862693
        >>> pxBS(right='put',S=50,T=1.0,T1=0.0,vol=.2,r=0.05,q=0) # Back to a European put option, Strike price=Current asset price
        2.7867630111284853
        ..seealso::
            https://en.wikipedia.org/wiki/Forward_start_option  -- WikiPedia: Forward start option
            http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2forward.pdf  -- How to pricing forward start opions
        """

        # Verify the input
        try:
            right   =   right.lower()
        except:
            print('Input error. right should be String')
            return False

        try:

            S   =   float(S)
            T   =   float(T)
            T1  =   float(T)
            vol =   float(vol)
            r   =   float(r)
            q   =   float(q)
        except:
            print('Input error. S, T, T1, vol, r, q should be floats.')
            return False

        assert right in ['call','put'], 'right should be either "call" or "put" '
        assert vol >= 0, 'vol >=0'
        assert T > 0, 'T > 0'
        assert T1 >=0, 'T1 >= 0'
        assert S >= 0, 'S >= 0'
        assert r >= 0, 'r >= 0'
        assert q >= 0, 'q >= 0'

        # Import external functions
        from scipy.stats import norm
        from math import sqrt, exp, log

        # Parameters in BSM
        K = S
        d1 = (log(S/K)+(r+vol**2/2)*T)/(vol*sqrt(T))
        d2 = d1 - vol*sqrt(T)

        # Calculate the option price
        if right == 'call':
            return (S*norm.cdf(d1)-K*exp(-(r-q)*T)*norm.cdf(d2))*exp(-q*T1)
        elif right == 'put':
            return (K*exp(-(r-q)*T)*norm.cdf(-d2)-S*norm.cdf(-d1))*exp(-q*T1)

