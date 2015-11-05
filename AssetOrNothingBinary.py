__author__ = 'pjgranahan'


def pxBS(right, S, K, T, vol, r, q=0.0):
    """
    pxBS computes the price of a Binary option using the Black-Scholes model, given the parameters.

    From https://en.wikipedia.org/wiki/Binary_option:
        In finance, a binary option is a type of option in which the payoff can take only two possible outcomes,
        either some fixed monetary amount (or a precise predefined quantity or units of some asset) or nothing at all
        (in contrast to ordinary financial options that typically have a continuous spectrum of payoff)...

        For example, a purchase is made of a binary cash-or-nothing call option on XYZ Corp's stock struck at $100
        with a binary payoff of $1,000. Then, if at the future maturity date, often referred to as an expiry date, the
        stock is trading at above $100, $1,000 is received. If the stock is trading below $100, no money is received.
        And if the stock is trading at $100, the money is returned to the purchaser.

    :param right: "Call" or "Put" (case-insensitive).
    :param S: Price of the underlying instrument.
    :param K: Strike price.
    :param T: Time until expiry of the option (annualized).
    :param vol: Volatility.
    :param r: Risk free rate of return, continuously compounded and annualized.
    :param q: Dividend yield of the underlying, continuously compounded and annualized.
    :return: Value of the Binary option according to the Black-Scholes model.
    """

    # Explicit imports
    from math import log, exp, sqrt
    from scipy.stats import norm

    # Convert right to lower case
    right = right.lower()

    # Calculate d1
    d1 = ((log(S/K)) + (r - q + vol**2 / 2) * T) / (vol * sqrt(T))

    # Calculate the discount (for an asset-or-nothing binary option)
    discount = S * exp(-q * T)

    # Multiply d1 by -1 for put rights
    if right == "put":
        d1 *= -1

    # Compute the price, and round it to 8 places
    price = discount * norm.cdf(d1)
    price = round(price, 8)

    return price


# Test cases - checked against http://www.math.drexel.edu/~pg/fin/VanillaCalculator.html to 8 places
print(pxBS('call', 100, 100, 1, .2,  .05,  0))
assert pxBS('call', 100, 100, 1, .2,  .05,  0) == 0.53232483
assert pxBS('put',  100, 100, 1, .2,  .05,  0) == -0.53232483
assert pxBS('call', 100, 100, 1,  2,   .5, .1) == 0.12849676
assert pxBS('put',  100, 100, 1,  2,   .5, .1) == -0.12849676
assert pxBS('call', 100, 110, 10, .2, .05,  0) == 0.38023151
assert pxBS('put',  100, 110, 10, .2, .05,  0) == -0.38023151
