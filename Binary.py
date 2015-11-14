from qfrm import *


class Binary(OptionValuation):
    """
    Binary option class.

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
        self : Binary

        .. sectionauthor:: Patrick Granahan

        Notes
        -----
        In finance, a binary option is a type of option in which the payoff can take only two possible outcomes,
        either some fixed monetary amount (or a precise predefined quantity or units of some asset) or nothing at all
        (in contrast to ordinary financial options that typically have a continuous spectrum of payoff)...

        For example, a purchase is made of a binary cash-or-nothing call option on XYZ Corp's stock struck at $100
        with a binary payoff of $1,000. Then, if at the future maturity date, often referred to as an expiry date, the
        stock is trading at above $100, $1,000 is received. If the stock is trading below $100, no money is received.
        And if the stock is trading at $100, the money is returned to the purchaser. [1]

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Binary_option

        Examples
        -------

        >>> s = Stock(S0=42, vol=.20)
        >>> o = Binary(ref=s, right='put', K=40, T=.5, rf_r=.1, desc='call @0.81, put @4.76, Hull p.339')

        >>> o.calc_px(method='BS').px_spec   # save interim results to self.px_spec. Equivalent to repr(o)
        qfrm.PriceSpec
        d1: 0.7692626281060315
        d2: 0.627841271868722
        keep_hist: false
        method: BS
        px: 0.8085993729000922
        px_call: 4.759422392871532
        px_put: 0.8085993729000922
        sub_method: standard; Hull p.335

        >>> (o.px_spec.px, o.px_spec.d1, o.px_spec.d2, o.px_spec.method)  # alternative attribute access
        (0.8085993729000922, 0.7692626281060315, 0.627841271868722, 'BS')

        >>> o.update(right='call').calc_px().px_spec.px  # change option object to a put
        4.759422392871532

        >>> European(clone=o, K=41, desc='Ex. copy params; new strike.').calc_px(method='LT').px_spec.px
        4.2270039114413125

        """
        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Binary

        .. sectionauthor:: Patrick Granahan

        """

        # Explicit imports
        from math import log, exp, sqrt
        from scipy.stats import norm

        # Calculate d1 and d2
        d1 = ((log(self.ref.S0/self.K)) + ((self.rf_r - self.ref.q + self.ref.vol**2 / 2) * self.T)) / (self.ref.vol * sqrt(self.T))
        d2 = d1 - (self.ref.vol * sqrt(self.T))

        # Price the asset-or-nothing binary option
        if isinstance(self.ref, Stock):
            # Calculate the discount
            discount = self.ref.S0 * exp(-self.ref.q * self.T)

            # Compute the put and call price
            px_call = discount * norm.cdf(d1)
            px_put = discount * norm.cdf(-d1)

            # Store the type of binary option we priced
            sub_method = "asset-or-nothing"

        # Price the cash-or-nothing binary option
        elif isinstance(self.ref, Cash):
            # Calculate the discount
            discount = exp(-self.rf_r * self.T)

            # Compute the put and call price
            px_call = discount * norm.cdf(d2)
            px_put = discount * norm.cdf(-d2)

            # Store the type of binary option we priced
            sub_method = "cash-or-nothing"

        # The underlying is unknown
        else:
            raise "Unknown underlying for binary option."

        # Store the correct price for the given right
        px = px_call if self.signCP == 1 else px_put if self.signCP == -1 else None

        # Record the price
        self.px_spec.add(px=px, method='BS', sub_method=sub_method, px_call=px_call, px_put=px_put, d1=d1, d2=d2)

        return self

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Binary

        .. sectionauthor::

        .. note::
        Implementing Binomial Trees:   http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1341181

        """
        return self

    def _calc_MC(self, nsteps=3, npaths=4, keep_hist=False):
        """ Internal function for option valuation.

        Returns
        -------
        self: Binary

        .. sectionauthor::

        Notes
        -----
        Implementing Binomial Trees:   http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1341181

        """
        return self

    def _calc_FD(self, nsteps=3, npaths=4, keep_hist=False):
        """ Internal function for option valuation.

        Returns
        -------
        self: Binary

        .. sectionauthor::

        """
        return self



# Test cases - checked against http://investexcel.net/excel-binary-options/
assert pxBS('cash', 'call', 100, 100, 1, .2,  .05,  0) == round(0.5323248155, 8)
assert pxBS('cash', 'put',  100, 100, 1, .2,  .05,  0) == round(0.418904609, 8)
assert pxBS('cash', 'call', 100, 100, 1,  2,   .5, .1) == round(0.1284967947, 8)
assert pxBS('cash', 'put',  100, 100, 1,  2,   .5, .1) == round(0.478033865, 8)
assert pxBS('cash', 'call', 100, 110, 10, .2, .05,  0) == round(0.3802315498, 8)
assert pxBS('cash', 'put',  100, 110, 10, .2, .05,  0) == round(0.2262991099, 8)
# Or, to print the test cases:
print(pxBS('cash', 'call', 100, 100, 1, .2,  .05,  0))
print(pxBS('cash', 'put',  100, 100, 1, .2,  .05,  0))
print(pxBS('cash', 'call', 100, 100, 1,  2,   .5, .1))
print(pxBS('cash', 'put',  100, 100, 1,  2,   .5, .1))
print(pxBS('cash', 'call', 100, 110, 10, .2, .05,  0))
print(pxBS('cash', 'put',  100, 110, 10, .2, .05,  0))
# Test cases are not recommended for now - most online calculators that I've found have errors in their formulae
# # Test cases - checked against http://investexcel.net/excel-binary-options/
print(pxBS('asset', 'call', 100, 100, 1, .2,  .05,  0))
print(pxBS('asset', 'put',  100, 100, 1, .2,  .05,  0))
print(pxBS('asset', 'call', 100, 100, 1,  2,   .5, .1))
print(pxBS('asset', 'put',  100, 100, 1,  2,   .5, .1))
print(pxBS('asset', 'call', 100, 110, 10, .2, .05,  0))
print(pxBS('asset', 'put',  100, 110, 10, .2, .05,  0))
