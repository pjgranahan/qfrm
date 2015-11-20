from qfrm import *


class Binary(OptionValuation):
    """
    Binary option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, method='BS', nsteps=None, npaths=None, keep_hist=False, payout_type="asset_or_nothing", Q=0.0):
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
        payout_type : str
                Required. Indicates whether the binary option is: "asset_or_nothing", "cash_or_nothing"
        Q : float
                Required if payout_type is "cash_or_nothing". Used in pricing a cash or nothing binary option.

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

        Use the Black-Scholes model to price an asset-or-nothing binary option

        >>> s = Stock(S0=42, vol=.20)
        >>> o = Binary(ref=s, right='put', K=40, T=.5, rf_r=.1)
        >>> o.calc_px(method='BS', payout_type="asset_or_nothing").px_spec
        qfrm.PriceSpec
        Q: 0.0
        d1: 0.7692626281060315
        d2: 0.627841271868722
        keep_hist: false
        method: BS
        payout_type: asset_or_nothing
        px: 9.276485780407903
        px_call: 32.7235142195921
        px_put: 9.276485780407903
        sub_method: asset_or_nothing
        <BLANKLINE>

        Access the attributes in other ways

        >>> o.px_spec.px, o.px_spec.d1, o.px_spec.d2, o.px_spec.method, o.px_spec.sub_method
        (9.276485780407903, 0.7692626281060315, 0.627841271868722, 'BS', 'asset_or_nothing')

        Change the option to be a call

        >>> o.update(right='call').calc_px().px_spec.px
        32.7235142195921

        Use the Black-Scholes model to price a cash-or-nothing binary option

        >>> s = Stock(S0=50, vol=.3)
        >>> o = Binary(ref=s, right='call', K=40, T=2, rf_r=.05)
        >>> o.calc_px(method='BS', payout_type="cash_or_nothing", Q=1000).px_spec
        qfrm.PriceSpec
        Q: 1000
        d1: 0.9737886891259003
        d2: 0.5495246204139719
        keep_hist: false
        method: BS
        payout_type: cash_or_nothing
        px: 641.2377052315655
        px_call: 641.2377052315655
        px_put: 263.59971280439396
        sub_method: cash_or_nothing
        <BLANKLINE>

        Access the attributes in other ways

        >>> o.px_spec.px, o.px_spec.d1, o.px_spec.d2, o.px_spec.method, o.px_spec.sub_method
        (641.2377052315655, 0.9737886891259003, 0.5495246204139719, 'BS', 'cash_or_nothing')

        Change the option to be a put

        >>> o.update(right='put').calc_px().px_spec.px
        8.2540367580782

        Example of option price development (BS method) with increasing maturities

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([o.update(T=t).calc_px(method='BS', payout_type="cash_or_nothing", Q=1000).px_spec.px for t in expiries], expiries)
        >>> O.plot(grid=1, title='Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()


        """
        self.px_spec = PriceSpec(method=method, sub_method=payout_type, nsteps=nsteps, npaths=npaths,
                                 keep_hist=keep_hist, payout_type=payout_type, Q=Q)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Binary

        .. sectionauthor:: Patrick Granahan

        """

        # Get additional pricing parameters that were provided
        payout_type = getattr(self.px_spec, 'payout_type')
        Q = getattr(self.px_spec, 'Q')

        # Convert the payout_type to lower case
        payout_type = payout_type.lower()

        # Explicit imports
        from math import log, exp, sqrt
        from scipy.stats import norm

        # Calculate d1 and d2
        d1 = ((log(self.ref.S0 / self.K)) + ((self.rf_r - self.ref.q + self.ref.vol ** 2 / 2) * self.T)) / (
            self.ref.vol * sqrt(self.T))
        d2 = d1 - (self.ref.vol * sqrt(self.T))

        # Price the asset-or-nothing binary option
        if payout_type == "asset_or_nothing":
            # Calculate the discount
            discount = self.ref.S0 * exp(-self.ref.q * self.T)

            # Compute the put and call price
            px_call = discount * norm.cdf(d1)
            px_put = discount * norm.cdf(-d1)

        # Price the cash-or-nothing binary option
        elif payout_type == "cash_or_nothing":
            # Calculate the discount
            discount = Q * exp(-self.rf_r * self.T)

            # Compute the put and call price
            px_call = discount * norm.cdf(d2)
            px_put = discount * norm.cdf(-d2)

        # The underlying is unknown
        else:
            raise "Unknown payout_type for binary option."

        # Store the correct price for the given right
        px = px_call if self.signCP == 1 else px_put if self.signCP == -1 else None

        # Record the price
        self.px_spec.add(px=float(px), px_call=float(px_call), px_put=float(px_put), d1=d1, d2=d2, Q=Q)

        return self

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Binary

        .. sectionauthor:: Tianyi Yao

        .. note::
        Implementing Binomial Trees:   http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1341181

        Examples
        -------
        #Example of using _calc_LT method
        >>> s = Stock(S0=42, vol=.20)
        >>> o = Binary(ref=s, right='put', K=40, T=.5, rf_r=.1)
        >>> print(o.calc_px(method='LT', nsteps=40,payout_type="asset_or_nothing").px_spec)
        qfrm.PriceSpec
        LT_specs:
          a: 1.0012507815756226
          d: 0.9778874672052759
          df_T: 0.951229424500714
          df_dt: 0.9987507809245809
          dt: 0.0125
          p: 0.5223760586914103
          u: 1.0226125536284048
        Q: 0.0
        keep_hist: false
        method: LT
        nsteps: 40
        payout_type: asset_or_nothing
        px: 8.952897200837354
        sub_method: asset_or_nothing
        <BLANKLINE>
        >>> print(o.update(right='put').calc_px().px_spec.px)
        9.276485780407903

        >>> s = Stock(S0=50, vol=.3)
        >>> o = Binary(ref=s, right='call', K=40, T=2, rf_r=.05)
        >>> print(o.calc_px(method='LT', nsteps=60, payout_type="cash_or_nothing", Q=1000).px_spec)
        qfrm.PriceSpec
        LT_specs:
          a: 1.001668056327482
          d: 0.9467007290508975
          df_T: 0.9048374180359595
          df_dt: 0.9983347214509387
          dt: 0.03333333333333333
          p: 0.5015299486608089
          u: 1.0563000210241065
        Q: 1000
        keep_hist: false
        method: LT
        nsteps: 60
        payout_type: cash_or_nothing
        px: 904.8374180359497
        sub_method: cash_or_nothing
        <BLANKLINE>

        >>> print(o.update(right='put').calc_px().px_spec.px)
        8.2540367580782

        Example of option price development (LT method) with increasing maturities

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([o.update(T=t).calc_px(method='LT', nsteps=60, payout_type="cash_or_nothing", Q=1000).px_spec.px for t in expiries], expiries)
        >>> O.plot(grid=1, title='Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()


        """
        from numpy import cumsum, log, arange, insert, exp, sqrt, sum, maximum

        #Retrieve parameters specific to binary option class
        payout_type = getattr(self.px_spec, 'payout_type')
        Q = getattr(self.px_spec, 'Q')

        # Convert the payout_type to lower case
        payout_type = payout_type.lower()

        #Extract LT parameters
        n = getattr(self.px_spec, 'nsteps', 3)
        _ = self.LT_specs(n)

        #Compute final nodes for asset_or_nothing payout type
        if payout_type == 'asset_or_nothing':
            S = self.ref.S0 * _['d'] ** arange(n, -1, -1) * _['u'] ** arange(0, n + 1) #Termial asset price
            O = maximum(self.signCP * (S - self.K), 0)           #terminal option value
            for ind in range(0,len(O)):
                if O[ind] > 0:
                    O[ind] = self.ref.S0
        #Compute final nodes for cash_or_nothing payout type
        else:
            S = Q * _['d'] ** arange(n, -1, -1) * _['u'] ** arange(0, n + 1)    #terminal stock price
            O = maximum(self.signCP * (S - self.K), 0)          # terminal option value
            for ind in range(0,len(O)):
                if O[ind] > 0:
                    O[ind] = Q

        #initialize tree structure
        S_tree, O_tree = None, None

        if getattr(self.px_spec, 'keep_hist', False):
            S_tree = (tuple([float(s) for s in S]),)
            O_tree = (tuple([float(o) for o in O]),)

            for i in range(n, 0, -1):
                O = _['df_dt'] * ((1 - _['p']) * O[:i] + ( _['p']) * O[1:])  #option prices at time step=i-1
                S = _['d'] * S[1:i+1]                                       # stock prices at time step=i-1

                S_tree = (tuple([float(s) for s in S]),) + S_tree
                O_tree = (tuple([float(o) for o in O]),) + O_tree

            out = O_tree[0][0]
        else:
            csl = insert(cumsum(log(arange(n) + 1)), 0, 0)
            tmp = csl[n] - csl - csl[::-1] + log(_['p']) * arange(n + 1) + log(1 - _['p']) * arange(n + 1)[::-1]
            out = (_['df_T'] * sum(exp(tmp) * tuple(O)))

        self.px_spec.add(px=float(out), sub_method=None,
                         LT_specs=_, ref_tree=S_tree, opt_tree=O_tree)


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
