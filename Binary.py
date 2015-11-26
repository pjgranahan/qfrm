from OptionValuation import *
import numpy as np
import math
import scipy.stats


class Binary(OptionValuation):
    """
    Binary option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, method='BS', nsteps=None, npaths=None, keep_hist=False, payout_type="asset-or-nothing", Q=0.0):
        """ Wrapper function that calls appropriate valuation method.

        User passes parameters to calc_px, which saves them to local PriceSpec object
        and calls specific pricing function (_calc_BS,...).
        This makes significantly less docstrings to write, since user is not interfacing pricing functions,
        but a wrapper function calc_px().

        Parameters
        ---------------
        method : str
                Required. Indicates a valuation method to be used: 'BS', 'LT', 'MC', 'FD'
        nsteps : int
                LT, MC, FD methods require number of times steps
        npaths : int
                MC, FD methods require number of simulation paths
        keep_hist : bool
                If True, historical information (trees, simulations, grid) are saved in self.px_spec object.
        payout_type : str
                Required. Indicates whether the binary option is: "asset-or-nothing", "cash-or-nothing"
        Q : float
                Required if payout_type is "cash-or-nothing". Used in pricing a cash or nothing binary option.

        Returns
        ------------
        self : Binary

        .. sectionauthor:: Patrick Granahan, Tianyi Yao

        Notes
        ----------
        In finance, a binary option is a type of option in which the payoff can take only two possible outcomes,
        either some fixed monetary amount (or a precise predefined quantity or units of some asset) or nothing at all
        (in contrast to ordinary financial options that typically have a continuous spectrum of payoff)...

        For example, a purchase is made of a binary cash-or-nothing call option on XYZ Corp's stock struck at $100
        with a binary payoff of $1,000. Then, if at the future maturity date, often referred to as an expiry date, the
        stock is trading at above $100, $1,000 is received. If the stock is trading below $100, no money is received.
        And if the stock is trading at $100, the money is returned to the purchaser. [1]

        References
        -------------
        [1] https://en.wikipedia.org/wiki/Binary_option

        Examples
        ------------

        Use the Black-Scholes model to price an asset-or-nothing binary option. Verifiable using DerivaGem.

        >>> s = Stock(S0=42, vol=.20)
        >>> o = Binary(ref=s, right='put', K=40, T=.5, rf_r=.1)
        >>> o.calc_px(method='BS', payout_type="asset-or-nothing").px_spec #doctest: +ELLIPSIS
        PriceSpec
        Q: 0.0
        d1: 0.76926...
        d2: 0.62784...
        keep_hist: false
        method: BS
        payout_type: asset-or-nothing
        px: 9.27648...
        px_call: 32.723...
        px_put: 9.2764...
        sub_method: asset-or-nothing
        <BLANKLINE>

        Access the attributes in other ways

        >>> o.px_spec.px, o.px_spec.d1, o.px_spec.d2, o.px_spec.method, o.px_spec.sub_method #doctest: +ELLIPSIS
        (9.276485..., 0.76926262..., 0.62784127..., 'BS', 'asset-or-nothing')

        Change the option to be a call

        >>> o.update(right='call').calc_px().px_spec.px #doctest: +ELLIPSIS
        32.723514...

        Use the Black-Scholes model to price a cash-or-nothing binary option. Verifiable using DerivaGem.

        >>> s = Stock(S0=50, vol=.3)
        >>> o = Binary(ref=s, right='call', K=40, T=2, rf_r=.05)
        >>> o.calc_px(method='BS', payout_type="cash-or-nothing", Q=1000).px_spec #doctest: +ELLIPSIS
        PriceSpec
        Q: 1000
        d1: 0.973788...
        d2: 0.549524...
        keep_hist: false
        method: BS
        payout_type: cash-or-nothing
        px: 641.2377...
        px_call: 641.237...
        px_put: 263.5997...
        sub_method: cash-or-nothing
        <BLANKLINE>

        Access the attributes in other ways

        >>> o.px_spec.px, o.px_spec.d1, o.px_spec.d2, o.px_spec.method, o.px_spec.sub_method #doctest: +ELLIPSIS
        (641.23770..., 0.9737886..., 0.549524..., 'BS', 'cash-or-nothing')

        Change the option to be a put

        >>> o.update(right='put').calc_px().px_spec.px #doctest: +ELLIPSIS
        8.25403...

        Example of option price development (BS method) with increasing maturities

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([o.update(T=t).calc_px(method='BS', \
        payout_type="cash-or-nothing", Q=1000).px_spec.px for t in expiries], expiries)
        >>> O.plot(grid=1, title='Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()







        Examples using _calc_LT()
        -------------------------------

        Notes
        -------
        Verification of examples

        DerivaGem software, Binary option (both cash_or_nothing and asset_or_nothing)

        Please note that the following LT examples will only generate results that matches the

        Use a binomial tree model to price a cash-or-nothing binary option
        >>> s = Stock(S0=50, vol=.3)
        >>> o = Binary(ref=s, right='call', K=40, T=2, rf_r=.05, desc='call @641.237 put @263.6  DerivaGem')
        >>> print(o.calc_px(method='LT', nsteps=365, \
        payout_type="cash-or-nothing", Q=1000).px_spec.px)#doctest: +ELLIPSIS
        640.43592...

        >>> print(o.calc_px(method='LT', nsteps=365, \
        payout_type="cash-or-nothing", Q=1000).px_spec)  # doctest: +ELLIPSIS
        PriceSpec
        LT_specs:
          a: 1.000274...
          d: 0.978037...
          df_T: 0.90483...
          df_dt: 0.99972...
          dt: 0.0054794...
          p: 0.5006174...
          u: 1.02245540...
        Q: 1000
        keep_hist: false
        method: LT
        nsteps: 365
        payout_type: cash-or-nothing
        px: 640.435924...
        sub_method: cash-or-nothing
        <BLANKLINE>

        Another way to view the specification of the binomial tree

        >>> print(o.calc_px(method='LT', nsteps=365, payout_type="cash-or-nothing", Q=1000))  # doctest: +ELLIPSIS
        Binary
        K: 40
        T: 2
        _right: call
        _signCP: 1
        desc: call @641.237 put @263.6  DerivaGem
        frf_r: 0
        px_spec: PriceSpec
          LT_specs:
            a: 1.00027...
            d: 0.9780...
            df_T: 0.904...
            df_dt: 0.99972...
            dt: 0.005479...
            p: 0.50061...
            u: 1.02245...
          Q: 1000
          keep_hist: false
          method: LT
          nsteps: 365
          payout_type: cash-or-nothing
          px: 640.4359...
          sub_method: cash-or-nothing
        ref: Stock
          S0: 50
          curr: -
          desc: -
          q: 0
          tkr: -
          vol: 0.3
        rf_r: 0.05
        seed0: -
        <BLANKLINE>

        For the purpose of illustration, I only use a 2-step tree to display, \
        where the option price here is not accurate
        #the reference tree
        >>> print(o.calc_px(method='LT', nsteps=2, payout_type="cash-or-nothing", Q=1000, \
        keep_hist=True).px_spec.ref_tree) #doctest: +ELLIPSIS
        ((50.0000...,), (37.0409..., 67.49294...), (27.44058..., 50.000000..., 91.10594...))

        #the option value tree
        >>> print(o.calc_px(method='LT', nsteps=2, payout_type="cash-or-nothing", Q=1000, \
        keep_hist=True).px_spec.opt_tree) # doctest: +ELLIPSIS
        ((687.35610...,), (484.88050..., 951.2294...), (0.0, 1000.0, 1000.0))

        Another way to display option price
        >>> print((o.pxLT(nsteps=365, keep_hist=True, payout_type='cash-or-nothing',Q=1000))) #doctest: +ELLIPSIS
        640.43592...

        >>> print((o.px_spec.px, o.px_spec.method))  # doctest: +ELLIPSIS
        (640.43592..., 'LT')

        >>> print(Binary(clone=o, right='put', desc='call @641.237 put @263.6  DerivaGem').calc_px(method='LT',\
        nsteps=365, payout_type='cash-or-nothing',Q=1000).px_spec.px) #doctest: +ELLIPSIS
        264.4014...


        Use a binomial tree model to price an asset-or-nothing binary option
        >>> s = Stock(S0=50, vol=.3)
        >>> o = Binary(ref=s, right='call', K=40, T=2, rf_r=.05, desc='call @41.74 put @8.254 DerivaGem')
        >>> print(o.calc_px(method='LT', nsteps=365, payout_type="asset-or-nothing").px_spec.px) #doctest: +ELLIPSIS
        41.71720...

        >>> print(o.calc_px(method='LT', nsteps=365, payout_type="asset-or-nothing").px_spec)  #doctest: +ELLIPSIS
        PriceSpec
        LT_specs:
          a: 1.0002...
          d: 0.97803...
          df_T: 0.9048...
          df_dt: 0.9997...
          dt: 0.005479...
          p: 0.50061742...
          u: 1.0224554...
        Q: 0.0
        keep_hist: false
        method: LT
        nsteps: 365
        payout_type: asset-or-nothing
        px: 41.7172...
        sub_method: asset-or-nothing
        <BLANKLINE>

        Another way to view the specification of the binomial tree

        >>> print(o.calc_px(method='LT', nsteps=365, payout_type="asset-or-nothing"))  #doctest: +ELLIPSIS
        Binary
        K: 40
        T: 2
        _right: call
        _signCP: 1
        desc: call @41.74 put @8.254 DerivaGem
        frf_r: 0
        px_spec: PriceSpec
          LT_specs:
            a: 1.0002...
            d: 0.978037...
            df_T: 0.90483...
            df_dt: 0.9997...
            dt: 0.0054794...
            p: 0.50061...
            u: 1.022455...
          Q: 0.0
          keep_hist: false
          method: LT
          nsteps: 365
          payout_type: asset-or-nothing
          px: 41.717204...
          sub_method: asset-or-nothing
        ref: Stock
          S0: 50
          curr: -
          desc: -
          q: 0
          tkr: -
          vol: 0.3
        rf_r: 0.05
        seed0: -
        <BLANKLINE>

        For the purpose of illustration, I only use a 2-step tree to display, \
        where the option price here is not accurate

        >>> print(o.calc_px(method='LT', nsteps=2, payout_type="asset-or-nothing", \
        keep_hist=True).px_spec.ref_tree) #doctest: +ELLIPSIS
        ((50.0000...,), (37.0409..., 67.49294...), (27.440581..., 50.000000..., 91.1059400...))

        >>> print(o.calc_px(method='LT', nsteps=2, payout_type="asset-or-nothing", \
        keep_hist=True).px_spec.opt_tree)#doctest: +ELLIPSIS
        ((44.0321...,), (24.24402..., 67.4929403...), (0.0, 50.0000000..., 91.105940...))

        Another way to display option price
        >>> print((o.pxLT(nsteps=365, keep_hist=True, payout_type='asset-or-nothing')))#doctest: +ELLIPSIS
        41.71720...

        >>> print((o.px_spec.px, o.px_spec.method))  #doctest: +ELLIPSIS
        (41.7172..., 'LT')

        >>> print(Binary(clone=o, right='put', desc='call @41.74 put @8.254 DerivaGem').calc_px(method='LT',\
        nsteps=365, payout_type='asset-or-nothing').px_spec.px) #doctest: +ELLIPSIS
        8.28279...


        Example of option price development (LT method) with increasing maturities

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([o.update(T=t).calc_px(method='LT', nsteps=365, payout_type="cash-or-nothing", \
        Q=1000).px_spec.px for t in expiries], expiries)
        >>> O.plot(grid=1, title='Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()
        """

        return super().calc_px(method=method, sub_method=payout_type, nsteps=nsteps, \
                               npaths=npaths, keep_hist=keep_hist, payout_type=payout_type, Q=Q)
    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        ----------
        self: Binary

        .. sectionauthor:: Patrick Granahan

        """

        # Get additional pricing parameters that were provided
        payout_type = getattr(self.px_spec, 'payout_type')
        Q = getattr(self.px_spec, 'Q')

        # Convert the payout_type to lower case
        payout_type = payout_type.lower()

        # Calculate d1 and d2
        d1 = ((math.log(self.ref.S0 / self.K)) + ((self.rf_r - self.ref.q + self.ref.vol ** 2 / 2) * self.T)) / (
            self.ref.vol * math.sqrt(self.T))
        d2 = d1 - (self.ref.vol * math.sqrt(self.T))

        # Price the asset-or-nothing binary option
        if payout_type == "asset-or-nothing":
            # Calculate the discount
            discount = self.ref.S0 * math.exp(-self.ref.q * self.T)

            # Compute the put and call price
            px_call = discount * scipy.stats.norm.cdf(d1)
            px_put = discount * scipy.stats.norm.cdf(-d1)

        # Price the cash-or-nothing binary option
        elif payout_type == "cash-or-nothing":
            # Calculate the discount
            discount = Q * math.exp(-self.rf_r * self.T)

            # Compute the put and call price
            px_call = discount * scipy.stats.norm.cdf(d2)
            px_put = discount * scipy.stats.norm.cdf(-d2)

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
        ---------
        self: Binary

        .. sectionauthor:: Tianyi Yao

        .. note::
        Implementing Binomial Trees:   http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1341181

        """


        #Retrieve parameters specific to binary option class
        payout_type = getattr(self.px_spec, 'payout_type')
        Q = getattr(self.px_spec, 'Q')

        # Convert the payout_type to lower case
        payout_type = payout_type.lower()

        #Extract LT parameters
        n = getattr(self.px_spec, 'nsteps', 3)
        _ = self.LT_specs(n)

        #Compute final nodes for asset_or_nothing payout type
        if payout_type == 'asset-or-nothing':
            S = self.ref.S0 * _['d'] ** np.arange(n, -1, -1) * _['u'] ** np.arange(0, n + 1) #Termial asset price
            O = np.maximum(self.signCP * (S - self.K), 0)           #terminal option value
            for ind in range(0,len(O)):
                if O[ind] > 0:
                    O[ind] = S[ind]
        #Compute final nodes for cash_or_nothing payout type
        else:
            S = self.ref.S0 * _['d'] ** np.arange(n, -1, -1) * _['u'] ** np.arange(0, n + 1)   #terminal stock price
            O = np.maximum(self.signCP * (S - self.K), 0)          # terminal option value
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
            csl = np.insert(np.cumsum(np.log(np.arange(n) + 1)), 0, 0)
            tmp = csl[n] - csl - csl[::-1] + np.log(_['p']) * np.arange(n + 1) + \
                  np.log(1 - _['p']) * np.arange(n + 1)[::-1]
            out = (_['df_T'] * sum(np.exp(tmp) * tuple(O)))


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
