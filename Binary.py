import math
import numpy as np

try:  from qfrm.European import *  # production:  if qfrm package is installed
except:    from European import *  # development: if not installed and running from source


class Binary(European):
    """ `Binary <https://en.wikipedia.org/wiki/Binary_option>`_ exotic option class.
    """

    def calc_px(self, payout_type="asset-or-nothing", Q=0, **kwargs):
        """ Wrapper function that calls appropriate valuation method.

        Parameters
        ----------
        payout_type : {'asset-or-nothing', 'cash-or-nothing'}
                Required. Indicates whether the binary option is: "asset-or-nothing", "cash-or-nothing"
        Q : float
                Required if payout_type is "cash-or-nothing". Used in pricing a cash or nothing binary option.
        kwargs : dict
            Keyword arguments (``method``, ``nsteps``, ``npaths``, ``keep_hist``, ``rng_seed``, ...)
            are passed to the parent. See ``European.calc_px()`` for details.

        Returns
        ------------
        self : Binary
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object).


        Notes
        ----------
        In finance, a binary option is a type of option in which the payoff can take only two possible outcomes,
        either some fixed monetary amount (or a precise predefined quantity or units of some asset) or nothing at all
        (in contrast to ordinary financial options that typically have a continuous spectrum of payoff)...

        For example, a purchase is made of a binary cash-or-nothing call option on XYZ Corp's stock struck at $100
        with a binary payoff of $1,000. Then, if at the future maturity date, often referred to as an expiry date, the
        stock is trading at above $100, $1,000 is received. If the stock is trading below $100, no money is received.
        And if the stock is trading at $100, the money is returned to the purchaser. [1]

        **FD**

        *References:*

        - Variable Time-Stepping Hybrid Finite Differences Methods for Pricing Binary Options, `Hongjoong Kim and Kyoung-Sook Moon, 2011 <http://1drv.ms/1ORS1xF>`_
        - Digital Options (Lecture 2, MFE5010 at NUS), `Lim Tiong Wee, 2001 <http://1drv.ms/1TA5dbz>`_
        - Excel Spreadsheets for Binary Options, `Excel tool. Samir Khan <http://investexcel.net/excel-binary-options/>`_
        - Pricing of binary options. `Online option pricer. <http://www.infres.enst.fr/~decreuse/pricer/en/index.php?page=binaire.html>`_
        - Extending [MC] model to price binary options, `2013. <http://www.codeandfinance.com/extending-price-binary-options.html>`_
        - Implementing Binomial Trees, `Manfred Gilli & Enrico Schumann, 2009 <http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1341181>`_

        Examples
        ------------

        **BS**

        User DerivaGem software (accompanying J.C.Hull's OFOD (2014) textbook) to verify examples below.

        >>> s = Stock(S0=42, vol=.20)
        >>> o = Binary(ref=s, right='put', K=40, T=.5, rf_r=.1)
        >>> o.pxBS(payout_type="asset-or-nothing", desc='DervaGem price is 9.276482815')
        9.27648578

        We can update the right from put to call and recompute.

        >>> o.update(right='call').pxBS(payout_type="asset-or-nothing", desc='DerivaGem price is 32.72351719'); o # doctest: +ELLIPSIS
        32.72351422...

        >>> s = Stock(S0=50, vol=.3)
        >>> o = Binary(ref=s, right='call', K=40, T=2, rf_r=.05)
        >>> o.pxBS(payout_type="cash-or-nothing", Q=1000, desc='DerivaGem price is 641.2377341'); o  # doctest: +ELLIPSIS
        641.237705232...

        Change the option to be a put

        >>> o.update(right='put').pxBS(payout_type="cash-or-nothing", Q=1000, desc='DervaGem price is 263.5996839')
        263.599712804

        Next example shows option (BS) price development with increasing maturities (1 year increments).

        >>> from pandas import Series, DataFrame
        >>> Ts = tuple(range(1, 101)) # a range of expiries
        >>> px_vec = Series([o.update(T=T).pxBS(payout_type="asset-or-nothing") for T in Ts], Ts)
        >>> px_vec.plot(grid=1, title='BS Price vs expiry (yrs) for ' + o.specs)  # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at 0x...>

        Next example shows sensitivity of Binary (asset-or-nothing) option's price to changes in expiry and strike.
        The visualization produces a contour plot at different strike levels.
        As expected for a call option, lower strike implies higher price (higher probability of in-the-money ITM).
        Note the more complex relationship between price and expiry dates.
        With distant expiries (over 5 years), the relationship is direct (higher expiry implies higher price).
        However, near maturity (1-5 years), the relationship depends on strike level (all else fixed).

        >>> from pandas import DataFrame
        >>> Ks = [30 + 4 * i for i in range(11)];   # a range of strikes
        >>> def px(K, T): return o.update(K=K, T=T).pxBS(payout_type="asset-or-nothing")
        >>> px_grid = [[px(K=K, T=T) for K in Ks] for T in Ts]
        >>> DataFrame(px_grid, columns=Ks).plot(grid=1, title='BS Price vs expiry at varying strikes, for ' + o.specs)  # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at 0x...>

        Next example shows option price sensitivity to expiry and current prices of the underlying asset.

        >>> from pandas import DataFrame
        >>> Ts = tuple(range(1, 101)) # a range of expiries
        >>> S0s = [30 + 4 * i for i in range(11)];   # a range of strikes
        >>> def px(S0, T):
        ...     s = Stock(S0=S0, vol=.3)
        ...     return Binary(ref=s, right='call', K=50, T=T, rf_r=.05).pxBS(payout_type="asset-or-nothing")
        >>> px_grid = [[px(S0=S0, T=T) for S0 in S0s] for T in Ts]
        >>> DataFrame(px_grid, columns=S0s).plot(grid=1, title='BS Price vs expiry at varying S0')  # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at 0x...>

        Next example shows option price sensitivity to expiry and volatility.

        >>> from pandas import DataFrame
        >>> Ts = range(1, 101) # a range of expiries
        >>> vols = [.05 + .025 * i for i in range(11)];   # a range of strikes
        >>> def px(vol, T):
        ...     s = Stock(S0=50, vol=vol)
        ...     return Binary(ref=s, right='call', K=50, T=T, rf_r=.05).pxBS(payout_type="asset-or-nothing")
        >>> px_grid = [[px(vol=vol, T=T) for vol in vols] for T in Ts]
        >>> DataFrame(px_grid, columns=vols).plot(grid=1, title='BS Price vs expiry at varying volatilities.')  # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at 0x...>

        **LT**

        Notes
        -------
        Verification of examples: DerivaGem software, Binary option (both cash_or_nothing and asset_or_nothing)

        *Cash or Nothing:*

        >>> s = Stock(S0=50, vol=.3)
        >>> o = Binary(ref=s, right='call', K=40, T=2, rf_r=.05, desc='call @641.237 put @263.6  DerivaGem')
        >>> o.calc_px(method='LT', nsteps=365, payout_type="cash-or-nothing", Q=1000).px_spec.px # doctest: +ELLIPSIS
        640.4359248459538

        >>> o.calc_px(method='LT', nsteps=365, payout_type="cash-or-nothing", Q=1000).px_spec  # doctest: +ELLIPSIS
        PriceSpec...px: 640.43592484...

        Another way to view the specification of the binomial tree

        >>> o.calc_px(method='LT', nsteps=365, payout_type="cash-or-nothing", Q=1000)  # doctest: +ELLIPSIS
        Binary...px: 640.43592484...

        Next example prints a 2 step (recombining) binomial tree of price progression of the underlying equity stock.

        >>> o.calc_px(method='LT', nsteps=2, payout_type="cash-or-nothing", Q=1000, keep_hist=True).px_spec.ref_tree #doctest: +ELLIPSIS
        ((50.000000000...,), (37.040911034..., 67.492940378...), (27.440581804..., 50.000000000..., 91.105940019...))

        Next example prints a 2 step (recombining) binomial tree of price progression of the binary option.

        >>> o.calc_px(method='LT', nsteps=2, payout_type="cash-or-nothing", Q=1000, keep_hist=True).px_spec.opt_tree # doctest: +ELLIPSIS
        ((687.356107822...,), (484.880509831..., 951.229424500...), (0.0, 1000.0, 1000.0))


        >>> o.pxLT(nsteps=365, payout_type='asset-or-nothing'); o    # doctest: +ELLIPSIS
        41.717204143...

        >>> o = Binary(clone=o, right='put', desc='DerivaGem: call @641.237, put @263.6')
        >>> o.calc_px(method='LT', nsteps=365, payout_type='cash-or-nothing', Q=1000) #doctest: +ELLIPSIS
        Binary...264.401493191...

        Use a binomial tree model to price an asset-or-nothing binary option

        >>> s = Stock(S0=50, vol=.3)
        >>> o = Binary(ref=s, right='call', K=40, T=2, rf_r=.05, desc='DerivaGem: call @41.74, put @8.254')
        >>> o.calc_px(method='LT', nsteps=365, payout_type="asset-or-nothing")  # doctest: +ELLIPSIS
        Binary...41.717204143...

        >>> o.calc_px(method='LT', nsteps=500, payout_type="asset-or-nothing").px_spec# doctest: +ELLIPSIS
        PriceSpec...px: 41.318664006...

        Another way to view the specification of the binomial tree

        >>> o.calc_px(method='LT', nsteps=500, payout_type="asset-or-nothing")  # doctest: +ELLIPSIS
        Binary...px: 41.318664006...


        Below is a 2-step binomial tree (for underlying stock prices and for option prices):

        >>> o.calc_px(method='LT', nsteps=2, payout_type="asset-or-nothing", keep_hist=True).px_spec.ref_tree #doctest: +ELLIPSIS
        ((50.000000000...,), (37.040911034..., 67.492940378...), (27.440581804..., 50.000000000..., 91.105940019...))

        >>> o.calc_px(method='LT', nsteps=2, payout_type="asset-or-nothing", keep_hist=True).px_spec.opt_tree #doctest: +ELLIPSIS
        ((44.032186316...,), (24.244025491..., 67.492940378...), (0.0, 50.000000000..., 91.105940019...))


        The following example will generate px = 8.282795856...with ``nsteps = 365``, \
        which can be verified by DerivaGem.
        However, for the purpose if fast runtime, I use ``nstep = 10`` in all following examples, \
        whose result does not match verification. If you want to verify my code, please use ``nsteps = 365``.

        >>> o = Binary(clone=o, right='put', desc='DerivaGem: call @41.74, put @8.254')
        >>> o.pxLT(nsteps=365, payout_type='asset-or-nothing'); o #doctest: +ELLIPSIS
        8.282795857...


        Example of option price development (LT method) with increasing maturities

        >>> from pandas import Series
        >>> s = Stock(S0=50, vol=.3)
        >>> o = Binary(ref=s, right='call', K=40, T=2, rf_r=.05, desc='DerivaGem: call @41.74, put @8.254')
        >>> Ts = range(1,11)   # range of expiries (in years)
        >>> O = Series([o.update(T=t).pxLT(nsteps=500, payout_type="cash-or-nothing", Q=1000) for t in Ts], Ts)
        >>> O.plot(grid=1, title='LT Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>

        **FD**

        Example #1 can verify this example with example 1 from pxBS above.

        >>> s = Stock(S0=42, vol=.20)
        >>> o = Binary(ref=s, right='put', K=40, T=.5, rf_r=.1)
        >>> (o.pxFD(payout_type="asset-or-nothing", nsteps=10, npaths=10), o.pxBS())  # doctest: +ELLIPSIS
        (8.35783977, 9.27648578)

        Example #2

        >>> o.update(right='call').pxFD(payout_type="asset-or-nothing", nsteps=10, npaths=10)  # doctest: +ELLIPSIS
        29.550733261

        Example #3

        >>> s = Stock(S0=50, vol=.3)
        >>> o = Binary(ref=s, right='call', K=40, T=2, rf_r=.05)
        >>> o.pxFD(payout_type="cash-or-nothing", Q=1000, nsteps=10, npaths=10)  # doctest: +ELLIPSIS
        473.924342837

        Example #4

        >>> o.update(right='put').pxFD(payout_type="cash-or-nothing", Q=1000, nsteps=10, npaths=10)  #doctest: +ELLIPSIS
        154.068135942

        Example #5 (plot): Example of option price development (FD method) with increasing maturities

        >>> from pandas import Series
        >>> s = Stock(S0=50, vol=.3)
        >>> o = Binary(ref=s, right='call', K=40, T=2, rf_r=.05)
        >>> O = Series([o.update(T=t).pxFD(payout_type="asset-or-nothing", nsteps=t*5) for t in range(1,101)],range(1,101))
        >>> O.plot(grid=1, title='FD Price vs expiry (in years)' + o.specs)  # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at 0x...>

        :Authors:
            Patrick Granahan,
            Tianyi Yao <ty13@rice.edu>,
            Andrew Weatherly <amw13@rice.edu>
        """

        assert payout_type in ['asset-or-nothing', 'cash-or-nothing']
        self.save2px_spec(payout_type=payout_type, Q=Q, **kwargs)
        return getattr(self, '_calc_' + self.px_spec.method.upper())()

    def _calc_BS(self):
        """ Internal function for option valuation.

        :Authors:
            Patrick Granahan
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
            px_call = discount * Util.norm_cdf(d1)
            px_put = discount * Util.norm_cdf(-d1)

        # Price the cash-or-nothing binary option
        elif payout_type == "cash-or-nothing":
            # Calculate the discount
            discount = Q * math.exp(-self.rf_r * self.T)

            # Compute the put and call price
            px_call = discount * Util.norm_cdf(d2)
            px_put = discount * Util.norm_cdf(-d2)

        # The underlying is unknown
        else:
            raise Exception("Unknown payout_type for binary option.")

        # Store the correct price for the given right
        px = px_call if self.signCP == 1 else px_put if self.signCP == -1 else None

        # Record the price
        self.px_spec.add(px=float(px), px_call=float(px_call), px_put=float(px_put), d1=d1, d2=d2, Q=Q)

        return self

    def _calc_LT(self):
        """ Internal function for option valuation.        See ``calc_px()`` for complete documentation.

        :Authors:
            Tianyi Yao <ty13@rice.edu>
        """


        #Retrieve parameters specific to binary option class
        payout_type = getattr(self.px_spec, 'payout_type')
        Q = getattr(self.px_spec, 'Q')

        # Convert the payout_type to lower case
        payout_type = payout_type.lower()

        #Extract LT parameters
        n = self.px_spec.nsteps
        _ = self._LT_specs()

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
        """ Internal function for option valuation.     See ``calc_px()`` for complete documentation.    """
        return self

    def _calc_FD(self, nsteps=10, npaths=10, keep_hist=False):
        """ Internal function for option valuation. See ``calc_px()`` for complete documentation.

        :Authors:
            Andrew Weatherly
        """
        _ = self;       signCP, T, rf_r, K = _.signCP, _.T, _.rf_r, _.K
        _ = self.ref;   S0, vol, q = _.S0, _.vol, _.q
        _ = self.px_spec;    n, m, payout_type = _.nsteps, _.npaths, _.payout_type
        S_Max, S_Min = 2 * S0, 0
        if payout_type == 'cash-or-nothing':        Q = getattr(_, 'Q', 5)

        C = np.zeros(shape=(n + 1, m + 1))  # value grid
        S_vec = np.linspace(S_Min, S_Max, m + 1)
        dt = T / (n + 0.0)
        dS = S_Max / (m + 0.0)
        n = int(T / dt)
        m = int(S_Max / dS)
        df = math.exp(-rf_r * dt)

        # equations defined by hull on pg. 481
        def a(x):     return df * (.5 * dt * ((vol ** 2) * (x ** 2) - (rf_r - q) * x))
        def b(x):     return df * (1 - dt * ((vol ** 2) * (x ** 2)))
        def c(x):     return df * (.5 * dt * ((vol ** 2) * (x ** 2) + (rf_r - q) * x))

        if payout_type == 'asset-or-nothing':
            if signCP == 1:
                # t = T boundary conditions
                for k in range(m + 1):
                    S = dS * k
                    if S > K: C[n, k] = S
                    else:     C[n, k] = 0
                # Top and Bottom boundary conditions
                for i in range(n + 1):
                    t = i * dt
                    C[t, m] = S_Max
                    C[t, 0] = 0
            else:
                # t = T boundary conditions
                for k in range(m + 1):
                    S = dS * k
                    if S < K: C[n, k] = S
                    else: C[n, k] = 0
                # Top and Bottom boundary conditions
                for i in range(n + 1):
                    t = i * dt
                    C[t, m] = 0
                    C[t, 0] = S_Min
        else:
            if signCP == 1:
                # t = T boundary conditions
                for k in range(m + 1):
                    S = dS * k
                    if S > K: C[n, k] = Q * math.exp(-rf_r * T)
                    else: C[n, k] = 0
                # Top and Bottom boundary conditions
                for i in range(n + 1):
                    t = i * dt
                    C[t, m] = Q * math.exp(-rf_r * (T - t))
                    C[t, 0] = 0

            else:
                # t = T boundary conditions
                for k in range(m + 1):
                    S = dS * k
                    if S < K: C[n, k] = Q * math.exp(-rf_r * T)
                    else: C[n, k] = 0
                # Top and Bottom boundary conditions
                for i in range(n + 1):
                    t = i * dt
                    C[t, m] = 0
                    C[t, 0] = Q * math.exp(-rf_r * (T - t))

        for i in range(n - 1, -1, -1):
            for k in range(1, m):
                j = m - k
                C[i, k] = a(j) * C[i + 1, k + 1] + b(j) * C[i + 1, k] + c(j) * C[i + 1, k - 1]

        self.px_spec.add(px=np.interp(S0, S_vec, C[0, :]), sub_method='Implicit FDM')
        return self
