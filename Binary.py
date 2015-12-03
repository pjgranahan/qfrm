import math
import numpy as np

try: from qfrm.OptionValuation import *  # production:  if qfrm package is installed
except:   from OptionValuation import *  # development: if not installed and running from source


class Binary(OptionValuation):
    """
    Binary option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, method='BS', nsteps=None, npaths=None, keep_hist=False, payout_type="asset-or-nothing", Q=0.0):
        """ Wrapper function that calls appropriate valuation method.

        All parameters of ``calc_px`` are saved to local ``px_spec`` variable of class ``PriceSpec`` before
        specific pricing method (``_calc_BS()``,...) is called.
        An alternative to price calculation method ``.calc_px(method='BS',...).px_spec.px``
        is calculating price via a shorter method wrapper ``.pxBS(...)``.
        The same works for all methods (BS, LT, MC, FD).

        Parameters
        ----------
        method : str
                Required. Indicates a valuation method to be used:
                ``BS``: Black-Scholes Merton calculation
                ``LT``: Lattice tree (such as binary tree)
                ``MC``: Monte Carlo simulation methods
                ``FD``: finite differencing methods
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
            Returned object contains specifications and calculated price in embedded ``PriceSpec`` object.


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
        [1] `Binary Option on Wikipedia <https://en.wikipedia.org/wiki/Binary_option>`_

        Examples
        ------------

        BS Examples
        --------------

        Example #1 (verifiable using DerivaGem):Use the Black-Scholes model to price an asset-or-nothing binary option

        >>> s = Stock(S0=42, vol=.20)
        >>> o = Binary(ref=s, right='put', K=40, T=.5, rf_r=.1)
        >>> o.pxBS(payout_type="asset-or-nothing")  # doctest: +ELLIPSIS
        9.27648578

        Example #2 (verifiable using DerivaGem): Change the option to be a call

        >>> o.update(right='call').pxBS(payout_type="asset-or-nothing")  # doctest: +ELLIPSIS
        32.72351422

        Example #3 (verifiable using DerivaGem): Use the Black-Scholes model to price a cash-or-nothing binary option

        >>> s = Stock(S0=50, vol=.3)
        >>> o = Binary(ref=s, right='call', K=40, T=2, rf_r=.05)
        >>> o.pxBS(payout_type="cash-or-nothing", Q=1000)  # doctest: +ELLIPSIS
        641.237705232

        Example #4 (verifiable using DerivaGem): Change the option to be a put

        >>> o.update(right='put').pxBS(payout_type="cash-or-nothing", Q=1000)  #doctest: +ELLIPSIS
        263.599712804

        Example #5 (plot): Example of option price development (BS method) with increasing maturities

        >>> from pandas import Series
        >>> O = Series([o.update(T=t).pxBS(payout_type="asset-or-nothing") for t in range(1,11)], range(1,11))
        >>> O.plot(grid=1, title='Price vs expiry (in years)')  # doctest: +ELLIPSIS
        <...>
        >>> plt.show()







        Examples using _calc_LT()
        ----------------------------------------------

        Notes
        -------
        Verification of examples: DerivaGem software, Binary option (both cash_or_nothing and asset_or_nothing)

        Please note that the following LT examples will only generate results that matches the output of DerivaGem\
        if we use ``nsteps=365``. For fast runtime purpose, I use ``nsteps=10`` in the following examples, which may\
        not generate results that match the output of DerivaGem



        Use a binomial tree model to price a cash-or-nothing binary option

        The following example will generate px = 640.435924845...with ``nsteps = 365``, \
        which can be verified by GerivaGem However, for the purpose if fast runtime, I use ``nstep = 10`` \
        in all following examples, whose result does not match verification. \
        If you want to verify my code, please use ``nsteps = 365``.

        >>> s = Stock(S0=50, vol=.3)
        >>> o = Binary(ref=s, right='call', K=40, T=2, rf_r=.05, desc='call @641.237 put @263.6  DerivaGem')
        >>> o.calc_px(method='LT', nsteps=10, \
        payout_type="cash-or-nothing", Q=1000).px_spec.px #doctest: +ELLIPSIS
        572.299478496...

        >>> o.calc_px(method='LT', nsteps=10, \
        payout_type="cash-or-nothing", Q=1000).px_spec  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px: 572.299478497...


        Another way to view the specification of the binomial tree

        >>> o.calc_px(method='LT', nsteps=10, \
        payout_type="cash-or-nothing", Q=1000)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        Binary...px: 572.299478497...


        For the purpose of illustration, I only use a 2-step tree to display, \
        where the option price here is not accurate. This is just for viewing purpose.
        #the reference tree

        >>> o.calc_px(method='LT', nsteps=2, payout_type="cash-or-nothing", Q=1000, \
        keep_hist=True).px_spec.ref_tree #doctest: +ELLIPSIS
        ((50.000000000...,), (37.040911034..., 67.492940378...), (27.440581804..., 50.000000000..., 91.105940019...))

        #display the option value tree

        >>> o.calc_px(method='LT', nsteps=2, payout_type="cash-or-nothing", Q=1000, \
        keep_hist=True).px_spec.opt_tree # doctest: +ELLIPSIS
        ((687.356107822...,), (484.880509831..., 951.229424500...), (0.0, 1000.0, 1000.0))


        Another way to display option price
        The following example will generate px = 640.435924845...with ``nsteps = 365``, \
        which can be verified by GerivaGem.
        However, for the purpose if fast runtime, I use ``nstep = 10`` in all following examples, \
        whose result does not match verification. If you want to verify my code, please use ``nsteps = 365``.

        >>> (o.pxLT(nsteps=10, keep_hist=True, \
        payout_type='cash-or-nothing',Q=1000))
        572.299478497

        >>> (o.px_spec.px, o.px_spec.method)  # doctest: +ELLIPSIS
        (572.299478496..., 'LT')

        The following example will generate px = 264.401493191...with ``nsteps = 365``, \
        which can be verified by GerivaGem.
        However, for the purpose if fast runtime, I use ``nstep = 10`` in all following examples, \
        whose result does not match verification. If you want to verify my code, please use ``nsteps = 365``.

        >>> Binary(clone=o, right='put', desc='call @641.237 put @263.6  DerivaGem').calc_px(method='LT',\
        nsteps=10, payout_type='cash-or-nothing',Q=1000).px_spec.px #doctest: +ELLIPSIS
        332.537939539...


        Use a binomial tree model to price an asset-or-nothing binary option

        The following example will generate px = 41.717204143...with ``nsteps = 365``, \
        which can be verified by GerivaGem.
        However, for the purpose if fast runtime, I use ``nstep = 10`` in all following examples, \
        whose result does not match verification. If you want to verify my code, please use ``nsteps = 365``.

        >>> s = Stock(S0=50, vol=.3)
        >>> o = Binary(ref=s, right='call', K=40, T=2, rf_r=.05, desc='call @41.74 put @8.254 DerivaGem')
        >>> o.calc_px(method='LT', nsteps=10, payout_type="asset-or-nothing").px_spec.px #doctest: +ELLIPSIS
        39.009817494...

        >>> o.calc_px(method='LT', nsteps=10, \
        payout_type="asset-or-nothing").px_spec# doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px: 39.009817494...


        Another way to view the specification of the binomial tree

        >>> o.calc_px(method='LT', nsteps=10, \
        payout_type="asset-or-nothing")# doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        Binary...px: 39.009817494...


        For the purpose of illustration, I only use a 2-step tree to display, \
        where the option price here is not accurate

        >>> o.calc_px(method='LT', nsteps=2, payout_type="asset-or-nothing", \
        keep_hist=True).px_spec.ref_tree #doctest: +ELLIPSIS
        ((50.000000000...,), (37.040911034..., 67.492940378...), (27.440581804..., 50.000000000..., 91.105940019...))

        >>> o.calc_px(method='LT', nsteps=2, payout_type="asset-or-nothing", \
        keep_hist=True).px_spec.opt_tree #doctest: +ELLIPSIS
        ((44.032186316...,), (24.244025491..., 67.492940378...), (0.0, 50.000000000..., 91.105940019...))

        Another way to display option price
        The following example will generate px = 41.717204143...with ``nsteps = 365``, \
        which can be verified by GerivaGem.
        However, for the purpose if fast runtime, I use ``nstep = 10`` in all following examples, \
        whose result does not match verification. If you want to verify my code, please use ``nsteps = 365``.

        >>> (o.pxLT(nsteps=10, keep_hist=True, payout_type='asset-or-nothing'))
        39.009817494

        >>> (o.px_spec.px, o.px_spec.method)  #doctest: +ELLIPSIS
        (39.009817494..., 'LT')

        The following example will generate px = 8.282795856...with ``nsteps = 365``, \
        which can be verified by GerivaGem.
        However, for the purpose if fast runtime, I use ``nstep = 10`` in all following examples, \
        whose result does not match verification. If you want to verify my code, please use ``nsteps = 365``.

        >>> Binary(clone=o, right='put', desc='call @41.74 put @8.254 DerivaGem').calc_px(method='LT',\
        nsteps=10, payout_type='asset-or-nothing').px_spec.px #doctest: +ELLIPSIS
        10.990182505...


        Example of option price development (LT method) with increasing maturities

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([o.update(T=t).calc_px(method='LT', nsteps=10, payout_type="cash-or-nothing", \
        Q=1000).px_spec.px for t in expiries], expiries)
        >>> O.plot(grid=1, title='Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()

        FD Examples
        --------------
        Example #1

        >>> s = Stock(S0=42, vol=.20)
        >>> o = Binary(ref=s, right='put', K=40, T=.5, rf_r=.1)
        >>> o.pxFD(payout_type="asset-or-nothing", nsteps=10, npaths=10)  # doctest: +ELLIPSIS
        8.35783977

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
        >>> O = Series([o.update(T=t).pxFD(payout_type="asset-or-nothing",nsteps=t*5) for t in range(1,11)],range(1,11))
        >>> O.plot(grid=1, title='Price vs expiry (in years)')  # doctest: +ELLIPSIS
        <...>
        >>> plt.show()

        :Authors:
            Patrick Granahan,
            Tianyi Yao <ty13@rice.edu>
            Andrew Weatherly <amw13@rice.edu>
        """

        return super().calc_px(method=method, sub_method=payout_type, nsteps=nsteps, \
                               npaths=npaths, keep_hist=keep_hist, payout_type=payout_type, Q=Q)

    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        ----------
        self: Binary

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
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        Notes
        ----------------
        [1] `Implementing Binomial Trees, Manfred Gilli & Enrico Schumann, 2009
            <http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1341181>`_

        :Authors:
            Tianyi Yao <ty13@rice.edu>
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

    def _calc_FD(self, nsteps=10, npaths=10, keep_hist=False):
        """ Internal function for option valuation.

        Returns
        -------
        self: Binary

        .. sectionauthor:: Andrew Weatherly

        Notes
        -----

        Formulas:
        http://www.mathnet.or.kr/mathnet/thesis_file/BKMS-48-2-413-426.pdf
        https://studentportalen.uu.se/uusp-filearea-tool/download.action%3FnodeId%3D1101907%26toolAttachmentId%3D205921%
        26uusp.userId%3Dguest+&cd=2&hl=en&ct=clnk&gl=us

        """

        _ = self.px_spec

        # getting attributes
        M = getattr(_, 'npaths', 3)
        N = getattr(_, 'nsteps', 3)
        payout_type = getattr(_, 'payout_type')
        assert payout_type in ['asset-or-nothing', 'cash-or-nothing']
        if payout_type == 'cash-or-nothing':
            Q = getattr(_, 'Q', 5)

        # value grid
        C = np.zeros(shape=(N + 1, M + 1))
        # helpful parameters
        signCP = 1 if self.right.lower() == 'call' else -1
        T = self.T
        vol = self.ref.vol
        S0 = self.ref.S0
        S_Max = 2 * S0
        S_Min = 0
        S_vec = np.linspace(S_Min, S_Max, M + 1)
        r = self.rf_r
        q = self.ref.q
        dt = T / (N + 0.0)
        dS = S_Max / (M + 0.0)
        N = int(T / dt)
        M = int(S_Max / dS)
        K = self.K
        df = math.exp(-r * dt)

        # equations defined by hull on pg. 481
        def a(x):
            return df * (.5 * dt * ((vol ** 2) * (x ** 2) - (r - q) * x))

        def b(x):
            return df * (1 - dt * ((vol ** 2) * (x ** 2)))

        def c(x):
            return df * (.5 * dt * ((vol ** 2) * (x ** 2) + (r - q) * x))

        if payout_type == 'asset-or-nothing':
            if signCP == 1:
                # t = T boundary conditions
                for k in range(M + 1):
                    S = dS * k
                    if S > K:
                        C[N, k] = S
                    else:
                        C[N, k] = 0
                # Top and Bottom boundary conditions
                for i in range(N + 1):
                    t = i * dt
                    C[t, M] = S_Max
                    C[t, 0] = 0
            else:
                # t = T boundary conditions
                for k in range(M + 1):
                    S = dS * k
                    if S < K:
                        C[N, k] = S
                    else:
                        C[N, k] = 0
                # Top and Bottom boundary conditions
                for i in range(N + 1):
                    t = i * dt
                    C[t, M] = 0
                    C[t, 0] = S_Min
        else:
            if signCP == 1:
                # t = T boundary conditions
                for k in range(M + 1):
                    S = dS * k
                    if S > K:
                        C[N, k] = Q * math.exp(-r * T)
                    else:
                        C[N, k] = 0
                # Top and Bottom boundary conditions
                for i in range(N + 1):
                    t = i * dt
                    C[t, M] = Q * math.exp(-r * (T - t))
                    C[t, 0] = 0

            else:
                # t = T boundary conditions
                for k in range(M + 1):
                    S = dS * k
                    if S < K:
                        C[N, k] = Q * math.exp(-r * T)
                    else:
                        C[N, k] = 0
                # Top and Bottom boundary conditions
                for i in range(N + 1):
                    t = i * dt
                    C[t, M] = 0
                    C[t, 0] = Q * math.exp(-r * (T - t))

        for i in range(N - 1, -1, -1):
            for k in range(1, M):
                j = M - k
                C[i, k] = a(j) * C[i + 1, k + 1] + b(j) * C[i + 1, k] + c(j) * C[i + 1, k - 1]

        self.px_spec.add(px=np.interp(S0, S_vec, C[0, :]), method='FD', sub_method='Implicit')
        return self


