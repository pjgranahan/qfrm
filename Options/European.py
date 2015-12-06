import math

try: from qfrm.Options.OptLib import *  # production:  if qfrm package is installed
except:   from Options.OptLib import *  # development: if not installed and running from source


class European(Opt):
    """ European option class.

    Inherits all methods and properties of ``OptValSpec`` class.
    """

    def __init__(self , *args, **kwargs):
        super().__init__(*args, **kwargs)  # pass remaining arguments to base (parent) class


    def calc_px(self, method='BS', nsteps=None, npaths=None, keep_hist=False, rng_seed=None):
        """ Wrapper function that calls appropriate valuation method.

        All parameters of ``calc_px`` are saved to local ``px_spec`` variable of class ``PriceSpec`` before
        specific pricing method (``_calc_BS()``,...) is called.
        An alternative to price calculation method ``.calc_px(method='BS',...).px_spec.px``
        is calculating price via a shorter method wrapper ``.pxBS(...)``.
        The same works for all methods (BS, LT, MC, FD).


        Parameters
        -------------
        method : str
            Required. Indicates a valuation method to be used:

            ``BS`` -- Black-Scholes Merton calculation

            ``LT`` -- Lattice tree (such as binary tree or binomial tree)

            ``MC`` -- Monte Carlo simulation methods

            ``FD`` -- finite differencing methods

        nsteps : int
            LT, MC, FD methods require number of times steps
        npaths : int
            MC, FD methods require number of simulation paths
        keep_hist : bool
            If ``True``, historical information (trees, simulations, grid) are saved in ``self.px_spec`` object.
        rng_seed : int, None
            (non-negative) integer used to seed random number generator (RNG) for MC pricing.

            ``None`` -- no seeding; generates random sequence for MC

        Returns
        -------
        European
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object).


        Notes
        -----
        **Black-Scholes Merton (BS)**, i.e. exact solution pricing.
        This pricing method uses Black Scholes Merton differential equation to price the European option.
        The computation is well-studied and covered by J.C.Hull (with numerous examples).

        *References:*
        The Black-Scholes-Merton Model (Ch.15), OFOD, J.C.Hull, 9ed, 2014, p.346


        **Lattice Tree (LT)**, i.e. binomial or binary (recombining) tree pricing.
        Binomial tree is used to (discretely) grow the underlying stock price.
        Then terminal option payoffs are discounted to the presented,
        while averaged at each prior node (as part of computing discounted expected value of an option).

        *References:*
        Binomial Trees (Ch.13), OFOD, J.C.Hull, 9ed, 2014, p.274

        **Monte Carlo simulation (MC)**.
        First, simulate stock prices, according to Geometric Brownian motion (GBM) model.
        Then compute terminal option payoffs and discount them back to present.
        Averaged present value is the desired option price.

        *References:*
        Monte Carlo Simulation (Ch.21-6), OFOD, J.C.Hull, 9ed, 2014, pp.469-475


        Examples
        --------

        **BS:**
        *Verifiable example:*
        See J.C.Hull's OFOD textbook, pp.338-339, for theory and calculation explanation.
        The ``px_spec`` variable references an object containing all intermediate calculations
        and final option price. ``calc_px()`` method runs the calculations and saves interim results.

        >>> s = Stock(S0=42, vol=.20)
        >>> o = European(ref=s, right='put', K=40, T=.5, rf_r=.1, desc='call @0.81, put @4.76, Hull p.339')
        >>> o.calc_px(method='BS').px_spec   # doctest: +ELLIPSIS
        PriceSpec...px: 0.808599373...

        Here are just some variables computed interim:

        >>> (o.px_spec.px, o.px_spec.d1, o.px_spec.d2, o.px_spec.method)  # doctest: +ELLIPSIS
        (0.808599372..., 0.769262628..., 0.627841271..., 'BS')

        You can update option specifications and recalculate the price.
        Here we change the right to a put. ``pxBS()`` is a wrapper method for ``calc_px()``.

        >>> o.update(right='call').pxBS()
        4.759422393


        **LT:**

        You can clone an existing option into a different option style.
        Here we create European option from American option (created earlier)
        and change the strike price ``K`` and ``desc`` specs.

        >>> European(clone=o, K=41, desc='Ex. copy params; new strike.').pxLT()
        4.227003911

        *Verifiable example:*
        Here we price an option on stock index, yielding a 2% (annualized rate) continuous dividend.
        See J.C.Hull's OFOD, Fig.13.11, p.291, Lattice tree yields an option price of 53.39.

        >>> s = Stock(S0=810, vol=.2, q=.02)
        >>> o = European(ref=s, right='call', K=800, T=.5, rf_r=.05, desc='53.39, Hull p.291')
        >>> o.pxLT(nsteps=2)
        53.394716375

        >>> o.pxLT(nsteps=2, keep_hist=True)  # option price from a 3-step tree (that's 2 time intervals)
        53.394716375

        Here are tree nodes of (recombining) binomial tree for the progression of prices of the underlying index.

        >>> o.px_spec.ref_tree  # prints reference tree  # doctest: +ELLIPSIS
        ((810.00...), (732.9183086091272, 895.1884436412747), (663.1719099931652, 810.0, 989.3362341097378))

        Here are tree nodes of (recombining) binomial tree for the progression of prices of the European stock.

        >>> o.calc_px(method='LT', nsteps=2, keep_hist=True).px_spec.opt_tree
        ((53.39471637496134,), (5.062315192620067, 100.66143225703827), (0, 10.0, 189.3362341097378))

        A complete output of all calculated values leading to the option price.

        >>> o.calc_px(method='LT', nsteps=2)   # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        European...px: 53.394716375...


        **MC:**

        >>> s = Stock(S0=810, vol=.2, q=.02)
        >>> o = European(ref=s, right='call', K=800, T=.5, rf_r=.05, desc='53.39, Hull p.291')
        >>> o.pxMC(nsteps=10, npaths=10, rng_seed=1)
        74.747081425000005



        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """

        return super().calc_px(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist, rng_seed=rng_seed)

    def _calc_BS(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """

        _ = self
        d1 = (math.log(_.ref.S0 / _.K) + (_.rf_r + _.ref.vol ** 2 / 2.) * _.T)/(_.ref.vol * math.sqrt(_.T))
        d2 = d1 - _.ref.vol * math.sqrt(_.T)
        N = Util.norm_cdf

        # if calc of both prices is cheap, do both and include them into Price object.
        # Price.px should always point to the price of interest to the user
        # Save values as basic data types (int, floats, str), instead of np.array
        px_call = float(_.ref.S0 * math.exp(-_.ref.q * _.T) * N(d1) - _.K * math.exp(-_.rf_r * _.T) * N(d2))
        px_put = float(- _.ref.S0 * math.exp(-_.ref.q * _.T) * N(-d1) + _.K * math.exp(-_.rf_r * _.T) * N(-d2))
        px = px_call if _.signCP == 1 else px_put if _.signCP == -1 else None

        self.px_spec.add(px=px, sub_method='standard; Hull p.335', px_call=px_call, px_put=px_put, d1=d1, d2=d2)

        return self

    def _calc_LT(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """

        n = getattr(self.px_spec, 'nsteps', 3)
        _ = self.LT_specs(n)
        incr_n, decr_n = Vec(Util.arange(0, n + 1)), Vec(Util.arange(n, -1)) #Vectorized tuple. See Util.py. 0..n; n..0.

        S = Vec(_['d'])**decr_n * Vec(_['u'])**incr_n * self.ref.S0
        O = ((S - self.K) * self.signCP ).max(0)
        S_tree = O_tree = None

        if getattr(self.px_spec, 'keep_hist', False):
            S_tree, O_tree = (tuple(S),), (tuple(O),)

            for i in range(n, 0, -1):
                O = (O[:i] * (1 - _['p']) + O[1:] * (_['p'])) * _['df_dt']  # prior option prices (@time step=i-1)
                S = S[1:i+1] * _['d']                   # prior stock prices (@time step=i-1)
                S_tree, O_tree = (tuple(S),) + S_tree, (tuple(O),) + O_tree
            out = O_tree[0][0]
        else:
            csl = (0.,) + Vec(Util.cumsum(Util.log(Util.arange(1, n + 1))))         # logs avoid overflow & truncation
            tmp = Vec(csl[n]) - csl - tuple(reversed(csl)) + incr_n * math.log(_['p']) + decr_n * math.log(1 - _['p'])
            out = (sum(tmp.exp * _['df_T'] * tuple(O)))

        self.px_spec.add(px=float(out), sub_method='binary tree; Hull p.135', LT_specs=_, ref_tree=S_tree, opt_tree=O_tree)
        return self

    def _calc_MC(self):
        """ Internal function for option valuation. See ``calc_px()`` for complete documentation.

        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """

        n = getattr(self.px_spec, 'nsteps', 3)
        m = getattr(self.px_spec, 'npaths', 3)
        Seed = getattr(self.px_spec, 'rng_seed', None)

        dt, df = self.LT_specs(n)['dt'], self.LT_specs(n)['df_dt']
        S0, vol = self.ref.S0, self.ref.vol
        K, r, signCP = self.K, self.rf_r, self._signCP

        np.random.seed(Seed)
        norm_mtx = np.random.normal((r - 0.5 * vol ** 2) * dt, vol * math.sqrt(dt), (n + 1, m))
        S = S0 * np.exp(np.cumsum(norm_mtx, axis=0))
        S[0] = S0
        payout = np.maximum(signCP * (S - K), 0)
        v = np.copy(payout)  # terminal payouts

        for i in range(n - 1, -1, -1):  v[i] = v[i + 1] * df  # discount to present, step by step (can also do at once)

        v0 = np.mean(v[0])
        self.px_spec.add(px=v0, submethod='Least Squares Monte Carlo (LSM)')

        return self

    def _calc_FD(self):
        """ Internal function for option valuation. See ``calc_px()`` for complete documentation.

        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """
        return self

