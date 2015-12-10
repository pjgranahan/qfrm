import math

try: from qfrm.OptionValuation import *  # production:  if qfrm package is installed
except:   from OptionValuation import *  # development: if not installed and running from source


class European(OptionValuation):
    """ Financial option derivative of `American <https://en.wikipedia.org/wiki/Option_style>`_ style."""

    def calc_px(self, **kwargs):
        """ Wrapper function that calls appropriate valuation method.

        All parameters of ``calc_px`` are saved to local ``px_spec`` variable of class ``PriceSpec`` before
        specific pricing method (``_calc_BS()``,...) is called.
        An alternative to price calculation method ``.calc_px(method='BS',...).px_spec.px``
        is calculating price via a shorter method wrapper ``.pxBS(...)``.
        The same works for all methods (BS, LT, MC, FD).


        Parameters
        -------------
        method : {'BS', 'LT', 'MC', 'FD'}
            Specifies option valuation method:

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

        - The Black-Scholes-Merton Model (Ch.15), `OFOD <http://www-2.rotman.utoronto.ca/~hull/ofod/index.html>`_, J.C.Hull, 9ed, 2014, p.346
        - Generalized Tree Building Procedure, `Technical Note #9, J.C.Hull <http://www-2.rotman.utoronto.ca/~hull/technicalnotes/TechnicalNote9.pdf>`_

        **Lattice Tree (LT)**, i.e. binomial or binary (recombining) tree pricing.
        Binomial tree is used to (discretely) grow the underlying stock price.
        Then terminal option payoffs are discounted to the presented,
        while averaged at each prior node (as part of computing discounted expected value of an option).

        *References:*
        Binomial Trees (Ch.13), `OFOD <http://www-2.rotman.utoronto.ca/~hull/ofod/index.html>`_, J.C.Hull, 9ed, 2014, p.274

        **Monte Carlo simulation (MC)**.
        First, simulate stock prices, according to Geometric Brownian motion (GBM) model.
        Then compute terminal option payoffs and discount them back to present.
        Averaged present value is the desired option price.

        *References:*
        Monte Carlo Simulation (Ch.21-6), `OFOD <http://www-2.rotman.utoronto.ca/~hull/ofod/index.html>`_, J.C.Hull, 9ed, 2014, pp.469-475


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

        >>> (o.px_spec.method, o.px_spec.px, o.px_spec.BS_specs['d1'], o.px_spec.BS_specs['d2'])  # doctest: +ELLIPSIS
        ('BS', 0.8085993729000904, 0.7692626281060315, 0.627841271868722)

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

        **Compare:**

        The following compares all available pricing methods for an American option.
        MC method appears off, but try larger simulations, ``nsteps=10000`` and ``npaths=10000``.
        Calculation may take a 1-3 minutes.

        >>> s = Stock(S0=42, vol=.20)
        >>> o = European(ref=s, right='put', K=40, T=.5, rf_r=.1, desc='call @0.81, put @4.76, Hull p.339')
        >>> (o.pxBS(), o.pxLT(nsteps=100), o.pxMC(nsteps=100, npaths=1000, rng_seed=0))
        (0.808599373, 0.810995338, 0.73588733399999995)

        Next, we visually compare the convergence performance of 3 methods.
        Notice the scale on counters ``nsteps`` and ``npaths``.

        >>> dBS = [o.pxBS() for i in range(20)]
        >>> dLT = [o.pxLT(nsteps=i) for i in range(20)]
        >>> dMC = [o.pxMC(nsteps=100, npaths=500*i, rng_seed=0) for i in range(20)]
        >>> from pandas import DataFrame
        >>> d = DataFrame({'BS': dBS, 'LT': dLT, 'MC': dMC})
        >>> d.plot(grid=1, title='Price convergence (vs. scaled iterations) for ' + o.specs)  # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot...>

        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """

        self.save2px_spec(**kwargs)
        return getattr(self, '_calc_' + self.px_spec.method.upper())()

    def save2px_spec(self, method='BS', nsteps=None, npaths=None, keep_hist=False, rng_seed=None, **kwargs):

        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist, rng_seed=rng_seed, **kwargs)
        assert getattr(self, 'ref') is not None, 'Ooops. Please supply referenced (underlying) asset, `ref`'
        assert getattr(self, 'rf_r') is not None, 'Ooops. Please supply risk free rate `rf_r`'
        assert getattr(self, 'K') is not None, 'Ooops. Please supply strike `K`'
        assert getattr(self, 'T') is not None, 'Ooops. Please supply time to expiry (in years) `T`'
        # assert getattr(self, '_signCP') is not None, 'Ooops. Please supply option right: call, put, ...'  # VarianceSwap

        if method in ('LT', 'MC', 'FD'):
            self.px_spec.add_verify(nsteps=nsteps, dtype=int, min=1, max=float("inf"), dflt=3)

        if method in ('MC', 'FD'):
            self.px_spec.add_verify(npaths=npaths, dtype=int, min=1, max=float("inf"), dflt=3)

        if method in ('MC'):
            self.px_spec.add_verify(rng_seed=rng_seed, dtype=int, min=0, max=float("inf"), dflt=None)

    def _calc_BS(self):
        """ Internal function for option valuation.        See ``calc_px()`` for complete documentation.

        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """

        if not self.style == 'European': return self   # if (exotic) sub-class inherits this method, don't calculate

        _ = self
        sp = self._BS_specs();
        d1, d2, Nd1, Nd2, N_d1, N_d2 = sp['d1'], sp['d2'], sp['Nd1'], sp['Nd2'], sp['N_d1'], sp['N_d2']

        # if calc of both prices is cheap, do both and include them into Price object.
        # Price.px should always point to the price of interest to the user
        # Save values as basic data types (int, floats, str), instead of np.array
        px_call = float(_.ref.S0 * math.exp(-_.ref.q * _.T) * Nd1 - _.K * math.exp(-_.rf_r * _.T) * Nd2)
        px_put = float(- _.ref.S0 * math.exp(-_.ref.q * _.T) * N_d1 + _.K * math.exp(-_.rf_r * _.T) * N_d2)
        px = px_call if _.signCP == 1 else px_put if _.signCP == -1 else None

        self.px_spec.add(px=px, sub_method='standard; Hull p.335', px_call=px_call, px_put=px_put)
        return self

    def _calc_LT(self):
        """ Internal function for option valuation.        See ``calc_px()`` for complete documentation.

        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """

        if not self.style == 'European': return self   # if (exotic) sub-class inherits this method, don't calculate

        _ = self._LT_specs()
        n = self.px_spec.nsteps
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

        if not self.style == 'European': return self   # if (exotic) sub-class inherits this method, don't calculate

        n = getattr(self.px_spec, 'nsteps', 3)
        m = getattr(self.px_spec, 'npaths', 3)
        Seed = getattr(self.px_spec, 'rng_seed', None)

        ltsp = self._LT_specs()
        dt, df = ltsp['dt'], ltsp['df_dt']
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
        if not self.style == 'European': return self   # if (exotic) sub-class inherits this method, don't calculate

        return self


    def _BS_specs(self):
        """ Calculates a collection of specs/parameters needed for Black-Scholes pricing.

        _BS_specs is a private method.
        It is used by other methods after the following are gathered:
        d1, d2, N(d1), N(d2), N(-d1), N(-d2), where N is standard normal distribution

        Returns
        -------
        dict
            A dictionary of calculated parameters.


        Examples
        --------

        Normally, user will not access these parameters directly.

        >>> from pprint import pprint   # helps printing ordered dictionaries (for doctesting)
        >>> s = Stock(S0=42, vol=.2)
        >>> o = European(ref=s, right='call', K=40, T=.5, rf_r=.1, desc={'See':'OFOD, J.C.Hull, 2014, p.338'})
        >>> pprint(o._BS_specs())      # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        {'N_d1': 0.22086870905733108, 'N_d2': 0.26505396315409147, 'Nd1': 0.7791312909426689,
         'Nd2': 0.7349460368459085, 'd1': 0.7692626281060315, 'd2': 0.627841271868722}

         """

        _ = self
        d1 = (math.log(_.ref.S0 / _.K) + (_.rf_r + _.ref.vol ** 2 / 2.) * _.T)/(_.ref.vol * math.sqrt(_.T))
        d2 = d1 - _.ref.vol * math.sqrt(_.T)
        N = Util.norm_cdf

        sp = {'d1': d1, 'd2': d2, 'Nd1':N(d1), 'Nd2':N(d2), 'N_d1':N(-d1), 'N_d2':N(-d2)}

        self.px_spec.add(BS_specs=sp)
        return sp


    def _LT_specs(self):
        """ Calculates a collection of specs/parameters needed for lattice tree pricing.

        _LT_specs is a private method.
        It is used by other methods after the following are gathered:
        nsteps, T, vol, risk free rate and net risk free rate.

        Calculated parameters:
            dt: time interval between consequtive two time steps
            u: Stock price up move factor
            d: Stock price down move factor
            a: growth factor. See OFOD, J.C.Hull, 9ed, 2014, p.452
            p: probability of up move over one time interval dt
            df_T: discount factor over full time interval dt, i.e. per life of an option
            df_dt: discount factor over one time interval dt, i.e. per step


        Returns
        -------
        dict
            A dictionary of calculated parameters.

        Examples
        --------

        Normally, user will not access these parameters directly.

        # >>> o = European(ref=Stock(S0=42, vol=.2), right='call', K=40, T=.5, rf_r=.1)
        # >>> o.px_spec.add(nsteps=2)   # required for calculation of LT specs                #    doctest: +SKIP
        # >>> o.px_spec.LT_specs
        # {'a': 1.025315120...'d': 0.904837418...'df_T': 0.951229424...
        #  'df_dt': 0.975309912...'dt': 0.25, 'p': 0.601385701...'u': 1.105170918...}

        # >>> s = Stock(S0=50, vol=.3)
        # >>> o = European(ref=s,right='put', K=52, T=2, rf_r=.05, desc={'See':'See OFOD, J.C.Hull, 2014, p.288'})
        # >>> o.px_spec.add(nsteps=3)   # required for calculation of LT specs    #    doctest: +SKIP
        # >>> o._LT_specs()      # doctest: +ELLIPSIS
        # {'a': 1.033895113...'d': 0.782744477...'df_T': 0.904837418...
        #  'df_dt': 0.967216100...'dt': 0.666...'p': 0.507568158...'u': 1.277556123...}

         """

        n = self.px_spec.nsteps     # number of steps in a tree
        T = self.T                  # time to maturity (in years)
        r = self.rf_r               # risk free rate
        nr = self.net_r     # net risk free rate (after dividend rate and foreign risk free rate are deducted)
        vol = self.ref.vol  # volatility of underlying

        sp = {'dt': T / n}
        sp['u'] = math.exp(vol * math.sqrt(sp['dt']))
        sp['d'] = 1 / sp['u']
        sp['a'] = math.exp(nr * sp['dt'])      # growth factor, p.452
        sp['p'] = (sp['a'] - sp['d']) / (sp['u'] - sp['d'])
        sp['df_T'] = math.exp(-r * T)
        sp['df_dt'] = math.exp(-r * sp['dt'])

        self.px_spec.add(LT_specs=sp)  # save calculated parameters for later access and display
        return sp


    def pxBS(self, **kwargs):
        """ Calls exotic pricing method `calc_px()`

        This property calls `calc_px()` method which should be overloaded
        by each exotic option class (inheriting OptionValuation)

        Parameters
        ----------
        kwargs
            Pricing parameters required to price this exotic option. See `calc_px()` for specifics and examples.

        Returns
        -------
        float
            price of the exotic option

        Examples
        --------
        >>> from qfrm import *
        >>> European(ref=Stock(S0=50, vol=.2), rf_r=.05, K=50, T=0.5, right='call').pxBS()
        3.444364289

        """
        return self.print_value(self.calc_px(method='BS', **kwargs).px_spec.px)

    def pxLT(self, **kwargs):
        """ Calls exotic pricing method `calc_px()`

        This property calls `calc_px()` method which should be overloaded
        by each exotic option class (inheriting OptionValuation)

        Parameters
        ----------
        kwargs
            Pricing parameters required to price this exotic option. See `calc_px()` for specifics and examples.

        Returns
        -------
        float
            price of the exotic option

        Examples
        --------
        >>> from qfrm import *
        >>> European(ref=Stock(S0=50, vol=.2), rf_r=.05, K=50, T=0.5, right='call').pxLT()
        3.669370702

        """
        return self.print_value(self.calc_px(method='LT', **kwargs).px_spec.px)

    def pxMC(self, **kwargs):
        """ Calls exotic pricing method `calc_px()`

        This property calls `calc_px()` method which should be overloaded
        by each exotic option class (inheriting OptionValuation)

        Parameters
        ----------
        kwargs
            Pricing parameters required to price this exotic option. See `calc_px()` for specifics and examples.

        Returns
        -------
        float
            price of the exotic option

        Examples
        --------
        >>> from qfrm import *
        >>> s = Stock(S0=50, vol=.2)
        >>> European(ref=s, rf_r=.05, K=50, T=0.5, right='call').pxMC(rng_seed=0, nsteps=10, npaths=10)
        4.1467906460000004
        """
        return self.print_value(self.calc_px(method='MC', **kwargs).px_spec.px)

    def pxFD(self, **kwargs):
        """ Calls exotic pricing method ``calc_px()``

        This property calls ``calc_px()`` method which should be overloaded
        by each exotic option class (inheriting OptionValuation)
        """
        return self.print_value(self.calc_px(method='FD', **kwargs).px_spec.px)

