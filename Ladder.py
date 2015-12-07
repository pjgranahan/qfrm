try:
    from qfrm.OptionValuation import *  # production:  if qfrm package is installed
except:
    from OptionValuation import *  # development: if not installed and running from source


class Ladder(OptionValuation):
    """
    Ladder option class.

    An option that locks-in gains once the underlying reaches predetermined price levels or "rungs," guaranteeing
    some profit even if the underlying security falls back below these levels before the option expires. [1]

    References
    -------------
    [1] `Ladder Option on Investopedia <http://www.investopedia.com/terms/l/ladderoption.asp>`_

    """

    def __init__(self, rungs, *args, **kwargs):
        """
        Parameters
        ----------
        rungs : tuple
                Required. The predetermined profit lock-in price levels.
                Example: (51, 52, 53, 54, 55)
        """
        super().__init__(*args, **kwargs)
        self.rungs = sorted(rungs, reverse=self.signCP == -1)  # ascending if option right is call, descending if put

    def calc_px(self, method='BS', nsteps=None, npaths=None, keep_hist=False):
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
                ``LT``: Lattice tree (such as Ladder tree)
                ``MC``: Monte Carlo simulation methods
                ``FD``: finite differencing methods
        nsteps : int
                LT, MC, FD methods require number of times steps. Must be >2.
        npaths : int
                MC, FD methods require number of simulation paths. Must be >2.
        keep_hist : bool
                If True, historical information (trees, simulations, grid) are saved in self.px_spec object.

        Returns
        ------------
        self : Ladder
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object)

        Examples
        ------------

        **FD Examples**

        Example #1

        >>> s = Stock(S0=50, vol=0.20, q=0.03)
        >>> o = Ladder(rungs=(51, 52, 53, 54, 55), ref=s, right='call', K=51, T=1, rf_r=0.05)
        >>> o.pxFD(npaths = 25, nsteps=10, keep_hist=True)  # npaths > 10 so that the plot is pretty
        3.6497147019999998

        Example #2 (plot)
        Shows the finite difference grid that is produced in Example #1

        >>> plt.matshow(o.px_spec.grid); plt.show()  # doctest: +ELLIPSIS
        <...>

        Example #3 (verifiable)
        As the number of rungs in a Ladder option approaches infinity, it becomes effectively the same as a Lookback.
        So, we can use a Lookback to validate the price of a Ladder option with many rungs.

        >>> s = Stock(S0=50, vol=.4, q=.0)
        >>> o = Ladder(rungs=([px for px in range(50, -1, -1)]), ref=s, right='put', K=50, T=0.25, rf_r=.1,
        ... desc='Example from Hull Ch.26 Example 26.2 (p608)')
        >>> actual = o.pxFD(npaths = 6, nsteps=10); expected = 8.067753794
        >>> (abs(actual - expected) / expected) < 0.10  # Verify within 10% of expected
        True

        :Authors:
            Patrick Granahan
        """

        return super().calc_px(rungs=self.rungs, method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)

    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Ladder
        """
        return self

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Ladder
        """
        return self

    def _calc_MC(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Ladder
        """
        return self

    def _calc_FD(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for full documentation.

        WARNING: Varying npaths or nsteps can produce dramatically different results.
        Therefore, results are unstable and probably unsuitable without further code refinements.

        Returns
        -------
        self: Ladder

        :Authors:
            Patrick Granahan
        """

        # The number of intervals to divide T into. N + 1 times/steps will be considered
        N = getattr(self.px_spec, 'nsteps')

        # Used to divide S_max into deltas. M + 1 stock prices/paths will be considered
        M = getattr(self.px_spec, 'npaths')

        # Create the grid/matrix
        grid = np.zeros(shape=(N + 1, M + 1))

        # Define stock price parameters
        S_max, d_S = Ladder._choose_S_max(M, self.ref.S0)  # Maximum stock price, stock price change interval
        S_min = 0.0  # Minimum stock price
        S_vec = np.arange(S_min, S_max + d_S,
                          d_S)  # Possible stock price vector. (+d_S to S_max so that S_max is included)

        # Define time parameters
        d_T = self.T / N  # Time step
        d_T_vec = np.arange(0, self.T + d_T, d_T)  # Delta time vector. (+d_T to T so that T is included)
        discount_vec = np.exp(-self.rf_r * (self.T - d_T_vec))  # Discount vector

        # Fill the matrix boundary at maturity
        grid[N, :] = [self.payoff((stock_price,)) for stock_price in S_vec]

        # Fill the matrix boundary when the stock price is S_min
        grid[:, 0] = discount_vec * self.payoff((S_min,))

        # Fill the matrix boundary when the stock price is S_max
        grid[:, M] = discount_vec * self.payoff((S_max,))

        # Explicit finite difference equations
        def a(j):
            discount = (1 / (1 + (self.rf_r * d_T)))
            return discount * (.5 * d_T * ((self.ref.vol ** 2 * j ** 2) - ((self.rf_r - self.ref.q) * j)))

        def b(j):
            discount = (1 / (1 + (self.rf_r * d_T)))
            return discount * (1 - (d_T * self.ref.vol ** 2 * j ** 2))

        def c(j):
            discount = (1 / (1 + (self.rf_r * d_T)))
            return discount * (.5 * d_T * ((self.ref.vol ** 2 * j ** 2) + ((self.rf_r - self.ref.q) * j)))

        # Fill out the finite difference grid, by stepping backwards through the y axis and solving 1 row at a time
        for i in range(N - 1, -1, -1):
            for k in range(1, M):
                j = M - k
                value = a(j) * grid[i + 1, k + 1] + b(j) * grid[i + 1, k] + c(j) * grid[i + 1, k - 1]
                if value == float("inf") or value == float("-inf") or math.isnan(value):
                    raise Exception(
                        "(-)Infinity or NaN found while attempting to fill the finite difference grid. "
                        "Maybe your inputs were too small.")
                grid[i, k] = value

        # Record the history if requested
        if getattr(self.px_spec, 'keep_hist'):
            self.px_spec.add(grid=grid)

        # Record the price
        self.px_spec.add(px=grid[0, M - (self.ref.S0 / d_S)], method='FDM', sub_method='Explicit')

        return self

    @staticmethod
    def _choose_S_max(M, S0):
        """
        Chooses an S_max. Also produces a d_S that satisfies the requirements of S_max.
        S_max is "a stock price sufficiently high that, when it is reached, the option has virtually no value." [1]
        S_max is chosen such that one of the multiples of d_S is the current stock price

        Parameters
        ----------
        M : int
            The number of intervals to divide S_max into.
        S0 : float
            The current stock price.

        References
        ----------
        [1] Hull 21.8 (p. 478)

        Returns
        -------
        d_S : float
        S_max : float

        Examples
        --------
        >>> S0 = 100; M = 10
        >>> Ladder._choose_S_max(M, S0)
        (200.0, 20.0)

        >>> S0 = 100; M = 7
        >>> Ladder._choose_S_max(M, S0)
        (233.33333333333334, 33.333333333333336)

        >>> S0 = 333; M = 10
        >>> Ladder._choose_S_max(M, S0)
        (666.0, 66.6)

        >>> S0 = 50; M = 123
        >>> Ladder._choose_S_max(M, S0)
        (100.81967213114754, 0.819672131147541)
        """

        d_S = S0 / (math.floor(M / 2))
        S_max = d_S * M

        return S_max, d_S

    def payoff(self, price_history):
        """
        Calculates the payoff of a Ladder option given a price history.

        Parameters
        ----------
        price_history : tuple
            The history of prices for this option.

        Returns
        -------
        payoff : float
            The payoff for this option.

        Examples
        --------
        >>> s = Stock(S0=50)
        >>> o = Ladder(rungs=(51, 52, 53, 54, 55), ref=s, right='call', K=51)
        >>> o.payoff((50, 50.5, 52, 49, 37, 52.5, 0))
        1

        >>> s = Stock(S0=50)
        >>> o = Ladder(rungs=(51, 52, 55, 54, 53), ref=s, right='call', K=53)
        >>> o.payoff((50, 50.5, 52, 49, 37, 52.5, 0))
        0

        >>> s = Stock(S0=50)
        >>> o = Ladder(rungs=(50, 48, 47, 42, 40.5), ref=s, right='put', K=45)
        >>> o.payoff((50, 55, 45, 60))
        0

        >>> s = Stock(S0=50)
        >>> o = Ladder(rungs=(50, 48, 47, 42, 40.5), ref=s, right='put', K=45)
        >>> o.payoff((50, 55, 45, 60, 41.9))
        3
        """

        # Find the extreme price in time for this option. Max for a call, min for a put
        if self.signCP == 1:
            extreme_historical_price = max(price_history)
        elif self.signCP == -1:
            extreme_historical_price = min(price_history)
        else:
            raise Exception("Unrecognized right for a Ladder option.")

        # The base case has the extreme stock price never reaching a rung
        rung_reached = -1
        # Climb the ladder, rung by rung, until the extreme stock price can't reach the next rung
        # (Note that each rung step could represent an increase OR decrease in strike, depending on the option right)
        for i in range(len(self.rungs)):
            if self.signCP * extreme_historical_price >= self.signCP * self.rungs[rung_reached + 1]:
                rung_reached += 1
            else:
                break

        payoff = max(self.signCP * (self.rungs[rung_reached] - self.K), 0)
        return payoff
