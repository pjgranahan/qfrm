import numpy as np
try:    from qfrm.European import *  # production:  if qfrm package is installed
except: from European import *  # development: if not installed and running from source


class Ladder(European):
    """ `Ladder <http://www.investopedia.com/terms/l/ladderoption.asp>`_ exotic option class.
    """

    def calc_px(self, rungs, **kwargs):
        """ Wrapper function that calls appropriate valuation method.

        Parameters
        ----------
        rungs : tuple
                Required. The predetermined profit lock-in price levels.
        kwargs : dict
            Keyword arguments (``method``, ``nsteps``, ``npaths``, ``keep_hist``, ``rng_seed``, ...)
            are passed to the parent. See ``European.calc_px()`` for details.


        Returns
        ------------
        self : Ladder
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object).


        Notes
        ----------
        An option that locks-in gains once the underlying reaches predetermined price levels or "rungs," guaranteeing
        some profit even if the underlying security falls back below these levels before the option expires. [1]

        WARNING: Varying ``npaths`` or ``nsteps`` can produce dramatically different results.


        Examples
        ---------

        **FD**

        Example #1

        >>> s = Stock(S0=50, vol=0.20, q=0.03)
        >>> o = Ladder(ref=s, right='call', K=51, T=1, rf_r=0.05)
        >>> o.pxFD(rungs=(51, 52, 53, 54, 55), npaths = 25, nsteps=10, keep_hist=True)  # npaths > 10 so that the plot is pretty
        3.649714702

        Example #2 (plot)
        Shows the finite difference grid that is produced in Example #1

        >>> plt.matshow(o.px_spec.grid);  # doctest: +ELLIPSIS
        <...>

        Example #3 (verifiable)
        As the number of rungs in a Ladder option approaches infinity, it becomes effectively the same as a Lookback.
        So, we can use a Lookback to validate the price of a Ladder option with many rungs.

        >>> s = Stock(S0=50, vol=.4, q=.0)
        >>> o = Ladder(ref=s, right='put', K=50, T=0.25, rf_r=.1, desc='Example from Hull Ch.26 Example 26.2 (p608)')
        >>> actual = o.pxFD(rungs=([px for px in range(50, -1, -1)]), npaths = 6, nsteps=10); expected = 8.067753794
        >>> (abs(actual - expected) / expected) < 0.10  # Verify within 10% of expected
        True

        Example 3

        :Authors:
            Patrick Granahan
        """
        self.save2px_spec(rungs=rungs, **kwargs)
        return getattr(self, '_calc_' + self.px_spec.method.upper())()

    def _calc_BS(self):
        """ Internal function for option valuation.        """
        return self

    def _calc_LT(self):
        """ Internal function for option valuation.        """
        return self

    def _calc_MC(self):
        """ Internal function for option valuation.        """
        return self

    def _calc_FD(self):
        """ Internal function for option valuation.        See ``calc_px()`` for full documentation.

        :Authors:
            Patrick Granahan
        """
        _ = self.px_spec;   n, m, rungs, keep_hist = _.nsteps, _.npaths, _.rungs, _.keep_hist
        _ = self.ref;       S0, vol, q = _.S0, _.vol, _.q
        _ = self;           T, K, rf_r, net_r, sCP = _.T, _.K, _.rf_r, _.net_r, _.signCP

        grid = np.zeros(shape=(n + 1, m + 1))   # Create the grid/matrix

        # Define stock price parameters
        S_max, d_S = Ladder._choose_S_max(m, S0)  # Maximum stock price, stock price change interval
        S_min = 0.0  # Minimum stock price
        S_vec = np.arange(S_min, S_max + d_S, d_S)  # Possible stock price vector. (+d_S to S_max so that S_max is included)

        # Define time parameters
        d_T = T / n  # Time step
        d_T_vec = np.arange(0, T + d_T, d_T)  # Delta time vector. (+d_T to T so that T is included)
        discount_vec = np.exp(-rf_r * (T - d_T_vec))  # Discount vector

        # Fill the matrix boundary at maturity
        grid[n, :] = [self.payoff((stock_price,)) for stock_price in S_vec]

        # Fill the matrix boundary when the stock price is S_min
        grid[:, 0] = discount_vec * self.payoff((S_min,))

        # Fill the matrix boundary when the stock price is S_max
        grid[:, m] = discount_vec * self.payoff((S_max,))

        # Explicit finite difference equations
        def a(j):
            discount = (1 / (1 + (rf_r * d_T)))
            return discount * (.5 * d_T * ((vol ** 2 * j ** 2) - (net_r * j)))

        def b(j):
            discount = (1 / (1 + (rf_r * d_T)))
            return discount * (1 - (d_T * vol ** 2 * j ** 2))

        def c(j):
            discount = (1 / (1 + (rf_r * d_T)))
            return discount * (.5 * d_T * ((vol ** 2 * j ** 2) + (net_r * j)))

        # Fill out the finite difference grid, by stepping backwards through the y axis and solving 1 row at a time
        for i in range(n - 1, -1, -1):
            for k in range(1, m):
                j = m - k
                value = a(j) * grid[i + 1, k + 1] + b(j) * grid[i + 1, k] + c(j) * grid[i + 1, k - 1]
                if value == float("inf") or value == float("-inf") or math.isnan(value):
                    raise Exception(
                        "(-)Infinity or NaN found while attempting to fill the finite difference grid. "
                        "Maybe your inputs were too small.")
                grid[i, k] = value

        if keep_hist:    self.px_spec.add(grid=grid)  # Record the history if requested

        self.px_spec.add(px=float(grid[0, m - (S0 / d_S)]), sub_method='Explicit FDM') # save price

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
        >>> o = Ladder(ref=s, right='call', K=51)
        >>> o.px_spec.rungs = rungs=(51, 52, 53, 54, 55)
        >>> o.payoff((50, 50.5, 52, 49, 37, 52.5, 0))
        1

        >>> s = Stock(S0=50)
        >>> o = Ladder(ref=s, right='call', K=53)
        >>> o.px_spec.rungs = rungs=(51, 52, 55, 54, 53)
        >>> o.payoff((50, 50.5, 52, 49, 37, 52.5, 0))
        0

        >>> s = Stock(S0=50)
        >>> o = Ladder(ref=s, right='put', K=45)
        >>> o.px_spec.rungs=(50, 48, 47, 42, 40.5)
        >>> o.payoff((50, 55, 45, 60))
        0

        >>> s = Stock(S0=50)
        >>> o = Ladder(ref=s, right='put', K=45)
        >>> o.px_spec.rungs = rungs=(50, 48, 47, 42, 40.5)
        >>> o.payoff((50, 55, 45, 60, 41.9))
        3
        """
        rungs = self.px_spec.rungs

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
        for i in range(len(rungs)):
            if self.signCP * extreme_historical_price >= self.signCP * rungs[rung_reached + 1]:
                rung_reached += 1
            else:
                break

        payoff = max(self.signCP * (rungs[rung_reached] - self.K), 0)
        return payoff
