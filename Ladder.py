try:
    from qfrm.OptionValuation import *  # production:  if qfrm package is installed
except:
    from OptionValuation import *  # development: if not installed and running from source


class Ladder(OptionValuation):
    """
    Ladder option class.
    """

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
                LT, MC, FD methods require number of times steps
        npaths : int
                MC, FD methods require number of simulation paths
        keep_hist : bool
                If True, historical information (trees, simulations, grid) are saved in self.px_spec object.

        Returns
        ------------
        self : Ladder
            Returned object contains specifications and calculated price in embedded ``PriceSpec`` object.


        Notes
        ----------
        An option that locks-in gains once the underlying reaches predetermined price levels or "rungs," guaranteeing
        some profit even if the underlying security falls back below these levels before the option expires. [1]


        References
        -------------
        [1] `Ladder Option on Investopedia <http://www.investopedia.com/terms/l/ladderoption.asp>`_

        Examples
        ------------

        FD Examples
        --------------


        :Authors:
            Patrick Granahan
        """

        return super().calc_px(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)

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

        Returns
        -------
        self: Ladder
        """

        # The number of intervals to divide T into. N + 1 times/steps will be considered
        N = getattr(self.px_spec, 'nsteps')

        # Used to divide S_max into deltas. M + 1 stock prices/paths will be considered
        M = getattr(self.px_spec, 'npaths')

        S_max = choose_S_max(M, self.ref.S0)

        S_min = 0.0  # Minimum stock price
        d_t = self.T / (N - 1)  # Time step
        S_vec = np.linspace(S_min, S_max, M)  # Initialize the possible stock price vector
        t_vec = np.linspace(0, self.T, N)  # Initialize the time vector

        f_px = np.zeros((M, N))  # Initialize the matrix. Hull's P482

        M = M - 1
        N = N - 1

        # Set boundary conditions.
        f_px[:, -1] = S_vec

        if self.right == 'call':
            # Payout at the maturity time
            init_cond = np.maximum((S_vec - self.K), 0) * (S_vec >= self.K2)
            # Boundary condition
            upper_bound = 0
            # Calculate the current value
            lower_bound = np.maximum((S_vec[-1] - self.K), 0) * (S_vec[-1] >= self.K2) * np.exp(
                -self.rf_r * (self.T - t_vec))
        elif self.right == 'put':
            # Payout at the maturity time
            init_cond = np.maximum((self.K - S_vec), 0) * (S_vec <= self.K2)
            # Boundary condition
            upper_bound = np.maximum((self.K - S_vec[0]), 0) * (S_vec[0] <= self.K2) * np.exp(
                -self.rf_r * (self.T - t_vec))
            # Calculate the current value
            lower_bound = 0

        # Generate Matrix B in http://www.goddardconsulting.ca/option-pricing-finite-diff-implicit.html
        j_list = np.arange(0, M + 1)
        a_list = 0.5 * d_t * ((self.rf_r - self.ref.q) * j_list - self.ref.vol ** 2 * j_list ** 2)
        b_list = 1 + d_t * (self.ref.vol ** 2 * j_list ** 2 + self.rf_r)
        c_list = 0.5 * d_t * (-(self.rf_r - self.ref.q) * j_list - self.ref.vol ** 2 * j_list ** 2)

        data = (a_list[2:M], b_list[1:M], c_list[1:M - 1])
        B = sparse.diags(data, [-1, 0, 1]).tocsc()

        # Using Implicit method to solve B-S equation
        f_px[:, N] = init_cond
        f_px[0, :] = upper_bound
        f_px[M, :] = lower_bound
        Offset = np.zeros(M - 1)
        for idx in np.arange(N - 1, -1, -1):
            Offset[0] = -a_list[1] * f_px[0, idx]
            Offset[-1] = -c_list[M - 1] * f_px[M, idx]
            f_px[1:M, idx] = sparse.linalg.spsolve(B, f_px[1:M, idx + 1] + Offset)
            f_px[:, -1] = init_cond
            f_px[0, :] = upper_bound
            f_px[-1, :] = lower_bound

        self.px_spec.add(px=float(np.interp(self.ref.S0, S_vec, f_px[:, 0])), sub_method='Implicit Method')

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

        d_S = S0 / (math.floor(M/2))
        S_max = d_S * M

        return S_max, d_S