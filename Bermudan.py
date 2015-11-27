from qfrm import *
from numpy import arange, maximum


class Bermudan(OptionValuation):
    """ Bermudan option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, method='LT', tex=(.12,.24,.46,.9,.91,.92,.93,.94,.95,.96,.97,.98,.99, 1.),
        nsteps=None, npaths=None, keep_hist=False, R=3):
        """ Wrapper function that calls appropriate valuation method.

        User passes parameters to calc_px, which saves them to local PriceSpec object
        and calls specific pricing function (_calc_BS,...).
        This makes significantly less docstrings to write, since user is not interfacing pricing functions,
        but a wrapper function calc_px().

        Parameters
        ----------
        method : str
                Required. Indicates a valuation method to be used: 'BS', 'LT', 'MC', 'FD'
        tex : list
                Required. Must be a vector (tuple; list; array, ...) of times to exercisability. 
                For Bermudan, assume that exercisability is for discrete tex times only.
                This also needs to be sorted ascending and the final value is the corresponding vanilla maturity.
                If T is not equal the the final value of T, then
                    the T will take precedence: if T < max(tex) then tex will be truncated to tex[tex < T] and will be 
                    appended to tex.
                    If T > max(tex) then the largest value of tex will be replaced with T.
        nsteps : int
                MC, FD methods require number of times steps. 
                Optional if using LT: n_steps = <integer> * <length of tex>. Will fill in the spaces between steps 
                implied by tex. 
                Useful if tex is regular or sparse to improve accuracy. Otherwise leave as None.
        npaths : int
                MC, FD methods require number of simulation paths
        keep_hist : bool
                If True, historical information (trees, simulations, grid) are saved in self.px_spec object.
        R : int
                Number of basis functions. Used to generate weighted Laguerre polynomial values.
                Used in MC method. Must be between 0 and 6.

        Returns
        -------
        self : Bermudan

        .. sectionauthor:: Oleg Melkinov; Andy Liao, Patrick Granahan


        Notes
        -----
        The Bermudan option is a modified American with restricted early-exercise dates. Due to this restriction, 
        Bermudans are named as such as they are "between" American and European options in exercisability, and as 
        this module demonstrates, in price.

        For our Monte Carlo pricing, we use the Longstaff-Schwartz algorithm. Our implementation is drawn heavily from
        this paper [1], while our method for generating the betas is drawn heavily from this paper [2].


        References
        ----------
        [1] http://eprints.maths.ox.ac.uk/789/1/Thom.pdf
        [2] http://eprints.maths.ox.ac.uk/934/1/longyun_chen.pdf

        Examples
        --------

        >>> #LT pricing of Bermudan options
        >>> s = Stock(S0=50, vol=.3)
        >>> o = Bermudan(ref=s, right='put', K=52, T=2, rf_r=.05)
        >>> o.calc_px(method='LT', keep_hist=True).px_spec.px
        7.251410363950508
        >>> ##Changing the maturity
        >>> o = Bermudan(ref=s, right='put', K=52, T=1, rf_r=.05)
        >>> o.calc_px(method='LT', keep_hist=True).px_spec.px
        5.9168269657242
        >>> o = Bermudan(ref=s, right='put', K=52, T=.5, rf_r=.05)
        >>> o.calc_px(method='LT', keep_hist=True).px_spec.px        
        4.705110748543638
        >>> ##Explicit input of exercise schedule
        >>> ##Explicit input of exercise schedule
        >>> from numpy.random import normal, seed
        >>> seed(12345678)
        >>> rlist = normal(1,1,20)
        >>> times = tuple(map(lambda i: float(str(round(abs(rlist[i]),2))), range(20)))
        >>> o = Bermudan(ref=s, right='put', K=52, T=1., rf_r=.05)
        >>> o.calc_px(method='LT', tex=times, keep_hist=True).px_spec.px
        5.8246496768398055
        >>> ##Example from p. 9 of http://janroman.dhis.org/stud/I2012/Bermuda/reportFinal.pdf
        >>> times = (3/12,6/12,9/12,12/12,15/12,18/12,21/12,24/12)
        >>> o = Bermudan(ref=Stock(50, vol=.6), right='put', K=52, T=2, rf_r=0.1)
        >>> o.calc_px(method='LT', tex=times, nsteps=40, keep_hist=False).px_spec.px       
        13.206509995991107
        >>> ##Price vs. strike curve - example of vectorization of price calculation
        >>> import matplotlib.pyplot as plt
        >>> from numpy import linspace
        >>> Karr = linspace(30,70,101)
        >>> px = tuple(map(lambda i:  Bermudan(ref=Stock(50, vol=.6), right='put', K=Karr[i], T=2, rf_r=0.1).
        ... calc_px(tex=times, nsteps=20).px_spec.px, range(Karr.shape[0])))
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111) 
        >>> ax.plot(Karr,px,label='Bermudan put')
        [<...>]
        >>> ax.set_title('Price of Bermudan put vs K')
        <...>
        >>> ax.set_ylabel('Px')
        <...>
        >>> ax.set_xlabel('K')
        <...>
        >>> ax.grid()
        >>> ax.legend() 
        <...>
        # >>> plt.show()

        MC example #1 - Verifiable example #1: See reference [1], section 5.1 and table 5.1 with arguments N=10^2, R=3

        >>> s = Stock(S0=11, vol=.4)
        >>> o = Bermudan(ref=s, right='put', K=15, T=1, rf_r=.05, desc="in-the-money Bermudan put")
        >>> o.calc_px(method='MC', R=3, npaths=10**2, tex=list([(i+1)/10 for i in range(10)])).px_spec.px
        4.200888

        MC example #2 - Verifiable example #2: See reference [1], section 5.1 and table 5.1 with arguments N=10^5, R=6

        >>> s = Stock(S0=11, vol=.4)
        >>> o = Bermudan(ref=s, right='put', K=15, T=1, rf_r=.05, desc="in-the-money Bermudan put")
        >>> o.calc_px(method='MC', R=6, npaths=10**5, tex=list([(i+1)/10 for i in range(10)])).px_spec.px
        4.204823

        """

        from numpy import asarray
        T = max(tex)
        if self.T < T:
            tex = tuple(asarray(tex)[asarray(tex) < self.T]) + (self.T,)
        if self.T > T:
            tex = tex[:-1] + (self.T,)            
        self.tex = tex
        knsteps = max(tuple(map(lambda i: int(T/(tex[i+1]-tex[i])), range(len(tex)-1))))
        if nsteps != None:
            knsteps = knsteps * nsteps 
        nsteps = knsteps
        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist, R=R)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Bermudan

        .. sectionauthor:: Oleg Melnikov; Andy Liao

        """

        keep_hist = getattr(self.px_spec, 'keep_hist', False)
        n = getattr(self.px_spec, 'nsteps', 3)
        _ = self.LT_specs(n)
        
        #Redo tree steps 

        S = self.ref.S0 * _['d'] ** arange(n, -1, -1) * _['u'] ** arange(0, n + 1)  # terminal stock prices
        O = maximum(self.signCP * (S - self.K), 0)          # terminal option payouts
        # tree = ((S, O),)
        S_tree = (tuple([float(s) for s in S]),)  # use tuples of floats (instead of numpy.float)
        O_tree = (tuple([float(o) for o in O]),)
        # tree = ([float(s) for s in S], [float(o) for o in O],)

        for i in range(n, 0, -1):
            O = _['df_dt'] * ((1 - _['p']) * O[:i] + ( _['p']) * O[1:])  #prior option prices 
            #(@time step=i-1)
            S = _['d'] * S[1:i+1]                   # prior stock prices (@time step=i-1)
            Payout = maximum(self.signCP * (S - self.K), 0)   # payout at time step i-1 (moving backward in time)
            if i*_['dt'] in self.tex:   #The Bermudan condition: exercise only at scheduled times         
                O = maximum(O, Payout)
            # tree = tree + ((S, O),)
            S_tree = (tuple([float(s) for s in S]),) + S_tree
            O_tree = (tuple([float(o) for o in O]),) + O_tree
            # tree = tree + ([float(s) for s in S], [float(o) for o in O],)

        self.px_spec.add(px=float(Util.demote(O)), method='LT', sub_method='binomial tree; Hull Ch.13',
                        LT_specs=_, ref_tree = S_tree if keep_hist else None, opt_tree = O_tree if keep_hist else None)

        # self.px_spec = PriceSpec(px=float(Util.demote(O)), method='LT', sub_method='binomial tree; Hull Ch.13',
        #                 LT_specs=_, ref_tree = S_tree if save_tree else None, opt_tree = O_tree if save_tree 
        #else None)
        return self

    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Bermudan

        .. sectionauthor:: 

        Note
        ----

        """
        return self

    def _calc_MC(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Bermudan

        .. sectionauthor:: Patrick Granahan

        Note
        ----
        Uses the Longstaff-Schwartz algorithm.
        """

        # Get arguments from calc_px
        npaths = getattr(self.px_spec, 'npaths')
        R = getattr(self.px_spec, 'R')

        def payout(stock_price):
            """
            Calculates the payout of a Bermudan option at a given stock_price.

            Parameters
            ----------
            stock_price : numpy.matrix
                A vector of stock prices

            Returns
            -------
            payout : numpy.matrix
                    The vector of payouts.
            """
            payout = np.maximum(self.signCP * (stock_price - self.K), 0)
            return payout

        def generate_GBM_paths():
            """
            Generates a matrix (dimensions tex * npaths) of paths.

            Returns
            -------
            paths : numpy.matrix
                    Matrix of paths generated.
            """
            # Create the zero matrix of paths
            paths = np.zeros((len(self.tex), npaths))

            # Seed the first row
            paths[0] = self.ref.S0 * np.ones(npaths)

            # Fill the matrix
            for i in range(len(self.tex) - 1):
                deltaT = self.tex[i+1] - self.tex[i]
                paths[i+1] = paths[i] * np.exp((((self.rf_r - self.ref.q) - ((self.ref.vol**2) / 2)) * deltaT) +
                                               (self.ref.vol * np.random.randn(npaths) * np.sqrt(deltaT)))

            return np.matrix(paths)

        def Laguerre(R, x):
            """
            Generates the weighted Laguerre polynomial [1] solution.
            Could be made more general by expanding the scope of R, but at potentially reduced speeds.

            For the Longstaff-Schwartz algorithm, we need a set of basis functions that are
                - complete
                - linearly independent
            The Weighted Laguerre polynomials satisfy these properties, and we will use them as our basis functions in
            the same way Longstaff and Schwartz chose them.

            Parameters
            ----------
            R : int
                    The R-th element in the Laguerre polynomial sequence
                    Must be between 0 and 6.
            x : numpy.matrix
                    A vector of values to solve for.

            References
            ----------
            [1] https://en.wikipedia.org/wiki/Laguerre_polynomials

            Returns
            -------
            phi : numpy.matrix
                    A vector of solutions.
            """
            # Convert x to an ndarray, so that this method is readable
            x = np.asarray(x)

            # The explicit solutions for R=0-6
            if R == 0:
                phi = 1
            elif R == 1:
                phi = -x + 1
            elif R == 2:
                phi = 1/2 * (x**2 - 4*x + 2)
            elif R == 3:
                phi = 1/6 * (-x**3 + 9*x**2 - 18*x + 6)
            elif R == 4:
                phi = 1/24 * (x**4 - 16*x**3 + 72*x**2 - 96*x + 24)
            elif R == 5:
                phi = 1/120 * (-x**5 + 25*x**4 - 200*x**3 + 600*x**2 - 600*x + 120)
            elif R == 6:
                phi = 1/720 * (x**6 - 36*x**5 + 450*x**4 - 2400*x**3 + 5400*x**2 - 4320*x + 720)
            else:
                phi = 0
                raise Exception("R is out of range.")

            # Weight s according to the Longstaff-Schwartz algorithm
            phi *= np.exp(-0.5 * x)

            return np.matrix(phi)

        def betas(paths):
            """
            Generates the discounted betas used in the Longstaff-Schwartz algorithm using a regression.

            Parameters
            ----------
            paths : numpy.matrix
                    List of paths produced by generate_GBM_paths

            Returns
            -------
            betas : numpy.matrix
                    A tex * R matrix (list of lists) holding the betas.
            """
            # Initialize betas as an empty matrix (dimensions R * (tex - 1))
            betas = np.matrix(np.zeros((R, len(self.tex))))

            # Laguerre_matrix is a matrix (dimensions npaths * R) that will hold the coefficients
            laguerre_matrix = np.matrix(np.zeros((npaths, R)))

            # Prices is a matrix (dimensions npaths * 1) that holds the prices for each path
            prices = payout(paths[len(self.tex)-1, :].getH())
            print(prices.shape)

            # Step backwards through the exercise dates
            for exercise_date_index in reversed(range(len(self.tex))):
                # Skip the first exercise date
                if exercise_date_index == 0:
                    break

                # Fill the laguerre_matrix
                for i in range(R):
                    # Select the laguerre solutions as a column
                    laguerre_column = Laguerre(i, paths[exercise_date_index, :].getH())
                    laguerre_matrix[:, i] = laguerre_column

                # B_phi_phi is a square matrix (dimensions R * R)
                B_phi_phi = laguerre_matrix.getH() * laguerre_matrix

                # B_prices_phi is a matrix (dimensions 1 * R)
                B_prices_phi = laguerre_matrix.getH() * prices

                # Solve the system of equations and store the betas
                betas[:, exercise_date_index] = np.linalg.solve(B_phi_phi, B_prices_phi)

            # Trim the empty column off the front of betas
            betas = np.delete(betas, 0, 1)

            return betas

        # Generate paths
        paths = generate_GBM_paths()

        # Generate betas
        betas = betas(paths)

        # Prices will store the price found at each path
        prices = np.zeros(npaths)

        # Fill prices
        for path in range(npaths):
            for exercise_date in range(len(self.tex)):

                # Find the continuation price
                continuation_price = 0
                stock_price = paths[exercise_date, path]  # stock price at the exercise date on a given path
                for R in range(R):
                    continuation_price += Laguerre(R, stock_price) * betas[exercise_date, R]

                # If the continuation price is less than the payout, or we have reached the last exercise date,
                # we have found the price for this path.
                if continuation_price < payout(stock_price) or exercise_date == len(self.tex)-1:
                    discount = np.exp(-(self.rf_r - self.ref.q) * (self.T * self.tex[exercise_date] / self.tex[-1]))
                    prices[path] = discount * payout(stock_price)
                    break

        # Find the average price across all the paths, then record it
        price = np.average(prices)
        self.px_spec.add(px=price, sub_method='Longstaff-Schwartz', prices=prices)

        return self

    def _calc_FD(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Bermudan

        .. sectionauthor:: 

        Note
        ----

        """

        return self
