import numpy as np

try: from qfrm.European import *  # production:  if qfrm package is installed
except:   from European import *  # development: if not installed and running from source

try: from qfrm.American import *  # production:  if qfrm package is installed
except:   from American import *  # development: if not installed and running from source



class Compound(European):
    """ Asian option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, **kwargs):
    # def calc_px(self, option, on='put', **kwargs):
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
        on : str
                Either 'call' or 'put'. i.e. 'call' is for call on call or put on call (depends on right)
        rho : float
                The correlation between the reference stock and S2
        nsteps : int
                LT, MC, FD methods require number of times steps
        npaths : int
                MC, FD methods require number of simulation paths
        keep_hist : bool
                If ``True``, historical information (trees, simulations, grid) are saved in ``self.px_spec`` object.

        Returns
        -------
        self : Spread
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object).


        Notes
        -----

        **FD**
        Note: the method is EXTREMELY SENSITIVE to ``S0``

        *References:*

        - `<http://www.wiley.com/legacy/wileychi/pwiqf2/supp/c28.pdf>`_ (SEE PROBLEM 4)
        - `<http://www.math.yorku.ca/~hmzhu/Math-6911/lectures/Lecture5/5_BlkSch_FDM.pdf>`_
        - `<http://www.cs.cornell.edu/info/courses/spring-98/cs522/content/lab4.pdf>`_


        Examples
        --------
        - `Compound Options Introduction and Pricing Spreadsheet <http://investexcel.net/compound-options-excel>`_

        **FD**
        Method is approximate and extremely unstable.
        The answers are thus only an approximate of the BSM Solution

        *Put on Put*

        >>> s = Stock(S0=90., vol=.12, q=.04)
        >>> o = American(ref=s, right='put', K=80., T=12./12., rf_r=.05, desc='http://investexcel.net/compound-options-excel/: POP')
        >>> o2 = Compound(right='put', T=6./12., K = 20.)
        >>> o2.calc_px(method='FD', ref=o, npaths=10, nsteps = 10).px_spec # doctest: +ELLIPSIS
        19.505177573...

        *Call on Put*

        >>> s = Stock(S0=90., vol=.12, q=.04)
        >>> o = American(ref=s, right='put', K=80., T=12./12., rf_r=.05, desc='http://investexcel.net/compound-options-excel/: COP')
        >>> o2 = Compound(right='call', T=6./12., K = 20.)
        >>> o2.calc_px(method='FD', ref=o,npaths=10, nsteps = 10).px_spec.px # doctest: +ELLIPSIS
        0.000112479...

        *Put on Call*

        >>> s = Stock(S0=90., vol=.12, q=.04)
        >>> o = American(ref=s, right='call', K=80., T=12./12., rf_r=.05, desc='http://investexcel.net/compound-options-excel/: POC')
        >>> o2 = Compound(right='put',T=6./12., K = 20.)
        >>> o2.calc_px(method='FD', ref=o, npaths=10,nsteps = 10).px_spec.px # doctest: +ELLIPSIS
        10.465470970...

        *Call on Call*

        >>> s = Stock(S0=90., vol=.12, q=.04)
        >>> o = American(ref=s, right='call', K=80., T=12./12., rf_r=.05, desc='http://investexcel.net/compound-options-excel/: COC')
        >>> o2 = Compound(right='call',T=6./12., K = 20.)
        >>> o2.calc_px(method='FD', ref=o, npaths=10, nsteps = 10).px_spec.px # doctest: +ELLIPSIS
        0.190332192...

        >>> s = Stock(S0=90., vol=.12, q=.04)
        >>> o = American(ref=s, right='call', K=80., T=12./12., rf_r=.05, desc='http://investexcel.net/compound-options-excel/: COC')
        >>> from pandas import Series;  steps = range(3,250)
        >>> O = Series([o.calc_px(method='FD', nsteps=s).px_spec.px for s in steps], steps)
        >>> O.plot(grid=1, title='Price vs Steps')       # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>


        :Authors:
            Scott Morgan
       """

        # self.save_specs(option=option, on=on, **kwargs)
        self.save_specs(on=on, **kwargs)
        return getattr(self, '_calc_' + self.px_spec.method.upper())()

        # self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        # self.on = on
        # self.o = option
        # return getattr(self, '_calc_' + method.upper())()

    def _calc_LT(self):
        """ Internal function for option valuation.        """
        return self


    def _calc_BS(self):
        """ Internal function for option valuation.       """
        return self


    def _calc_MC(self):
        """ Internal function for option valuation.        """
        return self

    def _calc_FD(self):
        """ Internal function for option valuation.

        :Authors:
            Scott Morgan
        """

        # Define grid
        n, m = self.px_spec.steps, self.px_spec.npaths
        x = np.matrix(np.zeros(shape=(n+1, m+1)))

        # Cache parameters
        _, ref = self, self.ref
        T, T_left = _.T
        T_left = ref.T - _.T
        orig_T = ref.T
        vol = ref.ref.vol
        Smax = ref.ref.S0 * 2 if _.signCP else 3
        r = ref.rf_r
        q = .0
        dt = T/(n+0.0)
        dS = Smax/(m+0.0)
        n = int(T/dt)
        m = int(Smax/dS)
        # K = _.K
        S0 = ref.ref.S0
        df = np.exp(-r*dt)

        # EQUATIONS
        def a(j):
            return df*(.5*dt*((vol**2)*(j**2)-(r-q)*j))
        def b(j):
            return df*(1 - dt*((vol**2)*(j**2)))
        def c(j):
            return df*(.5*dt*((vol**2)*(j**2)+(r-q)*j))


        ref.T = T_left
        ref.ref.S0 = Smax
        Smax_price = ref.pxLT(nsteps = 30)
        ref.ref.S0 = 0
        Smin_price = self.ref.pxLT(nsteps = 30)

        # FINAL CONDITIONS
        for i in range(0,m+1):
            ref.ref.S0 = dS*i
            x[n,i] = np.maximum(_.signCP*(ref.pxLT(nsteps=30).px_spec.px - _.K),0)

        # BOUNDARY CONDITIONS
        x[:,0] = np.matrix(list(map(lambda i: (np.maximum(_.signCP*(Smax_price - _.K),0)*np.exp(-r*(n-i)*dt)), range(0,n+1)))).transpose()
        x[:,m] = np.matrix(list(map(lambda i: (np.maximum(_.signCP*(Smin_price - _.K),0)*np.exp(-r*(n-i)*dt)), range(0,n+1)))).transpose()


        # CALCULATE THROUGH GRID
        for i in np.arange(n-1,-1,-1):
            for k in range(1,m):
                j = m-k
                x[i,k] = a(j)*x[i+1,k+1] + b(j)*x[i+1,k] + c(j)*x[i+1,k-1]

        # RETURN BACK TO NORMAL
        ref.ref.S0 = S0
        ref.T = orig_T
        self.px_spec.add(px=x[0,m-S0/dS], sub_method='Explicit FDM')

        return self

        # #GRID
        # x = np.matrix(np.zeros(shape=(n+1, m+1)))
        #
        # # PARAMETERS
        # signCP = 1 if self.right.lower() == 'call' else -1
        # T = self.T
        # T_left = self.o.T - self.T
        # orig_T = self.o.T
        # vol = self.o.ref.vol
        # Smax = 100
        # if self.o.right == 'call':
        #     Smax = 2*self.o.ref.S0
        # else:
        #     Smax = 3*self.o.ref.S0
        # r = self.o.rf_r
        # q = .0
        # dt = T/(n+0.0)
        # dS = Smax/(m+0.0)
        # n = int(T/dt)
        # m = int(Smax/dS)
        # K = self.K
        # S0 = self.o.ref.S0
        # df = np.exp(-r*dt)
        #
        # # EQUATIONS
        # def a(j):
        #     return df*(.5*dt*((vol**2)*(j**2)-(r-q)*j))
        # def b(j):
        #     return df*(1 - dt*((vol**2)*(j**2)))
        # def c(j):
        #     return df*(.5*dt*((vol**2)*(j**2)+(r-q)*j))
        #
        #
        # self.o.T = T_left
        # self.o.ref.S0 = Smax
        # Smax_price = self.o.calc_px(method='LT',nsteps = 30).px_spec.px
        # self.o.ref.S0 = 0
        # Smin_price = self.o.calc_px(method='LT',nsteps = 30).px_spec.px
        #
        # # FINAL CONDITIONS
        # for i in range(0,m+1):
        #     self.o.ref.S0 = dS*i
        #     x[n,i] = np.maximum(signCP*(self.o.calc_px(method='LT',nsteps = 30).px_spec.px-K),0)
        #
        # # BOUNDARY CONDITIONS
        # x[:,0] = np.matrix(list(map(lambda i: (np.maximum(signCP*(Smax_price - K),0)*np.exp(-r*(n-i)*dt)), \
        #                         range(0,n+1)))).transpose()
        # x[:,m] = np.matrix(list(map(lambda i: (np.maximum(signCP*(Smin_price - K),0)*np.exp(-r*(n-i)*dt)), \
        #                         range(0,n+1)))).transpose()
        #
        #
        # # CALCULATE THROUGH GRID
        # for i in np.arange(n-1,-1,-1):
        #     for k in range(1,m):
        #         j = m-k
        #         x[i,k] = a(j)*x[i+1,k+1] + b(j)*x[i+1,k] + c(j)*x[i+1,k-1]
        #
        # # RETURN BACK TO NORMAL
        # self.o.ref.S0 = S0
        # self.o.T = orig_T
        # self.px_spec.add(px=x[0,m-S0/dS], method='FDM', sub_method='Explicit')
        #
        # return self
