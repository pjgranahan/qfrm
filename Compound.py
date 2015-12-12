import numpy as np
import copy

try: from qfrm.European import *  # production:  if qfrm package is installed
except:   from European import *  # development: if not installed and running from source

try: from qfrm.American import *  # production:  if qfrm package is installed
except:   from American import *  # development: if not installed and running from source



class Compound(European):
    """ Asian option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, **kwargs):
        """ Wrapper function that calls appropriate valuation method.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments (``method``, ``nsteps``, ``npaths``, ``keep_hist``, ``rng_seed``, ...)
            are passed to the parent. See ``European.calc_px()`` for details.

        Returns
        -------
        self : Spread
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object).


        Notes
        -----

        **FD**
        Note: the method is EXTREMELY SENSITIVE to ``S0``

        *References:*

        - Compound Options (Lecture 4, MFE5010 at NUS), `Lim Tiong Wee, 2001 <http://1drv.ms/1NFtSeL>`_
        - Binomial Tree pricing for Compound, Chooser, Shout. `Excel Tool <http://goo.gl/AdgcqY>`_
        - Compound Options - Introduction and Pricing Spreadsheet. `Excel Tool. Samir Kahn <http://investexcel.net/compound-options-excel>`_
        - Calculation of Cumulative Probability in Bivariate Normal Distribution, `Technical Note #5, J.C.Hull <http://www-2.rotman.utoronto.ca/~hull/technicalnotes/TechnicalNote5.pdf>`_
        - Finite-Difference Methods for One-Factor Models `(Ch.28) <http://www.wiley.com/legacy/wileychi/pwiqf2/supp/c28.pdf>`_ (see problem 4)
        - Finite Difference Methods, (Ch.5 slides, Math-6911, York University), `Hongmei Zhu <http://www.math.yorku.ca/~hmzhu/Math-6911/lectures/Lecture5/5_BlkSch_FDM.pdf>`_
        - Finite Difference Approach to Option Pricing (Lab 4, CS522 Lab Note, Cornell University), `T.Coleman and R.Jarrow, 1998 <http://1drv.ms/1NVS3rh>`_
        - Finite Difference Approach to Option Pricing (Lab5, CS522 Lab Note, Cornell University), `T.Coleman and R.Jarrow, 1998 <http://1drv.ms/1NVS5zD>`_


        Examples
        --------

        **FD**
        Method is approximate and extremely unstable.
        The answers are thus only an approximate of the BSM Solution

        *Put on Put*

        >>> s = Stock(S0=90, vol=.12)
        >>> o = American(ref=s, right='put', K=80, T=1, rf_r=.05, desc='POP')
        >>> c = Compound(ref=o, right='put', K=20, T=.5, rf_r=.05)
        >>> c.pxFD(npaths=10, nsteps=10)  # doctest: +ELLIPSIS

        >>> s = Stock(S0=90, vol=.12, q=.04)
        >>> o = American(ref=s, right='put', K=80, T=1, rf_r=.05, desc='POP')
        >>> c = Compound(ref=o, right='put', K=20, T=.5, rf_r=.05)
        >>> c.calc_px(method='FD', npaths=10, nsteps=10)  # doctest: +ELLIPSIS
        Compound...19.505177573...

        *Call on Put*

        >>> s = Stock(S0=90, vol=.12, q=.04)
        >>> o = American(ref=s, right='put', K=80, T=12./12, rf_r=.05, desc='COP')
        >>> c = Compound(ref=o, right='call', K = 20, T=6./12, rf_r=.05)
        >>> c.pxFD(npaths=10, nsteps=10)  # doctest: +ELLIPSIS
        Compound...0.000112479...

        *Put on Call*

        >>> s = Stock(S0=90, vol=.12, q=.04)
        >>> o = American(ref=s, right='call', K=80, T=1, rf_r=.05, desc='POC')
        >>> c = Compound(ref=o, right='put', K = 20, T=.5, rf_r=.05)
        >>> c.pxFD(npaths=10, nsteps=10)  # doctest: +ELLIPSIS
        10.465470970...

        *Call on Call*

        >>> s = Stock(S0=90, vol=.12, q=.04)
        >>> o = American(ref=s, right='call', K=80, T=1, rf_r=.05, desc='COC')
        >>> c = Compound(ref=o, right='call', K=20, T=.5, rf_r=.05)
        >>> c.pxFD(npaths=10, nsteps=10)  # doctest: +ELLIPSIS
        0.190332192...

        # >>> s = Stock(S0=90, vol=.12, q=.04)
        # >>> o = American(ref=s, right='call', K=80, T=1, rf_r=.05, desc='COC')
        # >>> from pandas import Series;  steps = range(3, 250)
        # >>> O = Series([o.pxFD(nsteps=s, npaths=10).px_spec.px for s in steps], steps)
        # >>> O.plot(grid=1, title='Price vs Steps')       # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>


        :Authors:
            Scott Morgan
       """
        self.save_specs(**kwargs)
        return getattr(self, '_calc_' + self.px_spec.method.upper())()

    def _calc_LT(self):
        """ Internal function for option valuation.   See ``calc_px()`` for complete documentation.     """
        return self

    def _calc_BS(self):
        """ Internal function for option valuation.  See ``calc_px()`` for complete documentation.     """
        return self

    def _calc_MC(self):
        """ Internal function for option valuation. See ``calc_px()`` for complete documentation.       """
        return self

    def _calc_FD(self):
        """ Internal function for option valuation. See ``calc_px()`` for complete documentation.

        :Authors:
            Scott Morgan
        """

        # todo: computations are off after updating. verify.

        o2 = copy.deepcopy(self.ref)  # safer way of messing with attributes of underlying option.
        _ = self;   T1, K1, rf_r1, right1, ref1, sCP1 = _.T, _.K, _.rf_r, _.right, _.ref, _.signCP # option o1 on option o2
        _ = o2;     T2, K2, rf_r2, right2, ref2 = _.T, _.K, _.rf_r, _.right, _.ref  # option o2 on stock ref
        _ = self.ref.ref;   S0, vol, q = _.S0, _.vol, _.q
        _ = self.px_spec;   n, m = _.nsteps, _.npaths

        x = np.matrix(np.zeros(shape=(n+1, m+1)))  # Define grid

        # Cache parameters
        Smax = S0 * 2 if sCP1 else 3
        dt = T1 / (n + 0.0)
        dS = Smax / (m + 0.0)
        n = int(T1 / dt)
        m = int(Smax / dS)
        df = np.exp(-rf_r1 * dt)

        o2.T = T2 - T1   # time remaining between two expiry dates. Underlying option's expiry (T2) is longer.
        o2.ref.S0 = Smax;        Smax_price = o2.pxLT(nsteps=30)
        o2.ref.S0 = 0;           Smin_price = o2.pxLT(nsteps=30)

        # FINAL CONDITIONS
        for i in range(0, m + 1):
            o2.ref.S0 = dS * i
            x[n, i] = np.maximum(sCP1*(o2.pxLT(nsteps=30) - K1), 0)

        # BOUNDARY CONDITIONS
        def bc(S):
            it = map(lambda i: (np.maximum(sCP1 * (S - K1), 0) * np.exp(-rf_r1 * (n - i) * dt)), range(0, n + 1))
            return np.matrix(list(it)).transpose()
        x[:,0], x[:,m] = bc(Smax_price), bc(Smin_price)

        # EQUATIONS
        def a(j): return df*(.5*dt*((vol**2)*(j**2)-(rf_r1-q)*j))
        def b(j): return df*(1 - dt*((vol**2)*(j**2)))
        def c(j): return df*(.5*dt*((vol**2)*(j**2)+(rf_r1-q)*j))

        # CALCULATE THROUGH GRID
        for i in np.arange(n-1, -1, -1):
            for k in range(1, m):
                j = m-k
                x[i, k] = a(j) * x[i+1, k+1] + b(j) * x[i+1, k] + c(j) * x[i+1, k-1]

        self.px_spec.add(px=float(x[0, m - S0 / dS]), sub_method='Explicit FDM')

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
