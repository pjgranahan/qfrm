import math
import numpy as np
import scipy.stats

try: from qfrm.OptionValuation import *  # production:  if qfrm package is installed
except:   from OptionValuation import *  # development: if not installed and running from source


class ForwardStart(OptionValuation):
    """ ForwardStart option class

    Inherits all methods and properties of Optionvalueation class.
    """

    def calc_px(self, T_s=1, method='BS', nsteps=None, npaths=None, keep_hist=False):
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
        T_s : float
                Required. Indicates the time that the option starts.

        Returns
        -------
        ForwardStart
            Returned object contains specifications and calculated price in embedded ``PriceSpec`` object.

        Notes
        -----
        [1] https://en.wikipedia.org/wiki/Forward_start_option  -- WikiPedia: Forward start option
        [2] http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2forward.pdf -- \
        How to pricing forward start opions, resource for Example 1
        [3] http://www.globalriskguard.com/resources/deriv/fwd_4.pdf -- \
        How to pricing forward start opions, resource for Example 2




        Examples
        --------
        **BS Examples**

        #http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2forward.pdf
        >>> s = Stock(S0=50, vol=.15,q=0.05)
        >>> o=ForwardStart(ref=s, K=50,right='call', T=0.5, \
        rf_r=.1).calc_px(method='BS',T_s=0.5)
        >>> o.px_spec.px #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        2.628777266...



        >>> s = Stock(S0=60, vol=.30,q=0.04)
        >>> o=ForwardStart(ref=s, K=66,right='call', T=0.75, \
        rf_r=.08).calc_px(method='BS',T_s=0.25)
        >>> o.px_spec #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px: 6.760976029...


        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([ForwardStart(ref=s, K=66,right='call', T=0.75, \
        rf_r=.08).update(T=t).calc_px(method='BS',T_s=0.25).px_spec.px for t in expiries], expiries)
        >>> O.plot(grid=1, title='ForwardStart option Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> plt.show()



        Examples using _calc_MC()
        -------------------------------------

        Notes
        -----
        Verification of examples: page 2 http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2forward.pdf

        Please note that the following MC examples will only generate results that matches the output of online source\
        if we use nsteps=365 and npaths = 10000. For fast runtime purpose, I use nsteps=10 and npaths = 10 \
        in the following examples, which may not generate results that match the output of online source



        Use a Monte Carlo simulation to price a forwardstart option

        The following example will generate px = 2.620293977...with nsteps = 365 and npaths = 10000, \
        which can be verified by page 2 http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2forward.pdf
        However, for the purpose if fast runtime, I use nstep = 10 and npaths = 10 in all following examples, \
        whose result does not match verification.
        If you want to verify my code, please use nsteps = 365 and npaths = 10000 in the following example.

        >>> s = Stock(S0=50, vol=.15,q=0.05)
        >>> o=ForwardStart(ref=s, K=100, right='call', T=0.5, rf_r=.1, \
               desc='example from page 2 http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2forward.pdf'\
               ).calc_px(method='MC',nsteps=10,npaths=10,T_s=0.5) #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        >>> o.px_spec.px#doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        3.434189097...

        The following example uses the same parameter as the example above, but uses pxMC()
        >>> s = Stock(S0=50, vol=.15,q=0.05)
        >>> o=ForwardStart(ref=s, K=100, right='call', T=0.5, rf_r=.1, \
               desc='example from page 2 http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2forward.pdf')
        >>> o.pxMC(nsteps=10,npaths=10,T_s=0.5)
        3.434189097


        The following example will generate px = 1.438603501...with nsteps = 365 and npaths = 10000, \
        which can be verified by the xls file in http://investexcel.net/forward-start-options/
        However, for the purpose if fast runtime, I use nstep = 10 and npaths = 10 in all following examples, \
        whose result does not match verification.
        If you want to verify my code, please use nsteps = 365 and npaths = 10000

        >>> s = Stock(S0=50, vol=.15,q=0.05)
        >>> o=ForwardStart(ref=s, K=100, right='call', T=0.5, rf_r=.1, \
               desc='example from http://investexcel.net/forward-start-options/'\
               ).calc_px(method='MC',nsteps=10,npaths=10,T_s=0.5)
        >>> o.update(right='put').calc_px(method='MC',\
        nsteps=10,npaths=10,T_s=0.5).px_spec.px #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        1.279658389...

        >>> o.update(right='put').calc_px(method='MC',\
        nsteps=10,npaths=10,T_s=0.5).px_spec #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px: 1.279658389...

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([ForwardStart(ref=s, K=66,right='call', T=0.5, \
        rf_r=0.1).update(T=t).calc_px(method='MC',T_s=0.5).px_spec.px for t in expiries], expiries)
        >>> O.plot(grid=1, title='ForwardStart option Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> plt.show()



        Examples using _calc_FD()
        -------------------------------------

        Notes
        -----
        Verification of examples: page 2 http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2forward.pdf

        The result of this method is influenced by many parameters.
        This method is approximate and extremely unstable.
        The answers are thus only an approximate of the BSM Solution

        >>> s = Stock(S0=50, vol=.15,q=0.05)
        >>> o=ForwardStart(ref=s, K=50, right='call', T=1, rf_r=.01, \
               desc='example from page 2 http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2forward.pdf')
        >>> o.pxFD(nsteps=4,T_s=0.5) #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        2.38237683

        >>> s = Stock(S0=60, vol=.30,q=0.04)
        >>> o=ForwardStart(ref=s, K=66,right='call', T=0.75, \
        rf_r=.08).calc_px(method='FD',T_s=0.25)
        >>> o.px_spec #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px: 6.402007239...

        >>> from pandas import Series
        >>> expiries = range(1,5)
        >>> O = Series([ForwardStart(ref=s, K=66,right='call', T=0.5, \
        rf_r=.01).update(T=t).calc_px(method='FD',T_s=0.5).px_spec.px for t in expiries], expiries)
        >>> O.plot(grid=1, title='ForwardStart option Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> plt.show()

        :Authors:
            Runmin Zhang <Runmin.Zhang@rice.edu>,
            Tianyi Yao   <ty13@rice.edu>,
            Mengyan Xie  <xiemengy@gmail.com>
        """


        return super().calc_px(method=method, nsteps=nsteps, \
                               npaths=npaths, keep_hist=keep_hist, T_s=T_s)


    def _calc_BS(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Runmin Zhang <Runmin.Zhang@rice.edu>
        """

        _ = self

        # Verify the input
        try:
            right   =   _.right.lower()[0]
        except:
            print('Input error. right should be string')
            return False

        #Make sure strike price is set to the expected underlying price at T_S



        try:

            S0   =   float(_.ref.S0)
            T   =   float(_.T)
            T_s  =   float(_.px_spec.T_s)
            vol =   float(_.ref.vol)
            r   =   float(_.rf_r)
            q   =   float(_.ref.q)
        except:
            print('Input error. S, T, T1, vol, r, q should be floats.')
            return False

        _.K = _.ref.S0*np.exp((_.rf_r-_.ref.q)*_.px_spec.T_s)

        try:
            K = _.ref.S0*np.exp((_.rf_r-_.ref.q)*_.px_spec.T_s)
        except:
            print('Input error. K is None.')


        assert right in ['c','p'], 'right should be either "call" or "put" '
        assert vol >= 0, 'vol >=0'
        assert T > 0, 'T > 0'
        assert T_s >=0, 'T_s >= 0'
        assert S0 >= 0, 'S >= 0'
        assert r >= 0, 'r >= 0'
        assert q >= 0, 'q >= 0'

        # Import external functions


        # Parameters in BSM
        d1 = ((r-q+vol**2/2)*T)/(vol*math.sqrt(T))
        d2 = d1 - vol*math.sqrt(T)


        # Calculate the option price
        if right=='c':
            px = S0*math.exp(-q*T_s)*( math.exp(-q*T)*scipy.stats.norm.cdf(d1)\
                                       -math.exp(-r*T)*scipy.stats.norm.cdf(d2) )
        elif right=='p':
            px = S0*math.exp(-q*T_s)*( -math.exp(-q*T)*scipy.stats.norm.cdf(-d1)\
                                       +math.exp(-r*T)*scipy.stats.norm.cdf(-d2) )

        self.px_spec.add(px=float(px), method='BS', sub_method=None)
        return self


    def _calc_LT(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:

        """

        return self

    def _calc_MC(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        Note
        ----
        [1] `<http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2forward.pdf>`

        :Authors:
            Tianyi Yao <ty13@rice.edu>
        """

        #extract MC parameters
        n_steps = getattr(self.px_spec, 'nsteps', 3)
        n_paths = getattr(self.px_spec, 'npaths', 3)
        _ = self


        #Make sure strike price is set to the expected underlying price at T_S
        _.K = _.ref.S0*np.exp((_.rf_r-_.ref.q)*_.px_spec.T_s)

        #compute additional parameters such as time step and discount factor
        dt = _.T / n_steps
        df = np.exp(-_.rf_r * dt)


        np.random.seed(1) #set seed


        #initialize the price array
        S=np.zeros((n_steps+1,n_paths),'d')
        S[0,:]=_.ref.S0*np.exp((_.rf_r-_.ref.q)*_.px_spec.T_s) #set initial price

        #generate stock price path
        for t in range(1,n_steps+1):
            #generate random numbers
            rand=np.random.standard_normal(n_paths)

            S[t,:]=S[t-1,:]*np.exp(((_.rf_r-_.ref.q-((_.ref.vol**2)/2))*dt)+(_.ref.vol*rand*np.sqrt(dt)))

        #find the payout at maturity
        final=np.maximum(_.signCP*(S[-1]-_.K),0)

        #discount the expected payoff at maturity to present
        v0 = (np.exp(-_.rf_r*(_.T+_.px_spec.T_s))*sum(final))/n_paths

        self.px_spec.add(px=float(v0), method='MC', sub_method=None)

        return self

    def _calc_FD(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        Note
        ----
        [1] http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2forward.pdf

        :Authors:
            Mengyan Xie <xiemengy@gmail.com>

        """

        keep_hist = getattr(self.px_spec, 'keep_hist', False)
        n = getattr(self.px_spec, 'nsteps', 3)
        _ = self
        dt = _.T / n

        # Get the Price based on FD method
        S = np.linspace(0.5*_.ref.S0, 1.5*_.ref.S0, 9)[::-1]

        # Find the index of S which nearest to the strike price K
        idx = (np.abs(S - _.K)).argmin()

        # Make sure S_S is set to the expected underlying price at T_S
        S_S = _.ref.S0 * np.exp((_.rf_r - _.ref.q) * _.px_spec.T_s)

        O = np.maximum(_.signCP * (S - S_S), 0)          # terminal option payouts
        max_O = max(O)
        min_O = min(O)

        # The end node of grid
        O_grid = (tuple([float(o) for o in O]),)

        # Calculate a, b and c parameters
        a = [1 / (1 + _.rf_r * dt) * (-0.5 * (_.rf_r - _.ref.q) * j * dt + \
                                      0.5 * (_.ref.vol**2) * (j**2) * dt) for j in range(1, len(O)-1)]
        b = [1 / (1 + _.rf_r * dt) * (1 - (_.ref.vol**2) * (j**2) * dt) for j in range(1, len(O)-1)]
        c = [1 / (1 + _.rf_r * dt) * (0.5 * (_.rf_r - _.ref.q) * j * dt + 0.5 * (_.ref.vol**2) * (j**2) * dt)\
             for j in range(1, len(O)-1)]

        # Backward calculate each node of payoff
        for i in range(n, 0, -1):
            O_a = O[:-2]
            O_b = O[1:-1]
            O_c = O[2:]

            # original payoff
            O = a * O_a + b * O_b + c * O_c
            O_new = np.insert(O, min_O, max_O)
            O_new = np.append(O_new, 0)

            # final payoff is the maximum of payoff and 0
            O = np.maximum(O_new, 0)
            O_grid = (tuple([float(o) for o in O]),) + O_grid

        out = O_grid[0][idx]
        self.px_spec.add(px=float(out), method='FDM', sub_method='Explicit')
        return self
