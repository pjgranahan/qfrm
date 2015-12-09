import math
import numpy as np

try: from qfrm.European import *  # production:  if qfrm package is installed
except:   from European import *  # development: if not installed and running from source


class ForwardStart(European):
    """ ForwardStart option class

    Inherits all methods and properties of ``Optionvalueation`` class.
    """

    def calc_px(self, T_s, **kwargs):
        """ Wrapper function that calls appropriate valuation method.

        Parameters
        ----------
        T_s : float
                Indicates the time that the option starts.
        kwargs : dict
            Keyword arguments (``method``, ``nsteps``, ``npaths``, ``keep_hist``, ``rng_seed``, ...)
            are passed to the parent. See ``European.calc_px()`` for details.

        Returns
        -------
        self : ForwardStart
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object).


        Notes
        -----
        In this implementation, we assume that at time ``T_s``, the strike is automatically set
        to be equal to the underlying price at ``T_s``, meaning we have at-the-money option.
        Hence, the user-supplied strike price ``K`` is ignored.

        *References:*

        - Forward start option, `Wikipedia <https://en.wikipedia.org/wiki/Forward_start_option>`_
        - Verify example 1 with `Forward start options (Lecture 4, MFE5010 at NUS), Lim Tiong Wee, 2001 <http://1drv.ms/1XS4R1e>`_
        - Verify example 4 with `Forward Start Options, from GlobalRiskGuard.com. <http://1drv.ms/1R2gFiw>`_
        - Forward Start Options - Introduction and Spreadsheet. `Excel tool. Samir Khan <http://investexcel.net/forward-start-options>`_


        Examples
        --------

        See example on p.2 of `Forward start options, from edu.sg <http://1drv.ms/1XS4R1e>`_

        **BS**

        >>> s = Stock(S0=50, vol=.15,q=0.05)
        >>> ForwardStart(ref=s, K=50, right='call', T=0.5, rf_r=.1).pxBS(T_s=0.5)
        2.628777267

        >>> s = Stock(S0=60, vol=.30,q=0.04)
        >>> o=ForwardStart(ref=s, K=66, right='call', T=0.75, rf_r=.08).calc_px(method='BS',T_s=0.25)
        >>> o.px_spec #doctest: +ELLIPSIS
        PriceSpec...px: 6.760976029...

        >>> s = Stock(S0=60, vol=.30,q=0.04)
        >>> o=ForwardStart(ref=s, K=66, right='put', T=0.75, rf_r=.08).calc_px(method='BS',T_s=0.25)
        >>> o.px_spec #doctest: +ELLIPSIS
        PriceSpec...px: 5.057238874

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([ForwardStart(ref=s, K=66, right='call', T=0.75, rf_r=.08).update(T=t).pxBS(T_s=0.25) for t in expiries], expiries)
        >>> O.plot(grid=1, title='ForwardStart option Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>



        **MC**

        Please note that the following MC examples will only generate results that matches the output of online source
        if we use ``nsteps=365`` and ``npaths = 10000``.
        For fast runtime purpose, I use ``nsteps=10`` and ``npaths = 10``
        in the following examples, which may not generate results that match the output of online source

        Use a Monte Carlo simulation to price a forwardstart option

        The following example will generate ``px = 2.620293977...`` with ``nsteps = 365`` and ``npaths = 10000``,
        which can be verified by
        `Forward Start Options, p.2 <http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2forward.pdf>`_
        However, for the purpose if fast runtime, I use ``nstep = 10`` and ``npaths = 10`` in all following examples,
        whose result does not match verification.
        If you want to verify my code, please use ``nsteps = 365`` and ``npaths = 10000`` in the following example.


        >>> s = Stock(S0=50, vol=.15,q=0.05)
        >>> ForwardStart(ref=s, K=100, right='call', T=0.5, rf_r=.1).pxMC(nsteps=10, npaths=10, T_s=0.5)
        3.434189097

        The following example uses the same parameter as the example above, but uses ``pxMC()``.
        example from page 2 http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2forward.pdf

        >>> s = Stock(S0=50, vol=.15,q=0.05)
        >>> o = ForwardStart(ref=s, K=100, right='call', T=0.5, rf_r=.1)
        >>> o.pxMC(nsteps=10,npaths=10,T_s=0.5)
        3.434189097


        The following example will generate ``px = 1.438603501...`` with ``nsteps = 365`` and ``npaths = 10000``,
        which can be verified by the xls file in
        `Forward Start Options - Intro and Spreadsheet <http://investexcel.net/forward-start-options/>`_
        However, for the purpose if fast runtime, I use ``nstep = 10`` and ``npaths = 10`` in all following examples,
        whose result does not match verification.
        If you want to verify my code, please use ``nsteps = 365`` and ``npaths = 10000``

        >>> s = Stock(S0=50, vol=.15,q=0.05)
        >>> o = ForwardStart(ref=s, K=100, right='call', T=0.5, rf_r=.1).calc_px(method='MC', nsteps=10, npaths=10, T_s=0.5)
        >>> o.update(right='put').calc_px(method='MC', nsteps=10, npaths=10, T_s=0.5).px_spec.px # doctest: +ELLIPSIS
        1.279658389...

        >>> o.update(right='put').calc_px(method='MC',  nsteps=10, npaths=10, T_s=0.5).px_spec # doctest: +ELLIPSIS
        PriceSpec...px: 1.279658389...

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([ForwardStart(ref=s, K=66, right='call', T=0.5, rf_r=0.1).update(T=t).pxMC(T_s=0.5) for t in expiries], expiries)
        >>> O.plot(grid=1, title='ForwardStart option Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>


        **FD**

        The result of this method is influenced by many parameters.
        This method is approximate and extremely unstable.
        The answers are thus only an approximate of the BSM Solution

        >>> s = Stock(S0=50, vol=.15,q=0.05)
        >>> o = ForwardStart(ref=s, K=50, right='call', T=1, rf_r=.01)
        >>> o.pxFD(nsteps=4, npaths = 9, T_s=0.5) #doctest: +ELLIPSIS
        2.38237683

        >>> s = Stock(S0=60, vol=.30,q=0.04)
        >>> o = ForwardStart(ref=s, K=66,right='call', T=0.75, rf_r=.08).calc_px(method='FD', nsteps=4, npaths = 9, T_s=0.25)
        >>> o.px_spec # doctest: +ELLIPSIS
        PriceSpec...px: 6.4149988...

        >>> from pandas import Series
        >>> expiries = range(1,5)
        >>> O = Series([ForwardStart(ref=s, K=66,right='call', T=0.5, rf_r=.01).update(T=t).pxFD(T_s=0.5) for t in expiries], expiries)
        >>> O.plot(grid=1, title='ForwardStart option Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>


        :Authors:
            Runmin Zhang <Runmin.Zhang@rice.edu>,
            Tianyi Yao   <ty13@rice.edu>,
            Mengyan Xie  <xiemengy@gmail.com>
        """

        self.save2px_spec(T_s=T_s, **kwargs)
        return getattr(self, '_calc_' + self.px_spec.method.upper())()


    def _calc_BS(self):
        """ Internal function for option valuation.        See ``calc_px()`` for complete documentation.

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

        # Import external functions


        # Parameters in BSM
        d1 = ((r-q+vol**2/2)*T)/(vol*math.sqrt(T))
        d2 = d1 - vol*math.sqrt(T)


        # Calculate the option price
        N = Util.norm_cdf
        if _.signCP==1:
            px = S0*math.exp(-q*T_s)*( math.exp(-q*T)*N(d1) -math.exp(-r*T)*N(d2) )
        elif _.signCP==-1:
            px = S0*math.exp(-q*T_s)*( -math.exp(-q*T)*N(-d1) +math.exp(-r*T)*N(-d2) )

        self.px_spec.add(px=float(px), method='BS', sub_method=None)
        return self


    def _calc_LT(self):
        """ Internal function for option valuation.       See ``calc_px()`` for complete documentation.
        """

        return self

    def _calc_MC(self):
        """ Internal function for option valuation.        See ``calc_px()`` for complete documentation.

        :Authors:
            Tianyi Yao <ty13@rice.edu>
        """

        #extract MC parameters
        _ = self.px_spec;  n, m, T_s = _.nsteps, _.npaths, _.T_s
        _ = self.ref;     S0, vol, q = _.S0, _.vol, _.q
        _ = self;

        #Make sure strike price is set to the expected underlying price at T_S
        _.K = S0*np.exp((_.rf_r-q) * T_s)

        #compute additional parameters such as time step and discount factor
        dt = _.T / n
        df = np.exp(-_.rf_r * dt)

        np.random.seed(1) #set seed

        #initialize the price array
        S=np.zeros((n+1,m), 'd')
        S[0,:] = S0 * np.exp((_.rf_r - q) * T_s) # set initial price

        #generate stock price path
        for t in range(1, n+1):
            #generate random numbers
            rand=np.random.standard_normal(m)

            S[t,:]=S[t-1,:]*np.exp(((_.rf_r-q-((vol**2)/2))*dt)+(vol*rand*np.sqrt(dt)))

        #find the payout at maturity
        final=np.maximum(_.signCP*(S[-1]-_.K),0)

        #discount the expected payoff at maturity to present
        v0 = (np.exp(-_.rf_r*(_.T+T_s))*sum(final))/m

        self.px_spec.add(px=float(v0), sub_method=None)

        return self

    def _calc_FD(self):
        """ Internal function for option valuation.     See ``calc_px()`` for complete documentation.

        :Authors:
            Mengyan Xie <xiemengy@gmail.com>
        """
        _ = self.px_spec;  n, m = _.nsteps, _.npaths
        _ = self
        dt = _.T / n

        # Get the Price based on FD method
        S = np.linspace(0.5*_.ref.S0, 1.5*_.ref.S0, m)[::-1]

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
        self.px_spec.add(px=float(out), sub_method='Explicit')
        return self

