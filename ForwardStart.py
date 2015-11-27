from OptionValuation import *
import scipy.stats
import math
import numpy as np

class ForwardStart(OptionValuation):
    """ ForwardStart option class

    Inherits all methods and properties of Optionvalueation class.
    """

    def calc_px(self, T_s=1, method='BS', nsteps=None, npaths=None, keep_hist=False):
        """ Wrapper function that calls appropriate valuation method.

        User passes parameters to calc_px, which saves them to local PriceSpec object
        and calls specific pricing function (_calc_BS,...).
        This makes significantly less docstrings to write, since user is not interfacing pricing functions,
        but a wrapper function calc_px().

        Parameters
        ----------
        method : str
                Required. Indicates a valuation method to be used: 'BS', 'LT', 'MC', 'FD'
        nsteps : int
                LT, MC, FD methods require number of times steps
        npaths : int
                MC, FD methods require number of simulation paths
        keep_hist : bool
                If True, historical information (trees, simulations, grid) are saved in self.px_spec object.
        T1 : float
             Required. Indicates the time that the option starts.
        Returns
        -------
        self : ForwardStart

        Notes
        -----
        [1] https://en.wikipedia.org/wiki/Forward_start_option  -- WikiPedia: Forward start option
        [2] http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2forward.pdf -- \
        How to pricing forward start opions, resource for Example 1
        [3] http://www.globalriskguard.com/resources/deriv/fwd_4.pdf -- \
        How to pricing forward start opions, resource for Example 2

        .. sectionauthor:: Runmin Zhang


        Examples
        --------
        BS Examples
        -----------
        #http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2forward.pdf
        >>> s = Stock(S0=50, vol=.15,q=0.05)
        >>> o=ForwardStart(ref=s, K=50,right='call', T=0.5, \
        rf_r=.1).calc_px(method='BS',T_s=0.5)
        >>> print(o.px_spec.px) #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        2.62877...
        <BLANKLINE>


        >>> s = Stock(S0=60, vol=.30,q=0.04)
        >>> o=ForwardStart(ref=s, K=66,right='call', T=0.75, \
        rf_r=.08).calc_px(method='BS',T_s=0.25)
        >>> print(o.px_spec) #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px: 6.760...
        <BLANKLINE>

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

        The following example will generate px = 2.62029... with nsteps = 365 and npaths = 10000, \
        which can be verified by page 2 http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2forward.pdf
        However, for the purpose if fast runtime, I use nstep = 10 and npaths = 10 in all following examples, \
        whose result does not match verification.
        If you want to verify my code, please use nsteps = 365 and npaths = 10000

        >>> s = Stock(S0=50, vol=.15,q=0.05)
        >>> o=ForwardStart(ref=s, K=100, right='call', T=0.5, rf_r=.1, \
               desc='example from page 2 http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2forward.pdf'\
               ).calc_px(method='MC',nsteps=10,npaths=10,T_s=0.5) #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        >>> print(o.px_spec.px)#doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        3.434189...

        The following example uses the same parameter as the example above, but uses pxMC()
        >>> s = Stock(S0=50, vol=.15,q=0.05)
        >>> o=ForwardStart(ref=s, K=100, right='call', T=0.5, rf_r=.1, \
               desc='example from page 2 http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2forward.pdf'\
               ).pxMC(nsteps=10,npaths=10,T_s=0.5)#doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        >>> print(o)#doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        3.434189...


        The following example will generate px = 1.43860... with nsteps = 365 and npaths = 10000, \
        which can be verified by the xls file in http://investexcel.net/forward-start-options/
        However, for the purpose if fast runtime, I use nstep = 10 and npaths = 10 in all following examples, \
        whose result does not match verification.
        If you want to verify my code, please use nsteps = 365 and npaths = 10000

        >>> s = Stock(S0=50, vol=.15,q=0.05)
        >>> o=ForwardStart(ref=s, K=100, right='call', T=0.5, rf_r=.1, \
               desc='example from http://investexcel.net/forward-start-options/'\
               ).calc_px(method='MC',nsteps=10,npaths=10,T_s=0.5)
        >>> print(o.update(right='put').calc_px(method='MC',\
        nsteps=10,npaths=10,T_s=0.5).px_spec.px) #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        1.27965...

        >>> print(o.update(right='put').calc_px(method='MC',\
        nsteps=10,npaths=10,T_s=0.5).px_spec) #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px: 1.27965...
        <BLANKLINE>

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([ForwardStart(ref=s, K=66,right='call', T=0.5, \
        rf_r=0.1).update(T=t).calc_px(method='MC',T_s=0.5).px_spec.px for t in expiries], expiries)
        >>> O.plot(grid=1, title='ForwardStart option Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> plt.show()







        """


        return super().calc_px(method=method, nsteps=nsteps, \
                               npaths=npaths, keep_hist=keep_hist, T_s=T_s)


    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: ForwardStart

        .. sectionauthor:: Runmin Zhang

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

        Returns
        -------
        self: PerpetualAmerican

        .. sectionauthor::

        """

        return self

    def _calc_MC(self):
        """ Internal function for option valuation.

        Returns
        ---------
        self: ForwardStart

        .. sectionauthor:: Tianyi Yao

        Note
        ----
        [1] http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L2forward.pdf

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

        Returns
        -------
        self: PerpetualAmerican

        .. sectionauthor::

        Note
        ----

        """

        return self



