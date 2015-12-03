import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

try: from qfrm.OptionValuation import *  # production:  if qfrm package is installed
except:   from OptionValuation import *  # development: if not installed and running from source


class Shout(OptionValuation):
    """ Shout option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, method='LT', nsteps=None, npaths=None, keep_hist=False, seed=None, deg=3):
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
        seed : int
                Seed number for Monte Carlo simulation
        deg : int
                degree of polynomial fit in MC

        Returns
        -------
        self : Shout
            Returned object contains specifications and calculated price in embedded ``PriceSpec`` object.


        Notes
        -----
        Verification of Shout option: http://goo.gl/02jISW
        Hull Ch26.12 P609


        Examples
        --------
        This two excel spreadsheet price shout option.
        http://goo.gl/1rrTCG
        http://goo.gl/AdgcqY

        LT Examples
        -----------
        >>> s = Stock(S0=50, vol=.3)
        >>> o = Shout(ref=s, right='call', K=52, T=2, rf_r=.05, desc='Example from internet excel spread sheet')
        >>> o.pxLT(nsteps=2)
        11.803171357

        >>> o.calc_px(method='LT', nsteps=2, keep_hist=True).px_spec.opt_tree
        ((11.803171356649463,), (0.0, 24.34243306821051), (0.0, 0.0, 39.10594001952546))

        >>> o.calc_px(method='LT', nsteps=2) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        Shout...px: 11.803171357...

        >>> from pandas import Series
        >>> steps = range(1,11)
        >>> O = Series([o.calc_px(method='LT', nsteps=s).px_spec.px for s in steps], steps)
        >>> O.plot(grid=1, title='LT Price vs nsteps') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()

        MC Examples
        -----------
        See example on p.26, p.28 in `<http://core.ac.uk/download/pdf/1568393.pdf>`_
        Note 1:
        MC gives an approximate price. The price will not exactly fit the price in the reference example but fall
        in a range that is close to the example price.
        Suggest parameters: ``nsteps=252``, ``nsteps=10000``
        Note 2:
        Numpy Polyfit will give warnings: Polyfit may be poorly conditioned warnings.warn(msg, RankWarning)

        >>> s = Stock(S0=36, vol=.2)
        >>> o = Shout(ref=s, right='put', K=40, T=1, rf_r=.2, desc='http://core.ac.uk/download/pdf/1568393.pdf')
        >>> o.pxMC(nsteps=5, npaths=5, keep_hist=True, seed=1212)
        4.0

        >>> o.calc_px(method='MC', nsteps=5, npaths=5, keep_hist=True, seed=1212).px_spec
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px: 4.0...

        >>> from pandas import Series;  steps = [1,2,3,4,5]
        >>> O = Series([o.pxMC(nsteps=s, npaths=5, keep_hist=True, seed=1212) for s in steps], steps)
        >>> O.plot(grid=1, title='MC Price vs nsteps')# doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()

        :Authors:
            Mengyan Xie <xiemengy@gmail.com>,
            Hanting Li <hl45@rice.edu>,
            Yen-fei Chen <yensfly@gmail.com>

       """
        self.deg = deg
        self.seed = seed
        return super().calc_px(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)


    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Shout

        :Authors:
            Mengyan Xie <xiemengy@gmail.com>

        Notes
        -----

        The shout option is usually a call option, but with a difference: at any time t before maturity, the holder may
        "shout". The effect of this is that he is guaranteed a minimum payoff of St - K, although he will get the payoff
        of the call option if this is greater than the minimum. In spirit this is the same as the binomial method for
        pricing American options.



        """


        keep_hist = getattr(self.px_spec, 'keep_hist', False)
        n = getattr(self.px_spec, 'nsteps', 3)
        _ = self.LT_specs(n)

        # Get the Price based on Binomial Tree
        S = self.ref.S0 * _['d'] ** np.arange(n, -1, -1) * _['u'] ** np.arange(0, n + 1)  # terminal stock prices
        O = np.maximum(self.signCP * (S - self.K), 0)          # terminal option payouts

        # The end node of tree
        S_tree = (tuple([float(s) for s in S]),)  # use tuples of floats (instead of numpy.float)
        O_tree = (tuple([float(o) for o in O]),)

        for i in range(n, 0, -1):
            # Left number until duration
            left = n - i + 1
            # Left time until duration
            tleft = left * _['dt']
            # d1 and d2 from BS model
            d1 = (0 + (self.rf_r + self.ref.vol ** 2 / 2) * tleft) / (self.ref.vol * np.sqrt(tleft))
            d2 = d1 - self.ref.vol * np.sqrt(tleft)

            # payoff of not shout
            O = _['df_dt'] * ((1 - _['p']) * O[:i] + ( _['p']) * O[1:])  #prior option prices (@time step=i-1)
            # spot tree
            S = _['d'] * S[1:i+1]                   # prior stock prices (@time step=i-1)

            # payoff of shout
            Shout = self.signCP * S / np.exp(self.ref.q * tleft) * Util.norm_cdf(self.signCP * d1) - \
                    self.signCP * S / np.exp(self.rf_r * tleft) * Util.norm_cdf(self.signCP * d2) + \
                    self.signCP * (S - self.K) / np.exp(self.rf_r * tleft)

            # final payoff is the maximum of shout or not shout
            Payout = np.maximum(Shout, 0)
            O = np.maximum(O, Payout)

            S_tree = (tuple([float(s) for s in S]),) + S_tree
            O_tree = (tuple([float(o) for o in O]),) + O_tree

            out = O_tree[0][0]

        self.px_spec.add(px=float(Util.demote(O)), method='LT', sub_method='binomial tree; Hull Ch.13',
                        LT_specs=_, ref_tree = S_tree if keep_hist else None, opt_tree = O_tree if keep_hist else None)


        return self

    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Shout

        .. sectionauthor::

        Note
        ----

        """

        return self

    def _calc_MC(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Shout

        :Authors:
            Yen-fei Chen <yensfly@gmail.com>

        Note
        ----
        [1] http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L4shout.pdf
        [2] Hull, J.C., Options, Futures and Other Derivatives, 9ed, 2014. Prentice Hall, p609.

        """

        n_steps = getattr(self.px_spec, 'nsteps', 3)
        n_paths = getattr(self.px_spec, 'npaths', 3)
        _ = self

        dt = _.T / n_steps
        df = np.exp(-_.rf_r * dt)
        np.random.seed(_.seed)

        option_px = np.zeros((n_steps+1, n_paths) ,'d')
        S = np.zeros((n_steps+1, n_paths) ,'d') # stock price matrix
        S[0,:] = _.ref.S0 # initial value

        # stock price paths
        for t in range(1,n_steps+1):
            random = scipy.stats.norm.rvs(loc=0, scale=1, size=n_paths)
            S[t,:] = S[t-1,:] * np.exp((_.rf_r-_.ref.vol**2/2)*dt + _.ref.vol*random*np.sqrt(dt))

        option_px = np.maximum(_.signCP*(S-_.K), 0) # payoff when not shout
        final_payoff = np.repeat(S[-1,:], n_steps+1, axis=0).reshape(n_paths,n_steps+1)
        shout_px = np.maximum(_.signCP*(final_payoff.transpose()-_.K), _.signCP*(S-_.K))

        for t in range (n_steps-1,-1,-1): # valuation process is similar to American option
            rg = np.polyfit(S[t,:], df*np.array(option_px[t+1,:]), _.deg) # regression at time t
            C= np.polyval(rg, S[t,:]) # continuation values
            # exercise decision: shout or not shout
            option_px[t,:]= np.where(shout_px[t,:]>C, shout_px[t,:], option_px[t+1,:]*df)

        self.px_spec.add(px=float(np.mean(option_px[0,:])), sub_method='Hull p.609')
        return self

    def _calc_FD(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Shout

        .. sectionauthor::

        Note
        ----

        """

        return self

