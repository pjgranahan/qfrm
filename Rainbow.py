import numpy.random
import numpy as np

try: from qfrm.OptionValuation import *  # production:  if qfrm package is installed
except:   from OptionValuation import *  # development: if not installed and running from source


class Rainbow(OptionValuation):
    """ Rainbow option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, method='BS', corr = 0.65, nsteps=None, npaths=None, keep_hist=False):
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
        Corr: float
                 Correlation of two assets in a rainbow

        Returns
        -------
        self : Rainbow
            Returned object contains specifications and calculated price in embedded ``PriceSpec`` object.

        Notes
        -----
        The examples can be verified with `<http://goo.gl/7H3U0N> p.23`_.
        The results might differ a little due to the simulations.
        Since it takes time to run more paths and steps, the number of simulations is not very large in examples.
        To improve accuracy, please improve the ``npaths`` and ``nsteps``.

        Examples
        -------

        >>> s = Stock(S0=(100,50), vol=(.25,.45))
        >>> o = Rainbow(ref=s, right='call', K=40, T=.25, rf_r=.05, desc='http://goo.gl/7H3U0N p.23')

        >>> o.calc_px(method='MC',corr=0.65,npaths=10,nsteps=10).px_spec # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px: 14.394869309...


        >>> s = Stock(S0=(100,50), vol=(.25,.45))
        >>> o = Rainbow(ref=s, right='put', K=55, T=0.25, rf_r=.05, desc='Hull p.612')

        >>> o.pxMC(corr=0.65, npaths=10, nsteps=10)
        6.147189494

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([o.update(T=t).calc_px(method='MC',\
        corr=0.65,npaths=2,nsteps=3).px_spec.px for t in expiries], expiries)
        >>> O.plot(grid=1, title='Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()

        :Authors:
          Mengyan Xie <xiemengy@gmail.com>

        """
        self.corr = corr
        self.npaths = npaths
        self.nsteps = nsteps
        return super().calc_px(method=method,corr = corr, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)

    def _calc_BS(self):
        """ Internal function for option valuation.

        """

        return self

    def _calc_LT(self):
        """ Internal function for option valuation.

        """

        return self

    def _calc_MC(self, keep_hist=False):
        """ Internal function for option valuation.

        :Authors:
          Mengyan Xie <xiemengy@gmail.com>
        """

        _ = self
        n_steps = getattr(_.px_spec, 'nsteps', 3)
        n_paths = getattr(_.px_spec, 'npaths', 3)


        dt = _.T / n_steps
        df = np.exp(-_.rf_r * dt)
        np.random.seed(_.seed0)

        # Stock price paths


        S = _.ref.S0 * np.exp(np.cumsum(np.random.normal((_.rf_r - 0.5 * _.ref.vol ** 2) * dt,\
                                                         _.ref.vol * np.sqrt(dt), (n_steps + 1, n_paths)), axis=0))
        print(S)
        #S[0] = _.ref.S0
        #s2 = S

        # When the stock price is greater than K2
        #V = np.maximum(_.signCP * (S - _.K2), 0)

        # The payout is signCP * (S - K1)
        #payout = np.maximum(_.signCP * (s2 - _.K), 0) #payout
        #h = np.where(V > 0.0, payout, V) # payout if V > 0.0, payout else 0.0

        # Add the time value of each steps
        #for t in range(n_steps-1, -1, -1):
        #    h[t,:] = h[t+1,:] * df

        #self.px_spec.add(px=float(np.mean(h[0,:])), sub_method='Hull p.601')

        return self


    def _calc_FD(self, nsteps=3, npaths=4, keep_hist=False):
        """ Internal function for option valuation.

        Returns
        -------
        self: European

        .. sectionauthor::

        """
        return self

s = Stock(S0=(100,50), vol=(.25,.45))
o = Rainbow(ref=s, right='call', K=40, T=.25, rf_r=.05, desc='http://goo.gl/7H3U0N p.23')

o.calc_px(method='MC',corr=0.65,npaths=10,nsteps=10)