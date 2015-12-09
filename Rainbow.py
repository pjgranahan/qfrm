import numpy as np
import math

try: from qfrm.European import *  # production:  if qfrm package is installed
except:   from European import *  # development: if not installed and running from source


class Rainbow(European):
    """ Rainbow option class.
    """

    def calc_px(self, corr, **kwargs):
        """ Wrapper function that calls appropriate valuation method.

        Current implementation is for two underlying equities.

        Parameters
        ----------
        corr: float
             Correlation of two assets in a rainbow
        kwargs : dict
            Keyword arguments (``method``, ``nsteps``, ``npaths``, ``keep_hist``, ``rng_seed``, ...)
            are passed to the parent. See ``European.calc_px()`` for details.

            To pass multiple underlying stocks, input values as tuples into each argument of ``Stock.__init__()``,
            i.e. S0, vol, q, ...  Of course, tuples should be of the same size.

        Returns
        -------
        self : Rainbow
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object).

        Notes
        -----
        The examples can be verified with
        `Monte Carlo Simulation in the Pricing of Derivatives, Cara M.Marshall, 2008, p.23 <http://1drv.ms/1m4HPsj>`_
        To improve accuracy, raise iterations (``npaths`` and ``nsteps``).

        Examples
        --------

        **MC**

<<<<<<< .merge_file_a07444
        >>> s = Stock(S0=(100,50), vol=(.25, .45))
        >>> o = Rainbow(ref=s, right='call', K=40, T=.25, rf_r=.05, desc="See p.23 of Marshall's paper")
        >>> o.calc_px(method='MC', corr=0.65, nsteps=1, npaths=100, rng_seed=0) # doctest: +ELLIPSIS
        Rainbow...px: 14.908873539...


        >>> s = Stock(S0=(100,50), vol=(.25, .45))
        >>> o = Rainbow(ref=s, right='put', K=55, T=0.25, rf_r=.05, desc='Hull p.612')
        >>> o.pxMC(corr=0.65, nsteps=1, npaths=100, rng_seed=2)
=======
        >>> s = Stock(S0=(100, 50), vol=(.25, .45))
        >>> o = Rainbow(ref=s, right='call', K=40, T=.25, rf_r=.05, desc="See p.23 of Marshall's paper")
        >>> o.calc_px(method='MC', corr=0.65, nsteps=100, npaths=1000, rng_seed=0)  # doctest: +ELLIPSIS
        Rainbow...px: 14.908873539...


        >>> s = Stock(S0=(100, 50), vol=(.25, .45))
        >>> o = Rainbow(ref=s, right='put', K=55, T=0.25, rf_r=.05, desc='Hull p.612')
        >>> o.pxMC(corr=0.65, nsteps=100, npaths=5000, rng_seed=2)
>>>>>>> .merge_file_a04568
        13.91461925

        >>> from pandas import Series
        >>> Ts = range(1,11)   # expiries, in years
        >>> O = Series([o.update(T=t).pxMC(corr=0.65, npaths=2, nsteps=3) for t in Ts], Ts)
        >>> O.plot(grid=1, title='Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()


        :Authors:
          Mengyan Xie <xiemengy@gmail.com>
        """
<<<<<<< .merge_file_a07444
        # self.corr = corr
        # self.npaths = npaths
        # self.nsteps = nsteps
        # self.seed0 = seed0
        # return super().calc_px(method=method,corr = corr, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
=======
>>>>>>> .merge_file_a04568
        assert Util.is_number(corr) and abs(corr) <= 1, 'Correlation is number between -1 and 1, inclusive.'
        self.save2px_spec(corr=corr, **kwargs)
        return getattr(self, '_calc_' + self.px_spec.method.upper())()

    def _calc_BS(self):
        """ Internal function for option valuation.        """
        return self

    def _calc_LT(self):
        """ Internal function for option valuation.        """
        return self

    def _calc_MC(self, keep_hist=False):
        """ Internal function for option valuation.

        :Authors:
            Mengyan Xie <xiemengy@gmail.com>
        """
        _ = self.px_spec;   n, m, corr, rng_seed = _.nsteps, _.npaths, _.corr, _.rng_seed
        _ = self.ref;       S0, vol, q = _.S0, _.vol, _.q
        _ = self;           T, K, rf_r, net_r, sCP = _.T, _.K, _.rf_r, _.net_r, _.signCP

        dt = T / n
        df = math.exp(-rf_r * dt)
        np.random.seed(rng_seed)
        h = list()

<<<<<<< .merge_file_a07444
        for path in range(0, n):
=======
        for path in range(0, m):
>>>>>>> .merge_file_a04568
            # Generate correlated Wiener Processes
            # Compute random variables with correlation
            z1 = np.random.normal(loc=0.0, scale=1.0, size=n)
            z2 = np.random.normal(loc=0.0, scale=1.0, size=n)
            r1 = z1
            r2 = corr * z1 + math.sqrt(1 - corr ** 2) * z2

            # Simulate the paths
            S1 = [S0[0]]
            S2 = [S0[1]]
            mu = net_r * dt

            # Compute stock price
            for t in range(0, n):
                S1.append(S1[-1] * (mu + vol[0] * r1[t]) + S1[-1])
                S2.append(S2[-1] * (mu + vol[1] * r2[t]) + S2[-1])

            # Maximum payout of S1 and S2
            payout = np.maximum(sCP * (S1[-1] - S1[0]), sCP * (S2[-1] - S2[0]))
            v = np.maximum(payout, 0.0) * df
            # The payout is maximum of V and 0
            h.append(v)
        self.px_spec.add(px=float(np.mean(h)), sub_method='J.C.Hull p.601')

        return self

    def _calc_FD(self, nsteps=3, npaths=4, keep_hist=False):
        """ Internal function for option valuation.        """
        return self
