import math
import numpy.random
import numpy as np

try: from qfrm.European import *  # production:  if qfrm package is installed
except:   from European import *  # development: if not installed and running from source



class Spread(European):
    """ Spread option class.

    """

    def calc_px(self, ref2 = None, rho=.5, **kwargs):
        """ Wrapper function that calls appropriate valuation method.

        Parameters
        ----------
        ref2 : Stock
                Required. Indicated the second stock used in the spread option
        rho : float
                The correlation between the reference stock and ref2
        kwargs : dict
            Keyword arguments (``method``, ``nsteps``, ``npaths``, ``keep_hist``, ``rng_seed``, ...)
            are passed to the parent. See ``European.calc_px()`` for details.


        Returns
        -------
        self : Spread
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object).


        Notes
        ---------
        **BS** analytical solution is not exact and is only
        valid when ``K = 0``. Thus, ``K`` is ignored and this pricing method should only be used to price
        spreads with ``K = 0``.

        **MC** computes correlated paths and computes the average present value of the spread at expiry.

        *References:*

        - `Verify Examples: <http://www.fintools.com/resources/online-calculators/exotics-calculators/spread/>_`
        - `Spread Options (Lecture 3, MFE5010 at NUS), Lim Tiong Wee, 2001 <http://1drv.ms/1NUwPtZ>`_


        Examples
        ------------

        >>> s1 = Stock(S0=30, q=0, vol=.2)
        >>> s2 = Stock(S0=31, q=0, vol=.3)
        >>> o = Spread(ref=s1, rf_r=.05, right='call', K=0, T=2)
        >>> o.pxBS(ref2=s2, rho=.4); o   # doctest: +ELLIPSIS
        5.40990758...

        >>> s1 = Stock(S0=30, q=0, vol=.2)
        >>> s2 = Stock(S0=31, q=0, vol=.3)
        >>> o = Spread(ref = s1, rf_r=.05, right='put', K=0, T=2)
        >>> from pandas import Series;  exps = range(1,10)
        >>> O = Series([o.update(T=t).pxBS(ref2=s2, rho=.4, nsteps=100, npaths=100) for t in exps], exps)
        >>> O.plot(grid=1, title='Price vs Time to Expiry') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>

        **MC**

        >>> s1 = Stock(S0=30, q=0, vol=.2)
        >>> s2 = Stock(S0=31, q=0, vol=.3)
        >>> o = Spread(ref=s1, rf_r=.05, right='call', K=0, T=2)
        >>> o.pxMC(ref2=s2, rho=.4, nsteps=100, npaths=1000, rng_seed=0); o # doctest: +ELLIPSIS
        5.996962262...

        >>> s1 = Stock(S0=30, q=0, vol=.2)
        >>> s2 = Stock(S0=31, q=0, vol=.3)
        >>> o = Spread(ref = s1, rf_r = .05, right='put', K=2, T=2)
        >>> o.pxMC(ref2=s2, rho=.4, nsteps=100, npaths=1000, rng_seed=0); o # doctest: +ELLIPSIS
        5.130287061...

        >>> s1 = Stock(S0=30, q=0, vol=.2)
        >>> s2 = Stock(S0=30, q=0, vol=.2)
        >>> o = Spread(ref=s1, rf_r=.05, right='put', K=1, T=2, desc='Perfectly correlated -- present value of 1')
        >>> o.pxMC(ref2=s2, rho=1, nsteps=100, npaths=1000, rng_seed=2); o   # doctest: +ELLIPSIS
        0.904837418...



        >>> s1 = Stock(S0=30, q=0, vol=.2)
        >>> s2 = Stock(S0=31, q=0, vol=.3)
        >>> o = Spread(ref=s1, rf_r=.05, right='put', K=2, T=2)
        >>> from pandas import Series;  exps = range(1,51)
        >>> O = Series([o.update(T=t).pxMC(ref2=s2, rho=.4, nsteps = 100, npaths=100, rng_seed=0) for t in exps], exps)
        >>> O.plot(grid=1, title='Spread MC price vs time to expiry (years)' + o.specs) # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>

        :Authors:
            Scott Morgan
       """
        assert type(ref2).__name__ == 'Stock', 'ref2 parameter must be another Stock() object'
        assert abs(rho) <= 1
        self.save2px_spec(rho=rho, ref2=ref2, **kwargs)
        return getattr(self, '_calc_' + self.px_spec.method.upper())()

    def _calc_LT(self):
        """ Internal function for option valuation.        """
        return self

    def _calc_BS(self):
        """ Internal function for option valuation using the Black-Scholes Method

        :Authors:
            Scott Morgan
        """
        _ = self;               T, K, rf_r, net_r, sCP = _.T, _.K, _.rf_r, _.net_r, _.signCP
        _ = self.ref;           S, vol, q = _.S0, _.vol, _.q
        _ = self.px_spec;       rho = _.rho
        _ = self.px_spec.ref2;  S2, vol2, q2 = _.S0, _.vol, _.q

        vol = math.sqrt(vol**2 - 2 * rho * vol * vol2 + vol2**2)
        d1 = (1./(vol * math.sqrt(T)))*math.log((S2 * math.exp(-q2 * T))/(S * math.exp(-q * T)))
        d2 = d1 - (vol * math.sqrt(T)/2.)
        d1 = d1 + (vol * math.sqrt(T)/2.)
        p = S2 * math.exp(-q2 * T) * Util.norm_cdf(d1)
        p = p - S * math.exp(-q * T) * Util.norm_cdf(d2)

        self.px_spec.add(px=float(p))
        return self


    def _calc_MC(self):
        """ Internal function for option valuation using Monte-Carlo simulation

        :Authors:
            Scott Morgan
        """
        _ = self;               T, K, rf_r, net_r, sCP = _.T, _.K, _.rf_r, _.net_r, _.signCP
        _ = self.ref;           S, vol, q = _.S0, _.vol, _.q
        _ = self.px_spec;       n, m, keep_hist, rng_seed, rho = _.nsteps, _.npaths, _.keep_hist, _.rng_seed, _.rho
        _ = self.px_spec.ref2;  S2, vol2, q2 = _.S0, _.vol, _.q
        _ = self._LT_specs();   u, d, p, df, dt = _['u'], _['d'], _['p'], _['df_dt'], _['dt']

        px = list()
        numpy.random.seed(rng_seed)

        for path in range(0,m):

            ## Generate correlated Wiener Processes
            u = numpy.random.normal(size=n)
            v = numpy.random.normal(size=n)
            v = rho * u + math.sqrt(1 - rho**2) * v
            u = u * math.sqrt(dt)
            v = v * math.sqrt(dt)

            ## Simulate the paths
            s1, s2, mu1, mu2 = [S], [S2], (rf_r - q)*dt, (rf_r - q2)*dt

            for t in range(0, len(u)):
                s1.append(s1[-1] * (mu1 + vol * u[t]) + s1[-1])
                s2.append(s2[-1] * (mu2 + vol2 * v[t]) + s2[-1])

            val = np.maximum(sCP * (s2[-1] - s1[-1] - K), 0) * math.exp(-rf_r * T)  # Calculate the Payoff
            px.append((val))

        self.px_spec.add(px=float(np.mean(px)))

        return self

    def _calc_FD(self):
        """ Internal function for option valuation.        """
        return self




