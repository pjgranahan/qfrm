import numpy as np
import scipy.stats

try: from qfrm.European import *  # production:  if qfrm package is installed
except:   from European import *  # development: if not installed and running from source


class Shout(European):
    """ Shout option class.

    The shout option is usually a call option, but with a difference: at any time t before maturity, the holder may
    "shout". The effect of this is that he is guaranteed a minimum payoff of St - K, although he will get the payoff
    of the call option if this is greater than the minimum. In spirit this is the same as the binomial method for
    pricing American options.
    """

    def calc_px(self, deg=5, **kwargs):
        """ Wrapper function that calls appropriate valuation method.

        Parameters
        ----------
        deg : int
            degree of polynomial fit in MC. Usually, around 5.
        kwargs : dict
            Keyword arguments (``method``, ``nsteps``, ``npaths``, ``keep_hist``, ``rng_seed``, ...)
            are passed to the parent. See ``European.calc_px()`` for details.

        Returns
        -------
        self : Shout
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object).


        Notes
        -----
        Verification of Shout option:

        - Options, Futures and Other Derivatives, `OFOD <http://www-2.rotman.utoronto.ca/~hull/ofod/index.html>`_, J.C. Hull, 9ed, 2014. Prentice Hall, p609.
        - Shout Options, (Lecture 4, MFE5010 at NUS), `Lim Tiong Wee, 2001 <http://1drv.ms/1RDA4pm>`_
        - Shout Options - Introduction and Spreadsheet. `Samir Khan <http://investexcel.net/shout-options-excel/>`_
        - Binomial Tree pricing for Compound, Chooser, Shout. `Excel Tool <http://goo.gl/AdgcqY>`_
        - Numerical Pricing of Shout Options, `Lisa Yudaken, 2010 <http://1drv.ms/1M3j0lm>`_


        Examples
        --------
        **LT**

        >>> s = Stock(S0=50, vol=.3)
        >>> o = Shout(ref=s, right='call', K=52, T=2, rf_r=.05, desc='Example from internet excel spread sheet')
        >>> o.pxLT(nsteps=2)
        11.803171357

        >>> o.calc_px(method='LT', nsteps=2, keep_hist=True).px_spec.opt_tree
        ((11.803171356649463,), (0.0, 24.34243306821051), (0.0, 0.0, 39.10594001952546))

        >>> o.calc_px(method='LT', nsteps=2) # doctest: +ELLIPSIS
        Shout...px: 11.803171357...

        >>> from pandas import Series
        >>> steps = range(1,11)
        >>> O = Series([o.calc_px(method='LT', nsteps=s).px_spec.px for s in steps], steps)
        >>> O.plot(grid=1, title='LT Price vs nsteps') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>

        **MC**
        Note: When deg is too large, ``numpy.polyfit`` will give warnings: Polyfit may be poorly
        conditioned warnings.warn(msg, RankWarning)

        >>> s = Stock(S0=110, vol=.2, q=0.04)
        >>> o = Shout(ref=s, right='call', K=100, T=0.5, rf_r=.05, desc='See example in Notes [3]')
        >>> o.pxMC(nsteps=100, npaths=1000, keep_hist=True, rng_seed=314, deg=5)
        14.885085333

        >>> s = Stock(S0=36, vol=.2)
        >>> o = Shout(ref=s, right='put', K=40, T=1, rf_r=.2, desc="L. Yudaken\'s paper")
        >>> o.pxMC(nsteps=100, npaths=1000, keep_hist=True, rng_seed=0)
        4.1410886

        >>> o.calc_px(method='MC', nsteps=100, npaths=1000, keep_hist=True, rng_seed=0).px_spec  # doctest: +ELLIPSIS
        PriceSpec...px: 4.1410886...

        >>> from pandas import Series;  steps = [100 * i for i in range(1,21)]
        >>> O = Series([o.pxMC(nsteps=s, npaths=100, keep_hist=True, rng_seed=0, deg=0) for s in steps], steps)
        >>> O.plot(grid=1, title='Shout MC Price vs nsteps')# doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>

        :Authors:
            Mengyan Xie <xiemengy@gmail.com>,
            Hanting Li <hl45@rice.edu>,
            Yen-fei Chen <yensfly@gmail.com>
       """
        self.save2px_spec(deg=deg, **kwargs)
        return getattr(self, '_calc_' + self.px_spec.method.upper())()

    def _calc_BS(self):
        """ Internal function for option valuation.        """
        return self

    def _calc_LT(self):
        """ Internal function for option valuation.

        :Authors:
            Mengyan Xie <xiemengy@gmail.com>
        """
        _ = self.px_spec;   n, keep_hist = _.nsteps, _.keep_hist
        _ = self.ref;       S0, vol, q = _.S0, _.vol, _.q
        _ = self;           T, K, rf_r, net_r, sCP = _.T, _.K, _.rf_r, _.net_r, _.signCP
        _ = self._LT_specs(); u, d, p, df, dt = _['u'], _['d'], _['p'], _['df_dt'], _['dt']

        # Get the Price based on Binomial Tree
        S = S0 * d ** np.arange(n, -1, -1) * u ** np.arange(0, n + 1)  # terminal stock prices
        O = np.maximum(sCP * (S - K), 0)          # terminal option payouts

        # The end node of tree
        S_tree = (tuple([float(s) for s in S]),)  # use tuples of floats (instead of numpy.float)
        O_tree = (tuple([float(o) for o in O]),)

        for i in range(n, 0, -1):
            left = n - i + 1      # Left number until duration
            tleft = left * dt     # Time left until duration
            d1 = (0 + (rf_r + vol ** 2 / 2) * tleft) / (vol * np.sqrt(tleft))   # d1 and d2 from BS model
            d2 = d1 - vol * np.sqrt(tleft)

            # payoff of not shout
            O = df * ((1 - p) * O[:i] + p * O[1:])  # prior option prices (@time step=i-1)
            S = d * S[1:i+1]                        # spot tree: prior stock prices (@time step=i-1)

            # payoff of shout
            shout = sCP * S / np.exp(q * tleft) * Util.norm_cdf(sCP * d1) - \
                    sCP * S / np.exp(rf_r * tleft) * Util.norm_cdf(sCP * d2) + \
                    sCP * (S - K) / np.exp(rf_r * tleft)

            # final payoff is the maximum of shout or not shout
            payout = np.maximum(shout, 0)
            O = np.maximum(O, payout)

            S_tree = (tuple([float(s) for s in S]),) + S_tree
            O_tree = (tuple([float(o) for o in O]),) + O_tree

            out = O_tree[0][0]

        self.px_spec.add(px=float(Util.demote(O)), sub_method='binomial tree; Hull Ch.13',
                        ref_tree=S_tree if keep_hist else None, opt_tree=O_tree if keep_hist else None)

        return self

    def _calc_MC(self):
        """ Internal function for option valuation.

        :Authors:
            Yen-fei Chen <yensfly@gmail.com>
        """
        _ = self.px_spec;   n, m, rng_seed, keep_hist, deg = _.nsteps, _.npaths, _.rng_seed, _.keep_hist, _.deg
        _ = self.ref;       S0, vol, q = _.S0, _.vol, _.q
        _ = self;           T, K, rf_r, net_r, sCP = _.T, _.K, _.rf_r, _.net_r, _.signCP
        _ = self._LT_specs(); u, d, p, df, dt = _['u'], _['d'], _['p'], _['df_dt'], _['dt']

        np.random.seed(rng_seed)

        # option_px = np.zeros((n + 1, m) ,'d')
        S = np.zeros((n + 1, m), 'd')  # stock price matrix
        S[0, :] = S0  # initial value

        # stock price paths
        for t in range(1, n+1):
            random = scipy.stats.norm.rvs(loc=0, scale=1, size=m)
            S[t, :] = S[t-1, :] * np.exp((rf_r - vol**2 / 2) * dt + vol * random * np.sqrt(dt))

        option_px = np.maximum(sCP*(S - K), 0)  # payoff when not shout
        final_payoff = np.repeat(S[-1, :], n+1, axis=0).reshape(m, n + 1)
        shout_px = np.maximum(sCP*(final_payoff.transpose() - K), sCP * (S - K))

        for t in range (n - 1, -1, -1):  # valuation process is similar to American option
            rg = np.polyfit(S[t, :], df * np.array(option_px[t + 1, :]), deg) # regression at time t
            C = np.polyval(rg, S[t, :])  # continuation values
            # exercise decision: shout or not shout
            option_px[t, :] = np.where(shout_px[t, :] > C, shout_px[t, :], option_px[t+1,:] * df)

        self.px_spec.add(px=float(np.mean(option_px[0, :])), sub_method='Hull p.609')
        return self

    def _calc_FD(self):
        """ Internal function for option valuation.        """
        return self
