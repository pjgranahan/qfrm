import scipy.stats
import numpy as np
import scipy.special
import math

try: from qfrm.European import *  # production:  if qfrm package is installed
except:   from European import *  # development: if not installed and running from source


class Barrier(European):
    """ `Barrier <https://en.wikipedia.org/wiki/Barrier_option>`_ exotic option class.

    Payoff is a function depends on underlying security's crossing a specified barrier price level ``H`` until expiry.
    A similar option is Parisian, where the life of the option depends not only on ``H``,
    but also on how long the unnderlying's price stays above or below ``H``.
    See OFOD, J.C.Hull, 9ed, 2014, pp.604-606, pp.640-643.
    """

    def calc_px(self, H=None, knock=None, dir=None, **kwargs):
        """ Wrapper function that calls specified option valuation method.

        Parameters
        ----------
        H : float, >0
                Barrier price level which triggers option's exercisability into existence.
        knock : {'up', 'down'}
            Indicates what triggers the life/death of a barrier option:
            whether the price of the underlying crosses ``H`` from above (``down``) or from below (``up``).
        dir : {'in', 'out'}
            ``out`` indicates that option ceases to exist (or comes to life with ``in``),
            if the price of the underlying reaches a specified barrier price level ``H``.
        kwargs : dict
            Keyword arguments (``method``, ``nsteps``, ``npaths``, ``keep_hist``, ``rng_seed``, ...)
            are passed to the parent. See ``European.calc_px()`` for details.


        Returns
        -------
        self : Barrier
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object).


        Notes
        ---------

        *References:*

        - Barrier Option Calculator. `FinTools.com <http://www.fintools.com/resources/online-calculators/exotics-calculators/exoticscalc-barrier>`_
        -  DerivaGem software (accompanies J.C.Hull's `OFOD <http://www-2.rotman.utoronto.ca/~hull/ofod/index.html>`_ textbook)
        - See OFOD, J.C.Hull, 9ed, 2014, pp.604-606, pp.640-643.
        - Barrier Option Pricing. `CoggIt.com, Free Tools. <http://www.coggit.com/freetools>`_
        - Valuation of Up-In and Up-Out barrier options. `Online option pricer. <http://www.infres.enst.fr/~decreuse/pricer/en/index.php?page=barriereUp.html>`_
        - Binomial Trees for Barrier Options (Ch.8, FCFE Course, NTU) `Jr-Yan Wang, 2015 <http://goo.gl/zcPhJe>`_
        - `In-Out Parity <http://www.iam.uni-bonn.de/people/ankirchner/lectures/OP_WS1314/OP_chap_nine.pdf>`_

        Examples
        ---------

        **BS**  Next example is similar to exercise 26.19 on p.621 from OFOD, J.C.Hull, 2014,
        but the underlying is an equity, not a stock.
        The following barrier call option is valued at 0.693415883 by DerivaGem Software.

        >>> from qfrm import *
        >>> s = Stock(S0=19, vol=.4)
        >>> o = Barrier(ref=s, right='call', K=20, T=.25, rf_r=.05, desc='Ex. 26.19, p621, DerivaGem price is 0.6934')
        >>> o.pxBS(H=18, knock='down', dir='out')
        0.693415883

        Here is another down-and-out barrier option. We display all option object's specifications.

        >>> s = Stock(S0=50, vol=.25)
        >>> o = Barrier(ref=s, right='call', K=45, T=2, rf_r=.1, desc='DerivaGem price is 14.47441215')
        >>> o.calc_px(method='BS', H=35, knock='down', dir='out')   # doctest: +ELLIPSIS
        Barrier...px: 14.4744148...

        >>> o.pxBS(H=35, knock='down', dir='in')  # DerivaGem's price is 0.295501836
        0.295502923

        >>> o.pxBS(H=35, knock='up', dir='in')  # DerivaGem's price is 0.295501836
        14.769917723

        >>> o.pxBS(H=35, knock='up', dir='out')  # DerivaGem's price is 0.0
        0.0

        >>> s = Stock(S0=35, vol=.1, q=.1)
        >>> o = Barrier(ref=s, right='put', K=45, T=2.5, rf_r=.1, desc='up and out put')
        >>> o.pxBS(H=50,knock='up',dir='out')
        7.90173205

        >>> s = Stock(S0=85, vol=.35, q=.05)
        >>> o = Barrier(ref=s, right='call', K=80, T=.5, rf_r=.05, desc='up and in call')
        >>> o.pxBS(H=90, knock='up', dir='in')
        10.536077751

        >>> from pandas import Series
        >>> expiries = range(1, 11)  # in years
        >>> O = Series([o.update(T=t).pxBS(H=35, knock='down', dir='out') for t in expiries], expiries)
        >>> O.plot(grid=1, title='Barrier option price vs expiry (in years)')    # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>


        **LT** SEE NOTES for verification of examples.  Run with ``nsteps`` > 1000 for accurate results.

        >>> s = Stock(S0=95, vol=.25)
        >>> o = Barrier(ref=s, right='put', K=100, T=1, rf_r=.1, desc='down and in put')
        >>> o.pxLT(H=90, knock='down', dir='in', nsteps=10)
        6.962523053

        >>> o.px_spec     # doctest: +ELLIPSIS
        PriceSpec...px: 6.962523053...


        >>> s = Stock(S0=95, vol=.25)
        >>> o = Barrier(ref=s, right='call', K=100, T=2, rf_r=.1, desc='down and out call')
        >>> o.pxLT(H=87, knock='down', dir='out', nsteps=10)
        13.645434569

        >>> s = Stock(S0=95, vol=.25)
        >>> o = Barrier(ref=s, right='put', K=100, T=2, rf_r=.1, desc='up and out put')
        >>> o.pxLT(nsteps=10, H=105, knock='up', dir='out')
        3.513294523

        >>> s = Stock(S0=95, vol=.25)
        >>> o = Barrier(ref=s, right='call', K=100, T=2, rf_r=.1, desc='up and in call')
        >>> o.pxLT(H=105, knock='up', dir='in', nsteps=10)
        20.040606034

        Example of option price convergence (LT method)

        >>> s = Stock(S0=95, vol=.25)
        >>> o = Barrier(ref=s, right='call', K=100, T=2, rf_r=.1, desc='up and in call')
        >>> from pandas import Series;  steps = range(3,100)
        >>> O = Series([o.pxLT(H=105, knock='up', dir='in', nsteps=s) for s in steps], steps)
        >>> O.plot(grid=1, title='Price vs Steps')       # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>


        **MC** All examples below can be verified with DerivaGem software.
        *Note*: you would like to get the close results you would have to use ``nsteps = 500``, ``npaths = 10000``

        >>> s = Stock(S0=50, vol=.3)
        >>> o = Barrier(ref=s, right='put', K=50, T=1, rf_r=.1, desc='DerviaGem Up and Out Barrier')
        >>> o.pxMC(H=60, knock='up', dir='out', nsteps=100, rng_seed=0, npaths=100)
        3.964058729

        >>> s = Stock(S0=50, vol=.3)
        >>> o = Barrier(ref=s, right='call', K=50, T=1, rf_r=.1, desc='Up and in call')
        >>> o.pxMC(H=60, knock='up', dir='in', rng_seed=0, nsteps=500, npaths=100)
        7.173989151

        >>> s = Stock(S0=50, vol=.25)
        >>> o = Barrier(ref=s, right='call', K=45, T=2, rf_r=.3, desc='down and in call')
        >>> o.pxMC(H=35, knock='down', dir='in', rng_seed=4, nsteps=500, npaths=300)
        0.041420563

        :Authors:
            Scott Morgan,
            Hanting Li <hl45@rice.edu>,
            Thawda Aung <thawda.aung1@gmail.com>
       """

        if H is None:
            H = getattr(self.px_spec, 'H', None)
            assert H is not None, 'Assert failed: required input H'

        if knock is None:
            knock = getattr(self.px_spec, 'knock', None)
            assert knock is not None, 'Assert failed: required input knock'

        if dir is None:
            dir = getattr(self.px_spec, 'dir', None)
            assert dir is not None, 'Assert failed: required input dir'

        self.save2px_spec(knock=knock, dir=dir, H=H, **kwargs)
        return getattr(self, '_calc_' + self.px_spec.method.upper())()

    def _calc_BS(self):
        """ Internal function for option valuation. See ``calc_px()`` for complete documentation.

        :Authors:
            Hanting Li <hl45@rice.edu>
        """

        H = self.px_spec.H
        dir = self.px_spec.dir  # direction
        knock = self.px_spec.knock
        _ = self._BS_specs()
        T, K, rf_r, right, S0, net_r, vol, q = self.T, self.K, self.rf_r, self.right, self.ref.S0, self.net_r, self.ref.vol, self.ref.q


        # Compute Parameters
        N = Util.norm_cdf
        d1 = (np.log(S0/K) + (rf_r-q+(vol**2)/2)*T)/(vol*np.sqrt(T))
        d2 = d1 - vol*np.sqrt(T)

        c = S0*np.exp(-q*T)*N(d1) - K*np.exp(-rf_r*T)*N(d2)
        p = K*np.exp(-rf_r*T)*N(-d2) - S0*np.exp(-q*T)*N(-d1)

        l = (rf_r-q+(vol**2)/2)/(vol**2)
        y = np.log((H**2)/(S0*K))/(vol*np.sqrt(T)) + l*vol*np.sqrt(T)
        x1 = np.log(S0/H)/(vol*np.sqrt(T)) + l*vol*np.sqrt(T)
        y1 = np.log(H/S0)/(vol*np.sqrt(T)) + l*vol*np.sqrt(T)

        # Consider Call Option
        # Two Situations: H<=K vs H>K
        if (right == 'call'):
            if (H<=K):
                cdi = S0*np.exp(-q*T)*((H/S0)**(2*l))*N(y) - \
                      K*np.exp(-rf_r*T)*((H/S0)**(2*l-2))*N(y-vol*np.sqrt(T))
                cdo = S0*N(x1)*np.exp(-q*T) - K*np.exp(-rf_r*T)*N(x1-vol*np.sqrt(T)) \
                      - S0*np.exp(-q*T)*((H/S0)**(2*l))*N(y1) + \
                      K*np.exp(-rf_r*T)*((H/S0)**(2*l-2))*N(y1-vol*np.sqrt(T))
                cdo = c - cdi
                cuo = 0
                cui = c
            else:
                cdo = S0*N(x1)*np.exp(-q*T) - K*np.exp(-rf_r*T)*N(x1-vol*np.sqrt(T)) \
                      - S0*np.exp(-q*T)*((H/S0)**(2*l))*N(y1) + \
                      K*np.exp(-rf_r*T)*((H/S0)**(2*l-2))*N(y1-vol*np.sqrt(T))
                cdi = c - cdo
                cui = S0*N(x1)*np.exp(-q*T) -\
                      K*np.exp(-rf_r*T)*N(x1-vol*np.sqrt(T)) - \
                      S0*np.exp(-q*T)*((H/S0)**(2*l))*(N(-y)-N(-y1)) + \
                      K*np.exp(-rf_r*T)*((H/S0)**(2*l-2))*(N(-y+vol*np.sqrt(T))- N(-y1+vol*np.sqrt(T)))
                cuo = c - cui
        # Consider Put Option
        # Two Situations: H<=K vs H>K
        else:
            if (H>K):
                pui = -S0*np.exp(-q*T)*((H/S0)**(2*l))*N(-y) + K*np.exp(-rf_r*T)*((H/S0)**(2*l-2))*N(-y+vol*np.sqrt(T))
                puo = p - pui
                pdo = 0
                pdi = p
            else:
                puo = -S0*N(-x1)*np.exp(-q*T) + \
                      K*np.exp(-rf_r*T)*N(-x1+vol*np.sqrt(T)) + \
                      S0*np.exp(-q*T)*((H/S0)**(2*l))*N(-y1) - \
                      K*np.exp(-rf_r*T)*((H/S0)**(2*l-2))*N(-y1+vol*np.sqrt(T))
                pui = p - puo
                pdi = -S0*N(-x1)*np.exp(-q*T) +\
                      K*np.exp(-rf_r*T)*N(-x1+vol*np.sqrt(T)) + \
                      S0*np.exp(-q*T)*((H/S0)**(2*l))*(N(y)-N(y1)) - \
                      K*np.exp(-rf_r*T)*((H/S0)**(2*l-2))*(N(y-vol*np.sqrt(T)) -\
                                                                      N(y1-vol*np.sqrt(T)))
                pdo = p - pdi

        if (right == 'call'):
            if (knock == 'down'):
                if (dir == 'in'):
                    px = cdi
                else:
                    px = cdo
            else:
                if (dir == 'in'):
                    px = cui
                else:
                    px = cuo
        else:
            if (knock == 'down'):
                if (dir == 'in'):
                    px = pdi
                else:
                    px = pdo
            else:
                if (dir == 'in'):
                    px = pui
                else:
                    px = puo

        self.px_spec.add(px=float(px), sub_method='standard; Hull p.604')

        return self

    def _calc_LT(self):
        """ Internal function for option valuation.  See ``calc_px()`` for complete documentation.

        :Authors:
            Scott Morgan
        """

        if self.px_spec.knock == 'down':
            s = 1
        elif self.px_spec.knock == 'up':
            s = -1

        keep_hist = self.px_spec.keep_hist #getattr(self.px_spec, 'keep_hist', False)
        n = self.px_spec.nsteps  # getattr(self.px_spec, 'nsteps', 3)
        H = self.px_spec.H
        dir = self.px_spec.dir  # direction
        knock = self.px_spec.knock
        _ = self._LT_specs()
        T, K, r, right, S0 = self.T, self.K, self.rf_r, self.right, self.ref.S0

        S = S0 * _['d'] ** np.arange(n, -1, -1) * _['u'] ** np.arange(0, n + 1)  # terminal stock prices
        S2 = np.maximum(s*(S - H),0) # Find where crossed the barrier
        S2 = np.minimum(S2,1)  # 0 when across the barrier, 1 otherwise
        O = np.maximum(self.signCP * (S - K), 0)
        O = O * S2        # terminal option payouts
        S_tree = (tuple([float(s) for s in S]),)  # use tuples of floats (instead of np.float)
        O_tree = (tuple([float(o) for o in O]),)

        for i in range(n, 0, -1):
            O = _['df_dt'] * ((1 - _['p']) * O[:i] + ( _['p']) * O[1:])  #prior option prices (@time step=i-1)
            S = _['d'] * S[1:i+1]                   # prior stock prices (@time step=i-1)
            S2 = np.maximum(s*(S - H),0)
            S2 = np.minimum(S2,1)
            O = O * S2
            S_tree = (tuple([float(s) for s in S]),) + S_tree
            O_tree = (tuple([float(o) for o in O]),) + O_tree

        out_px = float(Util.demote(O))


        if dir == 'out':

            self.px_spec.add(px=out_px, sub_method='binomial tree; biased',
                        ref_tree = S_tree if keep_hist else None, opt_tree = O_tree if keep_hist else None)

            return self

        k = int(math.ceil(np.log(K/(S0*_['d']**n))/np.log(_['u']/_['d'])))
        h = int(math.floor(np.log(H/(S0*_['d']**n))/np.log(_['u']/_['d'])))
        l = list(map(lambda j: scipy.special.binom(n,n-2*h+j)*(_['p']**j)*((1-_['p'])**(n-j))* \
                               (S0*(_['u']**j)*(_['d']**(n-j))-K), range(k,n+1)))
        down_in_call = np.exp(-r*T)*sum(l)



        if dir == 'in' and right == 'call' and knock == 'down':
            self.px_spec.add(px=down_in_call, sub_method='combinatorial', ref_tree = None, opt_tree =  None)

        elif dir == 'in' and right == 'call' and knock == 'up':

            o = European(ref=self.ref, right='call', K=K, T=T, rf_r=r, desc='reference')
            call_px = o.calc_px(method='BS').px_spec.px   # save interim results to self.px_spec. Equivalent to repr(o)
            in_px = call_px - out_px
            self.px_spec.add(px=in_px, sub_method='in out parity', ref_tree = None, opt_tree =  None)


        elif dir == 'in' and right == 'put' and knock == 'up':

            o = European(ref=self.ref, right='put', K=K, T=T, rf_r=r, desc='reference')
            put_px = o.pxBS()   # save interim results to self.px_spec. Equivalent to repr(o)
            in_px = put_px - out_px
            self.px_spec.add(px=in_px, sub_method='in out parity', ref_tree = None, opt_tree =  None)

        elif dir == 'in' and right == 'put' and knock == 'down':

            o = European(ref=self.ref, right='put', K=K, T=T, rf_r=r, desc='reference')
            put_px = o.pxBS()   # save interim results to self.px_spec. Equivalent to repr(o)
            in_px = put_px - out_px
            self.px_spec.add(px=in_px, sub_method='in out parity', ref_tree = None, opt_tree =  None)

        return self

    def _calc_MC(self):
        """ Internal function for option valuation.   See ``calc_px()`` for complete documentation.

        :Authors:
            Thawda Aung <thawda.aung1@gmail.com>
        """

        _ = self;           T, K, rf_r, net_r, right, ref = _.T, _.K, _.rf_r, _.net_r, _.right, _.ref
        _ = self.ref;       spot, sigma, q = _.S0, _.vol, _.q
        _ = self.px_spec;   rng_seed, dir, knock, H, n, m = _.rng_seed, _.dir, _.knock, _.H, _.nsteps, _.npaths
        Sb = H

        np.random.seed(rng_seed)

        def AssetPaths(spot, r, sigma, T, n, m):
            sim_paths = np.zeros((m, n + 1))
            sim_paths[:,0] = spot
            dt = T / n
            for i in range(int(m)):
                for j in range(1,int(n +1) ):
                    wt = np.random.randn()
                    sim_paths[i,j] = sim_paths[i, j -1] * np.exp((r - 0.5 * sigma **2) * dt + sigma * np.sqrt(dt) * wt)

            return(sim_paths)

        def barrierCrossing2(spot, sb, path):
            if sb < spot:
                temp = [1 if i <= sb else 0 for i in path[0,]]
                if sum(temp) > 1: knocked = 1
                else: knocked = 0
            else:
                temp = [ 1 if i >= sb else 0 for i in path[0,]]
                if sum(temp) > 1: knocked = 1
                else: knocked = 0
            return(knocked)

        def knockedout_put(spot, Sb, K, r, T, sigma, n, m):
            payoff = np.zeros((m, 1))
            for i in range(m):
                path = AssetPaths( spot, r, sigma, T, n, 1)
                knocked = barrierCrossing(spot, Sb, path)
                if knocked == 0: payoff[i] = max(0, K - path[0, n ])
            return(scipy.stats.norm.fit(np.exp(-r*T) * payoff)[0])

        def knockedin_put(spot, Sb, K, r, T, sigma, n, m):
            payoff = np.zeros((m, 1))
            for i in range(m):
                path = AssetPaths( spot, r, sigma, T, n, 1)
                knocked = barrierCrossing(spot, Sb, path)
                if knocked == 1: payoff[i] = max(0, K - path[0,n])
            return(scipy.stats.norm.fit(np.exp(-r*T) * payoff)[0])

        def barrierCrossing(spot, sb, path):
            if sb < spot:
                temp = [1 if i <= sb else 0 for i in path[0,]]
                if sum(temp) >= 1: knocked = 1
                else: knocked = 0
            else:
                temp = [ 1 if i >= sb else 0 for i in path[0,]]
                if sum(temp) >= 1: knocked = 1
                else: knocked = 0
            return(knocked)


        def knockout_call(spot, Sb, K, r, T, sigma, n, m):
            payoff = np.zeros((m, 1))

            for i in range(m):
                path = AssetPaths(spot, r, sigma, T, n, 1 )
                knocked = barrierCrossing(spot, Sb, path)
                if knocked == 0: payoff[i] = max(0, path[0, n] - K)
            return(scipy.stats.norm.fit(np.exp(-r*T) * payoff)[0])


        def knockin_call(spot, Sb, K, r, T, sigma, n, m):
            payoff = np.zeros((m, 1))

            for i in range(m):
                path = AssetPaths(spot, r, sigma, T, n, 1 )
                knocked = barrierCrossing(spot, Sb, path)
                if knocked == 1: payoff[i] = max(0, path[0, n] - K)
            return(scipy.stats.norm.fit(np.exp(-r*T) * payoff)[0])

        px = 0
        if dir == 'out' and right == 'call' and knock == 'down':
            if spot > Sb: px = knockout_call(spot, Sb, K, net_r, T, sigma, n, m)
            else: px = European(ref=ref, right='call', K=K, T=T, rf_r=rf_r, desc='reference').pxBS()

        elif dir == 'in' and right == 'call' and knock == 'down':
            if spot > Sb: px = knockin_call(spot, Sb, K, net_r, T, sigma, n, m)
            else: px = European(ref=ref, right='call', K=K, T=T, rf_r=rf_r, desc='reference').pxBS()

        elif dir == 'in' and right == 'put' and knock == 'down':
            if spot > Sb: px = knockedin_put(spot, Sb, K, net_r, T, sigma, n, m)
            else: px = European(ref=ref, right='put', K=K, T=T, rf_r=rf_r, desc='reference').pxBS()

        elif dir == 'out' and right == 'put' and knock == 'down':
            if spot > Sb: px = knockedout_put(spot, Sb, K, net_r, T, sigma, n, m)
            else: px = European(ref=ref, right='put', K=K, T=T, rf_r=rf_r, desc='reference').pxBS()

        elif dir == 'out' and right == 'call' and knock == 'up':
            if spot < Sb: px = knockout_call(spot, Sb, K, net_r, T, sigma, n, m)
            else: px = European(ref=ref, right='call', K=K, T=T, rf_r=rf_r, desc='reference').pxBS()

        elif dir == 'in' and right == 'call' and knock == 'up':
            if spot < Sb: px = knockin_call(spot, Sb, K, net_r, T, sigma, n, m)
            else: px = European(ref=ref, right='call', K=K, T=T, rf_r=rf_r, desc='reference').pxBS()

        elif dir == 'in' and right == 'put' and knock == 'up':
            if spot < Sb: px = knockedin_put(spot, Sb, K, net_r, T, sigma, n, m)
            else: px = European(ref=ref, right='put', K=K, T=T, rf_r=rf_r, desc='reference').pxBS()

        elif dir == 'out' and right == 'put' and knock == 'up':
            if spot < Sb: px = knockedout_put(spot, Sb, K, net_r, T, sigma, n, m)
            else: px = European(ref=ref, right='put', K=K, T=T, rf_r=rf_r, desc='reference').pxBS()

        self.px_spec.add(px=float(px), sub_method='Monte Carlo Simulation')


        return self

    def _calc_FD(self):
        """ Internal function for option valuation.    See ``calc_px()`` for complete documentation.     """
        return self

