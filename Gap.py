import math
import numpy as np
from scipy import sparse

try: from qfrm.European import *  # production:  if qfrm package is installed
except:   from European import *  # development: if not installed and running from source

class Gap(European):
    """ Gap option class.

    Inherits all methods and properties of OptionValuation class.
    A gap option has a strike price, ``K1``, and a trigger price, ``K2``. The trigger price
    determines whether or not the gap option will have a nonzero payoff. The strike price
    determines the amount of the nonzero payoff. The strike price may be greater than or
    less than the trigger price.
    """

    def calc_px(self, K2=None, on = None, **kwargs):
        """ Wrapper function that calls appropriate valuation method.

        Parameters
        ----------
        K2: float
                Required. The secondary strike price in Gap option.
        kwargs : dict
            Keyword arguments (``method``, ``nsteps``, ``npaths``, ``keep_hist``, ``rng_seed``, ...)
            are passed to the parent. See ``European.calc_px()`` for details.


        Returns
        -------
        self : Gap
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object).


        Notes
        -----

        *References:*

        - Review Note Sample Excerpt. `Exotic Options. (Ch.14) <http://1drv.ms/1ONq7D1>`_
        - More Exotic Options (lecture slides), `Milica Cudina <http://1drv.ms/1ONpYiT>`_

        Examples
        --------

        **BS**

        >>> s = Stock(S0=500000, vol=.2)
        >>> o = Gap(ref=s, right='put', K=400000, T=1, rf_r=.05, desc='J.C.Hull, Example 26.1 on p.601')
        >>> o.pxBS(K2=350000)
        1895.688944397

        >>> s = Stock(S0=50, vol=.2)
        >>> o = Gap(ref=s, right='call', K=57, T=1, rf_r=.09)
        >>> o.pxBS(K2=50)
        2.266910325

        >>> o.calc_px(K2=50, method='BS').px_spec # doctest: +ELLIPSIS
        PriceSpec...px: 2.266910325...

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> o = Series([o.update(T=t).calc_px(K2=50, method='BS').px_spec.px for t in expiries], expiries)
        >>> o.plot(grid=1, title='BS Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>


        **LT**
        The price depends on the number of tree paths. ``n=22`` can give an answer in Hull's example

        >>> s = Stock(S0=500000, vol=.2,  q = 0)
        >>> o = Gap(ref=s, right='put', K=400000, T=1, rf_r=.05,  desc = 'HULL p. 601 Exp 26.1')
        >>> o.pxLT(K2=350000, nsteps = 3, on = (1000,)*4,)
        1839.1542566569999

        >>> s = Stock(S0=50, vol=.2,  q = 0)
        >>> o = Gap(ref=s, right='call', K=57, T=1, rf_r=.09)
        >>> o.pxLT(K2=50, nsteps = 3, on = (1000,)*4,)
        2.341912846

        >>> s = Stock(S0=50, vol=.2,  q = 0)
        >>> o = Gap(ref=s, right='put', K=57, T=1, rf_r=.09)
        >>> o.pxLT(K2=50, nsteps = 3, on = (1000,)*4,)
        4.4359904060000002

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> o = Series([o.update(T=t).pxLT(K2 = 50, nsteps=3, on = (1000,)*4,) for t in expiries], expiries)
        >>> o.plot(grid=1,title='Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>


        **MC**

        Due to slow convergence, iterations must be very high.
        For example, ``o.pxMC(K2=350000, nsteps=1000, npaths=100000, rng_seed=0)``
        yields (in 1-2 minutes) a Gap option price of 1839.844162184,
        which is similar to Gap's BS price of 1895.688944397. See J.C. Hull, Example 26.1 on p.601.

        >>> s = Stock(S0=500000, vol=.2)
        >>> o = Gap(ref=s, right='put', K=400000, T=1, rf_r=.05, desc='Hull p.601 Example 26.1')
        >>> o.pxMC(K2=350000, nsteps=10, npaths=10, rng_seed=10)
        7534.017075587

        >>> s = Stock(S0=500000, vol=.2)
        >>> o = Gap(ref=s, right='put', K=400000, T=1, rf_r=.05, desc='Hull p.601 Example 26.1')
        >>> o.pxMC(K2=350000, nsteps=1000, npaths=1000, rng_seed=0)  # better precision
        1362.367515835

        >>> from pandas import Series
        >>> Ts = range(1,101)
        >>> O = Series([o.update(T=T).pxMC(K2=350000, nsteps=3, npaths=2, rng_seed=1) for T in Ts], Ts)
        >>> O.plot(grid=1, title='Gap MC price vs expiry (in years)' + o.specs) # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>


        >>> s = Stock(S0=50, vol=.2)
        >>> o = Gap(ref=s, right='call', K=57, T=1, rf_r=.09)
        >>> o.pxMC(K2=50, nsteps=1000, npaths=1000, rng_seed=2)
        2.774272339

        The following example will generate px = 4.35362028... with nsteps = 100 and npaths = 250,
        which is similar to BS example above.

        >>> s = Stock(S0=50, vol=.2)
        >>> o = Gap(ref=s, right='put', K=57, T=1, rf_r=.09)
        >>> o.calc_px(K2=50, method='MC', nsteps=10, npaths=5, rng_seed=2).px_spec   # doctest: +ELLIPSIS
        PriceSpec...px: 6.803865574...


        **FD**
        FD methods require sufficient fine grids.  ``npath=100``, ``nsteps=100``
        can give the right answer in the verified example.

        >>> s = Stock(S0=500000, vol=.2)
        >>> o = Gap(ref=s, right='put', K=400000, T=1, rf_r=.05, desc='Hull p.601 Example 26.1')
        >>> o.pxFD(K2=350000,npaths=10, nsteps=10)
        5745.438398555

        >>> s = Stock(S0=50, vol=.2)
        >>> o = Gap(ref=s, right='call', K=50, T=1, rf_r=.09)
        >>> o.pxFD(K2=50, npaths=10, nsteps=10)
        6.811132138

        >>> s = Stock(S0=500000, vol=.2)
        >>> o = Gap(ref=s, right='put', K=400000, T=1, rf_r=.05, desc='Hull p.601 Example 26.1')
        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([o.update(T=t).pxFD(K2 = 350000, seed=1, npaths=10,nsteps=3) for t in expiries], expiries)
        >>> O.plot(grid=1, title='Price vs expiry (in years)-FD') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>


        :Authors:
            Yen-fei Chen <yensfly@gmail.com>,
            Thawda Aung,
            Mengyan Xie <xiemengy@gmail.com>,
            Runmin Zhang <z.runmin@gmail.com>
        """
        self.save2px_spec(on=on, K2=K2, **kwargs)
        return getattr(self, '_calc_' + self.px_spec.method.upper())()

    def _calc_BS(self):
        """ Internal function for option valuation.        See ``calc_px()`` for complete documentation.

        :Authors:
            Yen-fei Chen <yensfly@gmail.com>
        """
        _ = self.px_spec;       K2 = _.K2
        _ = self.ref;           S0, vol, q = _.S0, _.vol, _.q
        _ = self

        sp = European(clone=self, K=K2)._BS_specs()
        d1, d2 = sp['d1'], sp['d2']

        # if calc of both prices is cheap, do both and include them into Price object.
        # Price.px should always point to the price of interest to the user
        # Save values as basic data types (int, floats, str), instead of numpy.array
        N = Util.norm_cdf
        px_call = float(S0*math.exp(-q* _.T)*N(d1)-_.K*math.exp(-_.rf_r*_.T)*N(d2))
        px_put = float(-S0*math.exp(-q*_.T)*N(-d1)+_.K*math.exp(-_.rf_r*_.T)*N(-d2))
        px = px_call if _.signCP == 1 else px_put if _.signCP == -1 else None

        self.px_spec.add(px=px, sub_method='standard; Hull p.335', K2_BS_specs=sp, px_call=px_call, px_put=px_put)
        return self

    def _calc_LT(self):
        """ Internal function for option valuation.
        A binomial tree pricer of Gap options that takes the average results for given step sizes in NSteps.
        Large step sizes should be used for optimal accuracy but may take a minute or so.

        Returns
        -------------------------------------------
        self: Gap
        :param
                on : Numeric Vector
                A vector of number of steps to be used in binomial tree averaging, vector of positive intergers

        References :
        Hull, John C., Options, Futures and Other Derivatives, 9ed, 2014. Prentice Hall. ISBN 978-0-13-345631-8.
        http://www-2.rotman.utoronto.ca/~hull/ofod/index.html.
        Humphreys, Natalia. University of Dallas.

        :Authors:
            Thawda Aung
        """
        _ = self.px_spec;   n, on, K2 = _.nsteps, _.on, _.K2
        _ = self.ref;       S0, vol, q = _.S0, _.vol, _.q
        _ = self;           T, K, rf_r, net_r, sCP = _.T, _.K, _.rf_r, _.net_r, _.signCP

        # n = getattr(self.px_spec ,'nsteps', 5)
        assert len(on) > n , 'nsteps must be less than the vector on'

        px = np.zeros(n)
        for i in range(n):
            u1 = math.exp(vol * math.sqrt(T/ on[i]))
            d1 = 1/u1
            p1 = (math.exp( net_r * (T / on[i])) - d1) / (u1 - d1)
            leng = on[i]
            S = [S0 * d1**(leng - j) * u1**j for j in np.arange(0, on[i]+1)]
            O = np.zeros(len(S))
            for m in range(len(S)):
                if(sCP * (S[m] - K2) > 0 ):
                    O[m] = sCP * (S[m] - K)
                else:
                    O[m] = 0
            csl = np.cumsum([np.log(i) for i in np.arange(1, on[i] + 1)])
            a = np.array(0)
            a = np.insert(csl, 0, 0 )
            csl = a
            temp = [ csl[on[i]] - csl[j] - csl[ (leng - j) ] +
                     math.log(p1 ) * (j) + math.log( 1 - p1 ) * (leng - j) for j in np.arange(0, on[i] +1)]
            px[i] = math.exp(rf_r * -T) * sum([math.exp(temp[j]) * O[j] for j in np.arange(0, len(temp))])
            # tmp = [ csl[on[i] + 1] - csl -1 for i  ]
        Px = np.mean(px)
        self.px_spec.add(px=Px, sub_method='binomial_tree; Hull p.335', ref_tree=O, opt_tree=O )
        return self

    def _calc_MC(self):
        """ Internal function for option valuation.        See ``calc_px()`` for complete documentation.

        :Authors:
            Mengyan Xie <xiemengy@gmail.com>
        """
        _ = self.px_spec;   n, m, K2, rng_seed = _.nsteps, _.npaths, _.K2, _.rng_seed
        _ = self.ref;       S0, vol, q = _.S0, _.vol, _.q
        _ = self;           T, K, rf_r, net_r, sCP = _.T, _.K, _.rf_r, _.net_r, _.signCP

        dt = T / n
        df = np.exp(-rf_r * dt)
        np.random.seed(rng_seed)

        # Stock price paths
        S = S0 * np.exp(np.cumsum(np.random.normal((rf_r - 0.5 * vol ** 2) * dt, vol * np.sqrt(dt), (n + 1, m)), axis=0))
        S[0] = S0
        s2 = S

        V = np.maximum(_.signCP * (S - K2), 0)  # When the stock price is greater than K2
        payout = np.maximum(_.signCP * (s2 - K), 0)  # The payout is signCP * (S - K1)
        h = np.where(V > 0.0, payout, V)  # payout if V > 0.0, payout else 0.0

        for t in range(n-1, -1, -1): h[t,:] = h[t+1,:] * df   # Add the time value of each steps

        self.px_spec.add(px=float(np.mean(h[0, :])), sub_method='Hull p.601')
        return self

    def _calc_FD(self):
        """ Internal function for option valuation.        See ``calc_px()`` for complete documentation.

        :Authors:
            Runmin Zhang <z.runmin@gmail.com>
        """
        # Get parameters
        _ = self.px_spec;   n, m, K2 = _.nsteps, _.npaths, _.K2
        _ = self.ref;       S0, vol, q = _.S0, _.vol, _.q
        _ = self;           T, K, rf_r, net_r, sCP = _.T, _.K, _.rf_r, _.net_r, _.signCP

        S_max   = S0*2                                # Maximum stock price
        S_min   = 0.0                                 # Minimum stock price
        d_t     = T/(n-1)                  # Time step
        S_vec   = np.linspace(S_min,S_max,m)   # Initialize the possible stock price vector
        t_vec   = np.linspace(0,T,n)       # Initialize the time vector
        f_px    = np.zeros((m,n))     # Initialize the matrix. Hull's P482

        M = m - 1
        N = n-1

        # Set boundary conditions.
        f_px[:,-1]=S_vec

        if self.right=='call':
            # Payout at the maturity time
            init_cond = np.maximum((S_vec-K),0)*(S_vec>=K2)
            # Boundary condition
            upper_bound = 0
            # Calculate the current value
            lower_bound = np.maximum((S_vec[-1]-K),0)*(S_vec[-1]>=K2)*np.exp(-rf_r*(T-t_vec))
        elif self.right=='put':
            # Payout at the maturity time
            init_cond = np.maximum((K-S_vec),0)*(S_vec<=K2)
            # Boundary condition
            upper_bound = np.maximum((K-S_vec[0]),0)*(S_vec[0]<=K2)*np.exp(-rf_r*(T-t_vec))
            # Calculate the current value
            lower_bound = 0


        #Generate Matrix B in http://www.goddardconsulting.ca/option-pricing-finite-diff-implicit.html
        j_list = np.arange(0,M+1)
        a_list = 0.5*d_t*(net_r*j_list-vol**2*j_list**2)
        b_list = 1+d_t*(vol**2*j_list**2 + rf_r)
        c_list = 0.5*d_t*(-net_r*j_list-vol**2*j_list**2)

        data = (a_list[2:M],b_list[1:M],c_list[1:M-1])
        B=sparse.diags(data,[-1,0,1]).tocsc()

        # Using Implicit method to solve B-S equation
        f_px[:,N] = init_cond
        f_px[0,:] = upper_bound
        f_px[M,:]=lower_bound
        Offset = np.zeros(M-1)
        for idx in np.arange(N-1,-1,-1):
            Offset[0] = -a_list[1]*f_px[0,idx]
            Offset[-1] = -c_list[M-1]*f_px[M,idx]
            f_px[1:M,idx]=sparse.linalg.spsolve(B,f_px[1:M,idx+1]+Offset)
            f_px[:,-1] = init_cond
            f_px[0,:] = upper_bound
            f_px[-1,:]=lower_bound

        self.px_spec.add(px=float(np.interp(S0,S_vec,f_px[:,0])), sub_method='Implicit Method')
        return self

