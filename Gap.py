import math
import numpy as np
from scipy import sparse


try: from qfrm.OptionValuation import *  # production:  if qfrm package is installed
except:   from OptionValuation import *  # development: if not installed and running from source


class Gap(OptionValuation):
    """ Gap option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def __init__(self, on = None, *args, **kwargs):

        """ Constructor for Gap option class

        Passes additional arguments to OptionValuation class

        Parameters
        ---------------------------------------
        on : Numeric Vector
                A vector of number of steps to be used in binomial tree averaging, vector of positive intergers
        dir : string
                'in' or 'out'
        *args, **kwargs: varies
                arguments required by the constructor of OptionValuation class


        :Authors:
            Thawda Aung
       """

        self.on = on

        super().__init__(*args,**kwargs)

    def calc_px(self, K2=None, method='BS', nsteps=None, npaths=None, keep_hist=False, seed=None):
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

        Returns
        -----------------------------------------------------
        Gap
            Returned object contains specifications and calculated price in embedded ``PriceSpec`` object.

        Notes
        -----------------------------------------------------
        A gap option has a strike price, ``K1``, and a trigger price, ``K2``. The trigger price
        determines whether or not the gap option will have a nonzero payoff. The strike price
        determines the amount of the nonzero payoff. The strike price may be greater than or
        less than the trigger price.

        Examples
        --------------------------------------------------------

        **BS Examples**

        >>> s = Stock(S0=500000, vol=.2)
        >>> o = Gap(ref=s, right='put', K=400000, T=1, rf_r=.05, desc='Hull p.601 Example 26.1')
        >>> o.pxBS(K2=350000)
        1895.688944397

        >>> s = Stock(S0=50, vol=.2)
        >>> o = Gap(ref=s, right='call', K=57, T=1, rf_r=.09)
        >>> o.pxBS(K2=50)
        2.266910325

        >>> o.calc_px(K2=50, method='BS').px_spec # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px: 2.266910325...

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> o = Series([o.update(T=t).calc_px(K2=50, method='BS').px_spec.px for t in expiries], expiries)
        >>> o.plot(grid=1, title='BS Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()

        **LT Examples**
        The price depends on the number of tree paths. ``n=22`` can give an answer in Hull's example

        >>> s = Stock(S0=500000, vol=.2,  q = 0)
        >>> o = Gap(ref=s, right='put', K=400000, T=1, rf_r=.05, on = (90000,)*4, desc = 'HULL p. 601 Exp 26.1')
        >>> o.pxLT(K2=350000, nsteps = 3)
        1895.8012967929999

        >>> s = Stock(S0=50, vol=.2,  q = 0)
        >>> o = Gap(ref=s, right='call', K=57, T=1, rf_r=.09, on = (90000,)*4)
        >>> o.pxLT(K2=50, nsteps = 3)
        2.2749024279999999

        >>> s = Stock(S0=50, vol=.2,  q = 0)
        >>> o = Gap(ref=s, right='put', K=57, T=1, rf_r=.09, on = (90000,)*4)
        >>> o.pxLT(K2=50, nsteps = 3)
        4.3689799980000004

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> o = Series([o.update(T=t).pxLT(K2 = 50, nsteps=3) for t in expiries], expiries)
        >>> o.plot(grid=1,title='Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>


        **MC Examples**
        Because different number of seed, ``npaths`` and ``nsteps`` will influence the option price.
        The result of MC method may not as accurate as ``BS`` and ``LT`` methods.

        The following example will generate ``px = 1895.64429636`` with ``nsteps = 998`` and ``npaths = 1000``,
        which can be verified by Hull p.601 Example 26.1
        However, for the purpose if fast runtime, I use ``nstep = 10`` and ``npaths = 10`` in all following examples,
        whose result does not match verification.

        >>> s = Stock(S0=500000, vol=.2)
        >>> o = Gap(ref=s, right='put', K=400000, T=1, rf_r=.05, desc='Hull p.601 Example 26.1')
        >>> o.pxMC(K2=350000, seed=10, npaths=50, nsteps=10)
        2283.059032245

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([o.update(T=t).pxMC(K2 = 350000,seed=1, npaths=2,nsteps=3) for t in expiries], expiries)
        >>> O.plot(grid=1, title='Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()

        The following example will generate px = 2.258897568 with nsteps = 90 and npaths = 101, \
        which is similar to BS example.

        >>> s = Stock(S0=50, vol=.2)
        >>> o = Gap(ref=s, right='call', K=57, T=1, rf_r=.09)
        >>> o.pxMC(K2=50, seed=2, npaths=10, nsteps=50)
        1.342195428

        The following example will generate px = 4.35362028... with nsteps = 100 and npaths = 250, \
        which is similar to BS example.

        >>> s = Stock(S0=50, vol=.2)
        >>> o = Gap(ref=s, right='put', K=57, T=1, rf_r=.09)
        >>> o.calc_px(K2=50, method='MC',seed=2, npaths=10, nsteps=50).px_spec
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px: 3.672556646...


        **FD Examples**
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
        >>> import matplotlib.pyplot as plt
        >>> plt.show()

        See Also
        ---------------------------------------------------------
        [1] http://www.actuarialbookstore.com/samples/3MFE-BRE-12FSM%20Sample%20_4-12-12.pdf
        [2] https://www.ma.utexas.edu/users/mcudina/Lecture14_3_4_5.pdf

        :Authors:
            Yen-fei Chen <yensfly@gmail.com>,
            Thawda Aung,
            Mengyan Xie <xiemengy@gmail.com>,
            Runmin Zhang <z.runmin@gmail.com>
        """
        self.K2 = float(K2)
        self.seed0 = seed
        return super().calc_px(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)

    def _calc_BS(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Yen-fei Chen <yensfly@gmail.com>
        """

        _ = self
        d1 = (np.log(_.ref.S0 / _.K2) + (_.rf_r - _.ref.q + _.ref.vol ** 2 / 2.) * _.T)/(_.ref.vol * np.sqrt(_.T))
        d2 = d1 - _.ref.vol * np.sqrt(_.T)

        # if calc of both prices is cheap, do both and include them into Price object.
        # Price.px should always point to the price of interest to the user
        # Save values as basic data types (int, floats, str), instead of numpy.array
        N = Util.norm_cdf
        px_call = float(_.ref.S0*np.exp(-_.ref.q* _.T)*N(d1)-_.K*np.exp(-_.rf_r*_.T)*N(d2))
        px_put = float(-_.ref.S0*np.exp(-_.ref.q*_.T)*N(-d1)+_.K*np.exp(-_.rf_r*_.T)*N(-d2))
        px = px_call if _.signCP == 1 else px_put if _.signCP == -1 else None

        self.px_spec.add(px=px, sub_method='standard; Hull p.335', px_call=px_call, px_put=px_put, d1=d1, d2=d2)

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

        .. sectionauthor:: Thawda Aung

        References :
        Hull, John C., Options, Futures and Other Derivatives, 9ed, 2014. Prentice Hall. ISBN 978-0-13-345631-8.
        http://www-2.rotman.utoronto.ca/~hull/ofod/index.html.
        Humphreys, Natalia. University of Dallas.


        """
        n = getattr(self.px_spec ,'nsteps', 5)
        assert len(self.on) > n , 'nsteps must be less than the vector on'
        _ = self
        para = self.LT_specs(n)
        vol = _.ref.vol
        ttm = _.T
        on = self.on
        r = _.rf_r
        q = _.ref.q
        S0 = _.ref.S0
        sign = _.signCP
        K2 = _.K2
        K = _.K
        px = np.zeros(n)
        for i in range(n):
            u1 = math.exp(vol * math.sqrt(ttm/ on[i]))
            d1 = 1/u1
            p1 = (math.exp( (r-q) * (ttm / on[i])) - d1 ) / (u1 - d1)
            leng = on[i]
            S = [S0 * d1**(leng - j ) * u1**(j) for j in np.arange(0 , on[i]+1)]
            O = np.zeros(len(S))
            for m in range(len(S)):
                if(sign * (S[m] - K2) > 0 ):
                    O[m] = sign* (S[m] - K)
                else:
                    O[m] = 0
            csl = np.cumsum([np.log(i) for i in np.arange(1,on[i] + 1)])
            a = np.array(0)
            a = np.insert(csl , 0 , 0 )
            csl = a
            temp = [ csl[on[i]] - csl[j] - csl[ (leng - j) ] +
                     math.log(p1 ) * (j) + math.log( 1 - p1 ) * (leng - j) for j in np.arange(0 , on[i] +1)]
            px[i] = math.exp(r * -ttm) * sum([math.exp(temp[j]) * O[j]  for j in np.arange(0,len(temp))])
            # tmp = [ csl[on[i] + 1] - csl -1 for i  ]
        Px = np.mean(px)
        self.px_spec.add(px=Px, sub_method='binomial_tree; Hull p.335', L_Tspecs=para, ref_tree = O, opt_tree = O )
        return self

    def _calc_MC(self):

        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Mengyan Xie <xiemengy@gmail.com>
        """
        # Get parameters of steps and paths
        n_steps = getattr(self.px_spec, 'nsteps', 3)
        n_paths = getattr(self.px_spec, 'npaths', 3)
        _ = self

        dt = _.T / n_steps
        df = np.exp(-_.rf_r * dt)
        np.random.seed(_.seed0)

        # Stock price paths
        S = _.ref.S0 * np.exp(np.cumsum(np.random.normal((_.rf_r - 0.5 * _.ref.vol ** 2) * dt,\
                                                         _.ref.vol * np.sqrt(dt), (n_steps + 1, n_paths)), axis=0))
        S[0] = _.ref.S0
        s2 = S

        # When the stock price is greater than K2
        V = np.maximum(_.signCP * (S - _.K2), 0)

        # The payout is signCP * (S - K1)
        payout = np.maximum(_.signCP * (s2 - _.K), 0) #payout
        h = np.where(V > 0.0, payout, V) # payout if V > 0.0, payout else 0.0

        # Add the time value of each steps
        for t in range(n_steps-1, -1, -1):
            h[t,:] = h[t+1,:] * df

        self.px_spec.add(px=float(np.mean(h[0,:])), sub_method='Hull p.601')
        return self

    def _calc_FD(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Runmin Zhang <z.runmin@gmail.com>
        """
        # Get parameters
        time_steps = getattr(self.px_spec, 'nsteps', 5)
        px_paths = getattr(self.px_spec, 'npaths', 5)

        # Verify all the inputs are meaning full
        assert self.right in ['call', 'put'], 'right must be "call" or "put" '
        assert self.ref.vol > 0, 'vol must be >=0'
        assert self.K > 0, 'K must be > 0'
        assert self.K2 > 0, 'K2 must be > 0'
        assert self.T > 0, 'T must be > 0'
        assert self.ref.S0 >= 0, 'S must be >= 0'
        assert self.rf_r >= 0, 'r must be >= 0'

        S0 = self.ref.S0
        vol = self.ref.vol
        ttm = self.T
        K = self.K
        K2 = self.K2
        r = self.rf_r
        try: q = self.ref.q
        except: pass

        S_max   = S0*2                                # Maximum stock price
        S_min   = 0.0                                 # Minimum stock price
        d_t     = ttm/(time_steps-1)                  # Time step
        S_vec   = np.linspace(S_min,S_max,px_paths)   # Initialize the possible stock price vector
        t_vec   = np.linspace(0,ttm,time_steps)       # Initialize the time vector

        f_px    = np.zeros((px_paths,time_steps))     # Initialize the matrix. Hull's P482

        M = px_paths - 1
        N = time_steps-1

        # Set boundary conditions.
        f_px[:,-1]=S_vec

        if self.right=='call':
            # Payout at the maturity time
            init_cond = np.maximum((S_vec-K),0)*(S_vec>=K2)
            # Boundary condition
            upper_bound = 0
            # Calculate the current value
            lower_bound = np.maximum((S_vec[-1]-K),0)*(S_vec[-1]>=K2)*np.exp(-r*(ttm-t_vec))
        elif self.right=='put':
            # Payout at the maturity time
            init_cond = np.maximum((K-S_vec),0)*(S_vec<=K2)
            # Boundary condition
            upper_bound = np.maximum((K-S_vec[0]),0)*(S_vec[0]<=K2)*np.exp(-r*(ttm-t_vec))
            # Calculate the current value
            lower_bound = 0


        #Generate Matrix B in http://www.goddardconsulting.ca/option-pricing-finite-diff-implicit.html
        j_list = np.arange(0,M+1)
        a_list = 0.5*d_t*((r-q)*j_list-vol**2*j_list**2)
        b_list = 1+d_t*(vol**2*j_list**2 + r)
        c_list = 0.5*d_t*(-(r-q)*j_list-vol**2*j_list**2)

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

