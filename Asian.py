import matplotlib.pyplot as plt
import numpy as np

try: from qfrm.European import *  # production:  if qfrm package is installed
except:   from European import *  # development: if not installed and running from source


class Asian(European):
    """ Asian option class.

    Inherits all methods and properties of OptionValuation class.
    Asian options pay by the averaged historical value up to maturity of an underlying as the strike,
    against the maturity value of the underlying, or they compare the averaged value of the underlying
    up to maturity against a fixed strike.
    """

    def calc_px(self, sub_method='Arithmetic', strike='K', **kwargs):
        """ Wrapper function that calls appropriate valuation method.

        Parameters
        ----------
        sub_method : {'Arithmetic', 'Geometric'}
            Required. Calculation of price using 'Geometric' or 'Arithmetic' averages.
            Case-insensitive and may use partial string w/first letter.
        strike : {'K', 'S'}
            Required.
            If `'K'`, then the average asset price is compared against a fixed strike variable K to determine payoff.
            If `'S'`, then the asset price at maturity is compared against the average asset price
            over [0,T], i.e. the average underlying becomes the strike and what is assigned to variable ``K`` in
            ``OptionValuation`` is ignored.
        kwargs : dict
            Keyword arguments (``method``, ``nsteps``, ``npaths``, ``keep_hist``, ``rng_seed``, ...)
            are passed to the parent. See ``European.calc_px()`` for details.

        Returns
        -------
        self : Asian
            Returned object contains specifications and calculated price in  ``px_spec``
            variable (``PriceSpec`` object).


        Notes
        -----
        Use resources below to examine formulas and verify calculations.

        *References:*

        - Calculation of Moments for Valuing Asian Options, `Technical Note #27, J.C.Hull <http://www-2.rotman.utoronto.ca/~hull/technicalnotes/TechnicalNote27.pdf>`_
        - The referenced MC example is found in Lars Nielsen's 2001 paper, in section 4.3.
          `Pricing Asian Options, Lars B. Nielsen, 2001 <http://1drv.ms/1Ohf56n>`_.
          The results emulate the ``sigma`` = {0.05; 0.15; 0.45} elements of the third (C-hat) column of table
          4.1 on p.23. The computed MC prices for these simulations all fall within 2 standard errors of C-hat.
        - Asian Options - Tutorial and Excel Spreadsheet. `Excel tool. Samir Khan <http://investexcel.net/asian-options-excel>`_
        - Asian option with continuous arithmetical mean. `Online option pricer <http://www.infres.enst.fr/~decreuse/pricer/en/index.php?page=asiat_trapeze.html>`_
        - Arithmetic Average Options and Asian Options (Ch.10, FCFE Course, NTU) `Jr-Yan Wang, 2015 <http://1drv.ms/1SUmTyR>`_
        - An efficient convergent lattice algorithm for European Asian options (NTU.edu.tw), `Tian-Shyr Dai, et al., 2004  <http://www.csie.ntu.edu.tw/~lyuu/works/asian.pdf>`_
        - Simple, fast and flexible pricing of Asian options (Columbia.edu), `Timothy R. Klassen, <http://phys.columbia.edu/~klassen/asian.pdf>`_


        Examples
        --------

        SEE NOTES to verify first two examples

        >>> s = Stock(S0=30, vol=.3, q = .02)
        >>> o = Asian(ref=s, right='call', K=29, T=1., rf_r=.08, desc='See investexcel.net - GEO Call')
        >>> o.calc_px(method='BS').px_spec     # doctest: +ELLIPSIS
        PriceSpec...px: 2.777361113...

        >>> s = Stock(S0=30, vol=.3, q = .02)
        >>> o = Asian(ref=s, right='put', K=29, T=1., rf_r=.08, desc='See investexcel.net - GEO Put')
        >>> o.calc_px(method='BS').px_spec     # doctest: +ELLIPSIS
        PriceSpec...px: 1.224078447...

        >>> s = Stock(S0=30, vol=.3, q = .02)
        >>> o = Asian(ref=s, right='put', K=30., T=1., rf_r=.08)
        >>> o.calc_px(method='BS').px_spec     # doctest: +ELLIPSIS
        PriceSpec...px: 1.634104799...

        >>> s = Stock(S0=20, vol=.3, q = .00)
        >>> o = Asian(ref=s, right='put', K=21., T=1., rf_r=.08)
        >>> o.calc_px(method='BS').px_spec     # doctest: +ELLIPSIS
        PriceSpec...px: 1.489497403...

        >>> s = Stock(S0=20, vol=.3, q = .00)
        >>> o = Asian(ref=s, right='put', K=21., T=2., rf_r=.08)
        >>> o.calc_px(method='BS').px_spec     # doctest: +ELLIPSIS
        PriceSpec...px: 1.616211808...

        >>> s = Stock(S0=20, vol=.3, q = .00)
        >>> o = Asian(ref=s, right='put', K=21., T=2., rf_r=.08)
        >>> from pandas import Series;  exps = range(1,10)
        >>> O = Series([o.update(T=t).calc_px(method='BS').px_spec.px for t in exps], exps)
        >>> O.plot(grid=1, title='Price vs Time to Expiry') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> # import matplotlib.pyplot as plt
        >>> # plt.show() # run last two lines to show plot

        **MC**

        Examples below show option price sensitivity to volatility of the underlying stock.
        ``vol`` = 5%

        >>> s = Stock(S0=100, vol=.05, q = 0.0)
        >>> o = Asian(ref=s, right='call', K=100, T=1., rf_r=.05, desc='Nielsen, Lars. 2001. Pricing Asian Options.')
        >>> o.pxMC(nsteps=12, npaths=50000, rng_seed=1, sub_method='A', strike='K')
        ... # doctest: +ELLIPSIS
        2.94037098...

        ``vol`` = 15%

        >>> s = Stock(S0=100, vol=.15, q = 0.0)
        >>> o = Asian(ref=s, right='call', K=100, T=1., rf_r=.05, desc='Nielsen, Lars. 2001. Pricing Asian Options.')
        >>> o.pxMC(nsteps=12, npaths=50000, rng_seed=1, sub_method='A', strike='K')
        ... # doctest: +ELLIPSIS
        5.04298420...

        ``vol`` = 45%

        >>> s = Stock(S0=100, vol=.45, q = 0.0)
        >>> o = Asian(ref=s, right='call', K=100, T=1., rf_r=.05, desc='Nielsen, Lars. 2001. Pricing Asian Options.')
        >>> o.pxMC(nsteps=12, npaths=50000, rng_seed=1, sub_method='A', strike='K')
        ... # doctest: +ELLIPSIS
        12.0311406...

        In the following example the previous test will be run with only 100 trials on a different seed.

        >>> s = Stock(S0=100, vol=.05, q = 0.0)
        >>> o = Asian(ref=s, right='call', K=100, T=1., rf_r=.05, desc='Nielsen, Lars. 2001. Pricing Asian Options.')
        >>> o.pxMC(nsteps=12, npaths=100, rng_seed=1, sub_method='A', strike='K')
        ... # doctest: +ELLIPSIS
        3.15932733...

        In the following example, a average strike geometric put with the Hull example inputs is priced.

        >>> s = Stock(S0=50, vol=.4, q = 0.0)
        >>> o = Asian(ref=s, right='put', K=50, T=1., rf_r=.1, desc='Hull p. 610 Example 26.3')
        >>> o.pxMC(nsteps=12, npaths=50000, rng_seed=12, sub_method='G', strike='S')
        ... # doctest: +ELLIPSIS
        0.217125523...

        In the following example, a vector of fixed strikes generates a vector of Asian prices and is plotted.

        >>> import matplotlib.pyplot as plt
        >>> from numpy import linspace
        >>> Karr = linspace(30,70,101)
        >>> px = tuple(map(lambda i:  Asian(ref=Stock(50, vol=.6), right='call', K=Karr[i], T=1, rf_r=0.1).
        ... pxMC(nsteps=12, npaths=10000, rng_seed=i, sub_method='G', strike='K'),
        ... range(Karr.shape[0])))
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> ax.plot(Karr,px,label='AGK-call') # doctest: +ELLIPSIS
        [<...>]
        >>> ax.set_title('Price of AGK-call vs K') # doctest: +ELLIPSIS
        <...>
        >>> ax.set_ylabel('Px') # doctest: +ELLIPSIS
        <...>
        >>> ax.set_xlabel('K') # doctest: +ELLIPSIS
        <...>
        >>> ax.grid()
        >>> ax.legend() # doctest: +ELLIPSIS
        <...>
        >>> plt.show()


        **FD**

        >>> s = Stock(S0=0.5, vol=.01, q=.0)
        >>> o = Asian(ref=s, right='call', K=0.45, T=0.5, rf_r=.001)

        >>> o.calc_px(method='FD',nsteps=10,npaths=10).px_spec # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px:  0.025869147...

        >>> s = Stock(S0=1.5, vol=.01, q=.0)
        >>> o = Asian(ref=s, right='put', K=2, T=0.5, rf_r=.01)

        >>> o.calc_px(method='FD',npaths=10,nsteps=10).px_spec.px
        0.5770208101062311

        >>> s = Stock(S0=1.5, vol=.01)
        >>> o = Asian(ref=s, right='put', K=3.5, T=1.5, rf_r=.01)

        >>> o.calc_px(method='FD',npaths=10,nsteps=10).px_spec.px
        2.062508335359562

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([o.update(T=t).calc_px(method='FD',npaths=2,nsteps=3).px_spec.px for t in expiries], expiries)
        >>> O.plot(grid=1, title='Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()

        :Authors:
            Scott Morgan,
            Andrew Weatherly,
            Andy Liao <Andy.Liao@rice.edu>,
            Hanting Li <hl45@rice.edu>

        """

        self.save2px_spec(sub_method=sub_method, strike=strike, **kwargs)
        return getattr(self, '_calc_' + self.px_spec.method.upper())()

    def _calc_BS(self):
        """ Internal function for option valuation.   See ``calc_px()`` for complete documentation.

        :Authors:
            Scott Morgan
        """

        # Verify input
        try:
            right   =   self.right.lower()
            S       =   float(self.ref.S0)
            K       =   float(self.K)
            T       =   float(self.T)
            vol     =   float(self.ref.vol)
            r       =   float(self.rf_r)
            q       =   float(self.ref.q)


        except:
            print('right must be String. S, K, T, vol, r, q must be floats or be able to be coerced to float')
            return False

        assert right in ['call', 'put'], 'right must be "call" or "put" '
        assert vol > 0, 'vol must be >=0'
        assert K > 0, 'K must be > 0'
        assert T > 0, 'T must be > 0'
        assert S >= 0, 'S must be >= 0'
        assert r >= 0, 'r must be >= 0'
        assert q >= 0, 'q must be >= 0'


        # Parameters for Value Calculation (see link in docstring)
        a = .5 * (r - q - (vol ** 2) / 6.)
        vola = vol / np.sqrt(3.)
        d1 = (np.log(S * np.exp(a * T) / K) + (vola ** 2) * .5 * T) / (vola * np.sqrt(T))
        d2 = d1 - vola * np.sqrt(T)
        # sp = self._BS_specs();         d1, d2 = sp['d1'], sp['d2']

        # Calculate the value of the option using the BS Equation
        N = Util.norm_cdf
        if right == 'call':
            px = S * np.exp((a - r) * T) * N(d1) - K * np.exp(-r * T) * N(d2)
            self.px_spec.add(px=float(px), method='BSM', sub_method='Geometric')

        else:
            px = K * np.exp(-r * T) * N(-d2) - S * np.exp((a - r) * T) * N(-d1)
            self.px_spec.add(px=float(px), method='BSM', sub_method='Geometric')
        return self

    def _calc_LT(self):
        """ Internal function for option valuation.      See ``calc_px()`` for complete documentation.

        :Authors:
            Andrew Weatherly
        """
        #Imports
        import numpy as np
        from math import exp, sqrt, log, floor
        from scipy.interpolate import interp1d

        #helper function
        def interpolate(xArr, yArr, x):
            if xArr[0] == x:
                return yArr[0]
            for i in range(0, len(xArr) - 1):
                if xArr[i] >= x:
                    #print(yArr[i - 1], xArr[i - 1], x)
                    return float(yArr[i - 1] + (x - xArr[i - 1]) / (xArr[i] - xArr[i - 1]) * (yArr[i] - yArr[i - 1]))


        #Parameters for Lattice Tree
        nsteps = self.px_spec.nsteps
        dt = self.T / nsteps
        u = exp(self.ref.vol * sqrt(dt))
        d = exp(-self.ref.vol * sqrt(dt))
        growth_factor = exp((self.rf_r - self.ref.q) * dt)
        pu = (growth_factor - d) / (u - d)
        pd = 1 - pu
        df_T = exp(-self.rf_r * self.T)
        df_dt = exp(-(self.rf_r - self.ref.q) * dt)
        h = .1
        par = {'dt': dt,           # time interval between consecutive two time steps
               'u': u,             # stock price up move factor
               'd': d,             # stock price down move factor
               'a': growth_factor, # growth factor, p.452
               'pu': pu,             # probability of up move over one time interval dt
               'pd': pd,
               'df_T': df_T,       # discount factor over full time interval dt, i.e. per life of an option
               'df_dt': df_dt}     # discount factor over one time interval dt, i.e. per step

        S = np.zeros((nsteps + 1, nsteps + 1)) # Stock price paths
        Val = np.zeros((nsteps + 1, nsteps + 1)) # Inner value matrix
        S[0, 0] = self.ref.S0
        for i in range(1, nsteps + 1):
            for j in range(0, i + 1):
                if j <= i:
                    S[i, j] = self.ref.S0 * (par['u'] ** j) * (par['d'] ** (i - j))
                    if i == nsteps:
                        Val[i, j] = np.maximum((-self.K + S[i, j]) * self.signCP, 0)
        Fvec = self.ref.S0
        FTree = np.zeros((nsteps * 2, nsteps * 2))
        FTree[0] = Fvec
        for c in range(1, nsteps + 1):
            StockPriceVec = S[c]
            Smax = np.max(StockPriceVec[np.nonzero(StockPriceVec)])
            Smin = np.min(StockPriceVec[np.nonzero(StockPriceVec)])
            PrevAverageVec = FTree[c - 1]
            PrevMaxAverage = np.max(PrevAverageVec[np.nonzero(PrevAverageVec)])
            PrevMinAverage = np.min(PrevAverageVec[np.nonzero(PrevAverageVec)])
            MaxAverage = (PrevMaxAverage * i + Smax) / (i + 1)
            MinAverage = (PrevMinAverage * i + Smin) / (i + 1)
            #now find integer values of m which cover min and max average values
            tmpdbl = log(MaxAverage / self.ref.S0) / h
            MaxM = floor(tmpdbl) + 1
            tmpdbl = log(MinAverage / self.ref.S0) / h
            MinM = floor(abs(tmpdbl)) + 1
            tmpN = MaxM + MinM + 1
            Fvec = np.zeros((tmpN, 1))
            counter = -MinM
            for j in range(0, tmpN):
                Fvec[j] = self.ref.S0 * exp(counter * h)
                counter += 1
            for j, t in enumerate(Fvec):
                FTree[c, j] = t
        """
        for i in range(0, len(FTree)):
            for j in range(0, len(FTree)):
                print(FTree[i, j], i, j)
        """
        #Step 3 : Do backward recursion of the tree
        #initialize option values at maturity
        #print(Fvec)
        #running average values to consider for calculating set of option
        Fvec = FTree[nsteps, np.nonzero(FTree[nsteps])]
        # prices
        Fvec = [Fvec[0][i] for i in range(0, len(Fvec[0]))]
        VTree = np.zeros((nsteps * 2, 2 * nsteps, len(Fvec)))#stores option values at all nodes of the tree
        VTimevec = np.zeros((nsteps, len(Fvec)))  #stores list of vectors of option prices at a given time
        VNodeVec = np.zeros((len(Fvec), 1))
        for j in range(0, len(Fvec)): #loop over average values
            VNodeVec[j] = np.maximum(Fvec[j] - self.K, 0)
        for i in range(0, nsteps):  #loop over nodes at a given time
            for col, t in enumerate(VNodeVec):
                VTimevec[i, col] = t
        for g in range(0, len(VTimevec)):
            VTree[nsteps - 1][g] = VTimevec[g]

        for i in range(nsteps - 2, -1, -1):
            Svecnext = S[i + 1]
            VTimevec = np.zeros((len(FTree[i]), len(FTree[i])))
            VTimeVecNext = VTree[i + 1]
            Fvec = FTree[i, np.nonzero(FTree[i])] #running average values to consider for calculating set of option
            # prices
            #running average values to consider for calculating set of
            Fvec = [Fvec[0][z] for z in range(0, len(Fvec[0]))]
            #running average values to consider for calculating set of
            FVecNext = FTree[i + 1, np.nonzero(FTree[i + 1])]
            FVecNext = [FVecNext[0][z] for z in range(0, len(FVecNext[0]))]
            # option prices
            for j in range(0, i + 1):
                VNodeVec = np.zeros((len(Fvec)))
                for k in range(0, len(Fvec)):
                    #calculate option price using F at current node and Su
                    F = Fvec[k] #running average
                    #find running average at next time node of up-jump
                    Su = Svecnext[j + 1]
                    Fu = (F * (i + 1) + Su) / (i + 2)
                    VNodeVecNext = VTimeVecNext[j + 1] #vector of option prices
                    #get option value to the next timestep]
                    #print(FVecNext)
                    #print(VNodeVecNext)
                    #print(Fu)
                    Vu = interpolate(FVecNext, VNodeVecNext, Fu)
                    if Vu is None:
                        Vu = 0
                    #find running average at next time node of down-jump
                    Sd = Svecnext[j]
                    Fd = (F * (i + 1) + Sd) / (i + 2)
                    VNodeVecNext = VTimeVecNext[j]  #vector of option prices
                    #get option value to the next timestep
                    Vd = interpolate(FVecNext, VNodeVecNext, Fd)
                    if Vd is None:
                        Vd = 0
                    #Vnew = Exp(-r * dt) * (Vu * pu + Vd * pd)
                    present_value = par['df_dt'] * (Vu * pu + Vd * pd)
                    #immediate_val = self.signCP * (F - self.K)
                    #if self.right == "American":
                    #    Vnew = max(present_value, immediate_val)
                    #else:
                    Vnew = max(present_value, 0)
                    VNodeVec[k] = Vnew
                for t in range(0, len(VNodeVec)):
                    VTimevec[j][t] = VNodeVec[t]
            for z in range(0, VTimevec.shape[0]):
                for r in range(0, VTimevec.shape[1]):
                    VTree[z][r][i] = VTimevec[z][r]
        CRR_Price = VTree[0][0][0]
        self.px_spec.add(px=float(CRR_Price), method='LT', sub_method='Hull and White Interpolation')
        return self

    def _calc_MC(self):
        """ Internal function for option valuation.        See ``calc_px()`` for complete documentation.

        :Authors:
            Andy Liao <Andy.Liao@rice.edu>
        """

        #Throw exception if inputs are not the right type.
        try:
            right   =   self.right
            S0       =   float(self.ref.S0)
            K       =   float(self.K)
            T       =   float(self.T)
            vol     =   float(self.ref.vol)
            r       =   float(self.rf_r)
            q       =   float(self.ref.q)
            n_steps =   int(self.px_spec.nsteps)
            n_paths =   int(self.px_spec.npaths)
            rng_seed =  int(self.px_spec.rng_seed)
            sub_method =    self.px_spec.sub_method.upper()[0]  # 'A' for 'G'
            strike =    self.px_spec.strike

        except:
            print('right must be String. ')
            print('S, K, T, vol, r, q must be floats or be able to be coerced to float; nsteps, npaths, rng_seed to \
            ints')
            print('sub_method must be String')
            return False

        #Generate evenly spaced observations of stock price with the Geometric Brownian Motion process.
        dt = T / n_steps

        np.random.seed(seed=rng_seed)
        S = S0*np.cumprod(np.exp((r-0.5*vol**2)*dt+vol*np.sqrt(dt)*np.random.randn(n_paths,n_steps)),1)

        #Calculate an average stock price over (0,T] for each path by the selected sub-method.
        S_avg = np.zeros(S.shape[1])
        if sub_method == 'G':
            # self.px_spec.add(sub_method='Geometric')
            S_avg = np.exp(np.sum(np.log(S),axis=1)/(n_steps+1))
        if sub_method == 'A':
            # self.px_spec.add(sub_method='Arithmethic')
            S_avg = np.mean(S,axis=1)

        #The price at maturity is needed if the user wants a Average-Strike Asian option.
        S_T = S.transpose()[n_steps-1]

        df = np.exp(-r*T)

        #Payoffs calculated: 2 rights x 2 variants = 4 outcomes
        #Payoffs calculated: 4 outcomes x 2 definitions of "mean" = 8 different prices
        pay = np.zeros(S_avg.shape)
        if strike == 'K':
            if right == 'call':
                pay = np.maximum(0,S_avg-K) #fixed strike call
            if right == 'put':
                pay = np.maximum(0,K-S_avg) #fixed strike put
        if strike == 'S':
            if right == 'call':
                pay = np.maximum(0,S_T-S_avg) #average strike call
            if right == 'put':
                pay = np.maximum(0,S_avg-S_T) #average strike put

        #compute the average of the distribution as the price of the option.
        v0 = sum(pay)/n_paths*df
        self.px_spec.add(px=float(v0))
        return self

    def _calc_FD(self):
        """ Internal function for option valuation. See ``calc_px()`` for complete documentation.

        :Authors:
            Hanting Li <hl45@rice.edu>

        """

        # Get Parameters
        M = getattr(self.px_spec, 'npaths', 3)
        N = getattr(self.px_spec, 'nsteps', 3)
        signCP = 1 if self.right == 'call' else -1
        T = self.T
        vol = self.ref.vol
        S0 = self.ref.S0
        Smax = 2*S0
        Smin = S0/2
        r = self.rf_r
        q = self.ref.q
        dt = T/(N)
        dS = (Smax-Smin)/M
        K = self.K
        df = np.exp(-r*dt)

        # Define the grid
        PriceM = np.matrix(np.zeros(shape=(N+1,M+1)))

        # Simulate a price path, to get the average price
        # Compute the stock price at t
        def calS(St,mu,sigma,param):
            deltaS = mu*St*dt + sigma*St*param*np.sqrt(dt)
            S_update = St+deltaS
            return(S_update.item())

        Nsimu = 50

        np.random.seed(0)
        param = np.random.normal(0,0.1,Nsimu)
        # Generate one path
        def one_path(S0,mu,vol,param):
            S0 = (S0,)
            for i in range(Nsimu):
                parami = param[i]
                S0 = S0 + (calS(S0[len(S0)-1],mu,vol,parami),)
            return(S0)

        # Set Boundary Condition
        for i in range(0,M+1):
            Sinput = Smin + i*dS
            PriceM[N,i] = np.maximum(signCP*(np.mean(one_path(Sinput,0,vol,param))-K),0)
        PriceM[:,0] = np.matrix(list(map(lambda i: (PriceM[N,0]*np.exp(-r*(N-i)*dt)), \
                                range(0,N+1)))).transpose()
        PriceM[:,M] = np.matrix(list(map(lambda i: (PriceM[N,M]*np.exp(-r*(N-i)*dt)), \
                                range(0,N+1)))).transpose()

        # Compute the grid
        for i in np.arange(N-1,-1,-1):
            for k in range(1,M):
                j = M-k
                PriceM[i,k] = df*(.5*dt*((vol**2)*(j**2)-(r-q)*j))*PriceM[i+1,k+1] + \
                         df*(1 - dt*((vol**2)*(j**2)))*PriceM[i+1,k] + \
                         df*(.5*dt*((vol**2)*(j**2)+(r-q)*j))*PriceM[i+1,k-1]

        # Return
        self.px_spec.add(px=float(np.maximum(PriceM[0,(S0-Smin)/dS],0)), method='FD')

        return self


# s = Stock(S0=50, vol=.4, q = 0.0)
# o = Asian(ref=s, right='put', K=50, T=1., rf_r=.1, desc='Hull p. 610 Example 26.3')
# o.pxMC(nsteps=12, npaths=50000, rng_seed=12, sub_method='Arithmetic', strike='S')