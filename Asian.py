from OptionValuation import *
from math import sqrt
from numpy import cumsum, maximum, sum,  exp, mean, zeros, log
from numpy.random import normal, seed
from math import exp as mexp
from math import log as mlog
from scipy.stats import norm

class Asian(OptionValuation):
    """ SHORT DESCRIPTION: Asian option class.

    LONG DESCRIPTION:
    Inherits all methods and properties of OptionValuation class.
    Asian options pay by the averaged historical value up to maturity of an underlying as the strike, against the maturity 
    value of the underlying, or they compare the averaged value of the underlying up to maturity against a fixed strike.
    """


    def calc_px(self, method='MC', nsteps=3, npaths=10000, keep_hist=False, rng_seed=1, sub_method='Arithmetic', strike='K'):
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
        rng_seed : int
                MC method requires the seed for RNG to generate historical prices in (0,T). 
        sub_method : str
                Required. Calculation of price using 'Geometric' or 'Arithmetic' averages. 
                Case-insensitive and may use partial string w/first letter. 
        strike : str
                Required. If 'K', then the average asset price is compared against a fixed strike variable K to determine payoff.
                If 'S', then the asset price at maturity is compared against the average asset price over [0,T], i.e. the average underying
                becomes the strike and what is assigned to variable K in OptionValuation is ignored.


        Returns
        -------
        self : Asian

        .. sectionauthor:: Scott Morgan & Andrew Weatherly & Andy Liao

        Notes
        -----
        BS; LT Methods,
        Verification of First and Second Examples: http://investexcel.net/asian-options-excel/

        ================================
        README ABOUT MONTE CARLO METHODS
        ================================
        When you use the Monte Carlo method to price the Asian option, you will find that the result is an
        approximation of the BS result described in the Hull 9e textbook. By going through the examples 
        below, you will also see the dependency of the goodness of the MC method, as approximation, 
        against the "analytic" results, against the number of observations and paths for the discrete scenario.
        The MC generated distribution does not necessarily generate a lognormal distribution around the BS result,
        they converge to an overestimate by up to 3%. What you will see however, is that the relation between 
        the step number for the discrete scenario and the price follows what is predicted analytically, and they 
        converge to the continuous case.
        
        To improve the speed of convergence, an antithetic control variate approach is recommended.
        This method is not implemented here.
        
        The continuous case is not implemented here. It makes little sense to do so, since the prices generated
        by the GBM process are discrete.
        
        Finally the referenced examples are only for the Geometric, Fixed-Strike variant of the Asian option.
        To obtain pricing for other variants of Asian options, I suggest that you pick up a copy of
        DerivaGem software.
        ================================       

        Examples
        -------

        >>> # SEE NOTES to verify first two examples
        >>> s = Stock(S0=30, vol=.3, q = .02)
        >>> o = Asian(ref=s, right='call', K=29, T=1., rf_r=.08, desc='http://investexcel.net/asian-options-excel/ - GEO Call')
        >>> o.calc_px(method='BS').px_spec
        PriceSpec
        keep_hist: false
        method: BSM
        npaths: 10000
        nsteps: 3
        px: 2.777361113
        rng_seed: 1
        strike: K
        sub_method: Geometric
        <BLANKLINE>

        >>> s = Stock(S0=30, vol=.3, q = .02)
        >>> o = Asian(ref=s, right='put', K=29, T=1., rf_r=.08, desc='http://investexcel.net/asian-options-excel/ - GEO Put')
        >>> o.calc_px(method='BS').px_spec
        PriceSpec
        keep_hist: false
        method: BSM
        npaths: 10000
        nsteps: 3
        px: 1.224078447
        rng_seed: 1
        strike: K
        sub_method: Geometric
        <BLANKLINE>

        >>> s = Stock(S0=30, vol=.3, q = .02)
        >>> o = Asian(ref=s, right='put', K=30., T=1., rf_r=.08)
        >>> o.calc_px(method='BS').px_spec
        PriceSpec
        keep_hist: false
        method: BSM
        npaths: 10000
        nsteps: 3
        px: 1.634104799
        rng_seed: 1
        strike: K
        sub_method: Geometric
        <BLANKLINE>

        >>> s = Stock(S0=20, vol=.3, q = .00)
        >>> o = Asian(ref=s, right='put', K=21., T=1., rf_r=.08)
        >>> o.calc_px(method='BS').px_spec
        PriceSpec
        keep_hist: false
        method: BSM
        npaths: 10000
        nsteps: 3
        px: 1.489497403
        rng_seed: 1
        strike: K
        sub_method: Geometric
        <BLANKLINE>

        >>> s = Stock(S0=20, vol=.3, q = .00)
        >>> o = Asian(ref=s, right='put', K=21., T=2., rf_r=.08)
        >>> o.calc_px(method='BS').px_spec
        PriceSpec
        keep_hist: false
        method: BSM
        npaths: 10000
        nsteps: 3
        px: 1.616211808
        rng_seed: 1
        strike: K
        sub_method: Geometric
        <BLANKLINE>

        >>> s = Stock(S0=20, vol=.3, q = .00)
        >>> o = Asian(ref=s, right='put', K=21., T=2., rf_r=.08)
        >>> from pandas import Series;  exps = range(1,10)
        >>> O = Series([o.update(T=t).calc_px(method='BS').px_spec.px for t in exps], exps)
        >>> O.plot(grid=1, title='Price vs Time to Expiry') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> # import matplotlib.pyplot as plt
        >>> # plt.show() # run last two lines to show plot
        
        >>> #BEGIN MONTE CARLO EXAMPLES-----
        
        >>> #In the following 3 examples, show the effect of changing observation number on price. 
        >>> #Compare with the analytic result on Hull p 610, example 26.3:
        >>> #Given the input parameters quoted in the examples following, 
        >>> #for ((nsteps,px)) -> ((12,6.00),(52,5.70),(250,5.63))
        
        >>> #12 steps
        >>> s = Stock(S0=50, vol=.4, q = 0.0)
        >>> o = Asian(ref=s, right='call', K=50, T=1., rf_r=.1, desc='Hull p. 610 Example 26.3')
        >>> o.calc_px(method='MC', nsteps=12, npaths=10000, rng_seed=38, sub_method='G', strike='K').px_spec.px 
        ... # doctest: +ELLIPSIS
        6.32551519...
        
        >>> #52 steps
        >>> s = Stock(S0=50, vol=.4, q = 0.0)
        >>> o = Asian(ref=s, right='call', K=50, T=1., rf_r=.1, desc='Hull p. 610 Example 26.3')
        >>> o.calc_px(method='MC', nsteps=52, npaths=10000, rng_seed=38, sub_method='G', strike='K').px_spec.px
        ... # doctest: +ELLIPSIS
        5.75569993...
        
        >>> #250 steps
        >>> s = Stock(S0=50, vol=.4, q = 0.0)
        >>> o = Asian(ref=s, right='call', K=50, T=1., rf_r=.1, desc='Hull p. 610 Example 26.3')
        >>> o.calc_px(method='MC', nsteps=250, npaths=10000, rng_seed=38, sub_method='G', strike='K').px_spec.px
        ... # doctest: +ELLIPSIS
        5.68771397...
        
        >>> #In the following example the previous test will be run with only 100 trials on a different seed.
        >>> #This will demonstrate that having too few trials give unreliable estimates and 
        >>> #The Asian price converges slowly without control-variate techniques.
        >>> #250 steps
        >>> s = Stock(S0=50, vol=.4, q = 0.0)
        >>> o = Asian(ref=s, right='call', K=50, T=1., rf_r=.1, desc='Hull p. 610 Example 26.3')
        >>> o.calc_px(method='MC', nsteps=250, npaths=100, rng_seed=0, sub_method='G', strike='K').px_spec.px
        ... # doctest: +ELLIPSIS
        5.28352690...
                
        >>> #In the following example, a average strike arithmetic put with the Hull example inputs is priced.
        >>> s = Stock(S0=50, vol=.4, q = 0.0)
        >>> o = Asian(ref=s, right='put', K=50, T=1., rf_r=.1, desc='Hull p. 610 Example 26.3')
        >>> o.calc_px(method='MC', nsteps=250, npaths=10000, rng_seed=38, sub_method='A', strike='S').px_spec.px
        ... # doctest: +ELLIPSIS
        3.66169948...
                
        >>> #In the following example, a vector of fixed strikes generates a vector of Asian prices and is plotted.
        >>> import matplotlib.pyplot as plt
        >>> from numpy import linspace
        >>> Karr = linspace(30,70,101)
        >>> px = tuple(map(lambda i:  Asian(ref=Stock(50, vol=.6), right='call', K=Karr[i], T=1, rf_r=0.1).
        ... calc_px(method='MC', nsteps=12, npaths=10000, rng_seed=i, sub_method='G', strike='K').px_spec.px, 
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

       """

        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist, \
            rng_seed=rng_seed, sub_method=sub_method, strike=strike)        
        return getattr(self, '_calc_' + method.upper())()

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Asian

        .. sectionauthor:: Andrew Weatherly

        Note
        ----

        Formulae:

        http://homepage.ntu.edu.tw/~jryanwang/course/Financial%20Computation%20or%20Financial%20Engineering%20(graduate
            %20level)/FE_Ch10%20Asian%20Options.pdf
        http://www.csie.ntu.edu.tw/~lyuu/works/asian.pdf
        http://phys.columbia.edu/~klassen/asian.pdf

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
        Fvec = FTree[nsteps, np.nonzero(FTree[nsteps])] #running average values to consider for calculating set of option
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
            Fvec = [Fvec[0][z] for z in range(0, len(Fvec[0]))] #running average values to consider for calculating set of

            FVecNext = FTree[i + 1, np.nonzero(FTree[i + 1])] #running average values to consider for calculating set of
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

    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Asian

        .. sectionauthor:: Scott Morgan

        Note
        ----

        Formulae: http://homepage.ntu.edu.tw/~jryanwang/course/Financial%20Computation%20or%20Financial%20Engineering%20(graduate%20level)/FE_Ch10%20Asian%20Options.pdf

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

        # Imports


        # Parameters for Value Calculation (see link in docstring)
        a = .5 * (r - q - (vol ** 2) / 6.)
        vola = vol / sqrt(3.)
        d1 = (mlog(S * mexp(a * T) / K) + (vola ** 2) * .5 * T) / (vola * sqrt(T))
        d2 = d1 - vola * sqrt(T)

        # Calculate the value of the option using the BS Equation
        if right == 'call':
            px = S * mexp((a - r) * T) * norm.cdf(d1) - K * mexp(-r * T) * norm.cdf(d2)
            self.px_spec.add(px=float(px), method='BSM', sub_method='Geometric')

        else:
            px = K * mexp(-r * T) * norm.cdf(-d2) - S * mexp((a - r) * T) * norm.cdf(-d1)
            self.px_spec.add(px=float(px), method='BSM', sub_method='Geometric')
        return self

    def _calc_MC(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Asian

        .. sectionauthor:: Andy Liao

        Note
        ----

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
            sub_method =    self.px_spec.sub_method
            strike =    self.px_spec.strike
            
        except:
            print('right must be String. ')
            print('S, K, T, vol, r, q must be floats or be able to be coerced to float; nsteps, npaths, rng_seed to \
            ints')
            print('sub_method must be String')
            return False

        #Generate evenly spaced observations of stock price with the Geometric Brownian Motion process.
        dt = T / n_steps
        
        seed(seed=rng_seed)
        S = S0 * exp(cumsum(normal((r - 0.5 * vol ** 2) * dt, vol * sqrt(dt), (n_steps + 1, n_paths)), axis=0)) 
        S[0] = S0
        S = S.transpose()[:][1:]

        #Calculate an average stock price over (0,T] for each path by the selected sub-method.
        S_avg = zeros(S.shape[1])
        if sub_method[0] == 'g' or sub_method[0] == 'G':
            self.px_spec.add(sub_method='Geometric')  
            S_avg = exp(sum(log(S),axis=1)/(n_steps+1))
        if sub_method[0] == 'A' or sub_method[0] == 'A':
            self.px_spec.add(sub_method='Arithmethic')  
            S_avg = mean(S,axis=1)    

        #The price at maturity is needed if the user wants a Average-Strike Asian option.
        S_T = S.transpose()[n_steps-1]
        
        #Payoffs calculated: 2 rights x 2 variants = 4 outcomes
        #Payoffs calculated: 4 outcomes x 2 definitions of "mean" = 8 different prices
        pay = zeros(S_avg.shape)
        if strike == 'K':
            if right == 'call':
                pay = maximum(0,S_avg-K) #fixed strike call
            if right == 'put': 
                pay = maximum(0,K-S_avg) #fixed strike put
        if strike == 'S':
            if right == 'call':
                pay = maximum(0,S_T-S_avg) #average strike call
            if right == 'put':
                pay = maximum(0,S_avg-S_T) #average strike put
                               
        #compute the average of the distribution as the price of the option.
        v0 = sum(pay)/n_paths
        
        self.px_spec.add(px=v0)
        return self

    def _calc_FD(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: American

        .. sectionauthor:: Oleg Melnikov

        Note
        ----

        """

        return self
