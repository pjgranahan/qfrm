from OptionValuation import *


class Asian(OptionValuation):
    """ Asian option class.

    Inherits all methods and properties of OptionValuation class.
    """


    def calc_px(self, method='BS', nsteps=None, npaths=None, keep_hist=False):
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

        Returns
        -------
        self : Asian

        .. sectionauthor:: Scott Morgan & Andrew Weatherly

        Notes
        -----

        Verification of First and Second Examples: http://investexcel.net/asian-options-excel/

        Examples
        -------

        >>> # SEE NOTES to verify first two examples
        >>> s = Stock(S0=30, vol=.3, q = .02)
        >>> o = Asian(ref=s, right='call', K=29, T=1., rf_r=.08, desc='Example from Internet - Call')
        >>> o.calc_px()
        >>> print(o.px_spec)

            qfrm.PriceSpec
            keep_hist: false
            method: BSM
            px: 2.777361112923389
            sub_method: Geometric

        >>> s = Stock(S0=30, vol=.3, q = .02)
        >>> o = Asian(ref=s, right='put', K=29, T=1., rf_r=.08, desc='Example from Internet - Put')
        >>> o.calc_px()
        >>> print(o.px_spec)

            qfrm.PriceSpec
            keep_hist: false
            method: BSM
            px: 1.2240784465431602
            sub_method: Geometric

        >>> s = Stock(S0=30, vol=.3, q = .02)
        >>> o = Asian(ref=s, right='put', K=30., T=1., rf_r=.08)
        >>> o.calc_px()
        >>> print(o.px_spec)

            qfrm.PriceSpec
            keep_hist: false
            method: BSM
            px: 1.6341047993229445
            sub_method: Geometric

        >>> s = Stock(S0=20, vol=.3, q = .00)
        >>> o = Asian(ref=s, right='put', K=21., T=1., rf_r=.08)
        >>> o.calc_px()
        >>> print(o.px_spec)

            qfrm.PriceSpec
            keep_hist: false
            method: BSM
            px: 1.489497403315955
            sub_method: Geometric

        >>> s = Stock(S0=20, vol=.3, q = .00)
        >>> o = Asian(ref=s, right='put', K=21., T=2., rf_r=.08)
        >>> o.calc_px()
        >>> print(o.px_spec)

            qfrm.PriceSpec
            keep_hist: false
            method: BSM
            px: 1.6162118076748948
            sub_method: Geometric

        >>> s = Stock(S0=20, vol=.3, q = .00)
        >>> o = Asian(ref=s, right='put', K=21., T=2., rf_r=.08)
        >>> from pandas import Series;  exps = range(1,10)
        >>> O = Series([o.update(T=t).calc_px(method='BS').px_spec.px for t in exps], exps)
        >>> O.plot(grid=1, title='Price vs Time to Expiry')
        >>> # import matplotlib.pyplot as plt
        >>> # plt.show() # run last two lines to show plot


       """

        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
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

        http://homepage.ntu.edu.tw/~jryanwang/course/Financial%20Computation%20or%20Financial%20Engineering%20(graduate%20level)/FE_Ch10%20Asian%20Options.pdf
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
        from math import exp
        from math import log
        from math import sqrt
        from scipy.stats import norm

        # Parameters for Value Calculation (see link in docstring)
        a = .5 * (r - q - (vol ** 2) / 6.)
        vola = vol / sqrt(3.)
        d1 = (log(S * exp(a * T) / K) + (vola ** 2) * .5 * T) / (vola * sqrt(T))
        d2 = d1 - vola * sqrt(T)

        # Calculate the value of the option using the BS Equation
        if right == 'call':
            px = S * exp((a - r) * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
            self.px_spec.add(px=float(px), method='BSM', sub_method='Geometric')

        else:
            px = K * exp(-r * T) * norm.cdf(-d2) - S * exp((a - r) * T) * norm.cdf(-d1)
            self.px_spec.add(px=float(px), method='BSM', sub_method='Geometric')
        return self

    def _calc_MC(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: American

        .. sectionauthor:: Oleg Melnikov

        Note
        ----

        """
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

