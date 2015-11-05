# -*- coding: utf-8 -*-

#Class YieldCurve from HWO8.Computes through interpolation the interest rates for given TTM.
#Deleted the output methods from the original in this version.
class YieldCurve:
    """ YieldCurve class downloads (from FRED) historical interest rates for 11 maturities.

    .. seealso::
        https://research.stlouisfed.org/pdl/804
        http://stackoverflow.com/questions/11766536/matplotlib-3d-surface-from-a-rectangular-array-of-heights
        http://docs.scipy.org/doc/numpy/reference/generated/numpy.interp.html

    :Examples:

        >>> yc = YieldCurve(); yc.yc; yc.recent_curve(); yc.interp((.1,.2,.3,.4,.5))
        >>> yc =YieldCurve();  yc.yc;  yc.plot3d()
    """
    def __init__(self, start='2008-01-01', end=None):
        """ Downloads historical interest rates for 11 maturities:
        'GS1M','GS3M','GS6M','GS1','GS2','GS3','GS5','GS7','GS10','GS20','GS30'

        :param start: starting historical date; see pandas.DataReader()
        :type start: str
        :param end: ending historical date; see pandas.DataReader()
        :type end: str
        :return: __init__ always returns self
        :rtype: YieldCurve

        .. seealso::
            https://research.stlouisfed.org/pdl/804
        """
        from pandas.io.data import DataReader

        yc = DataReader(('GS1M','GS3M','GS6M','GS1','GS2','GS3','GS5','GS7','GS10','GS20','GS30'), 'fred', start=start, end=None)
        # yc = DataReader(['GS1M','GS3M'], "fred", '2015-07-01')
        yc.columns = [1/12, 1/4, 1/2, 1, 2, 3, 5, 7, 10, 20, 30]
        self.yc = yc / 100

    def recent_curve(self):
        """ Extracts most recent yield curve.
        :return: most recent yield curve.
        :rtype: iterable object
        """
        return self.yc.tail(1).transpose()

    def interp(self, ttm=None):
        """ Linearly interpolate yields at specified maturities

        :param ttm: user provided maturities (in years)
        :type ttm: iterable of positive floats
        :return: interpolated yields
        :rtype: iterable object
        """
        from numpy import interp, array

        return interp(ttm, array(self.yc.tail(1).columns), self.yc.tail(1).values[0])
        #tuple(x for x in array(self.yc.tail(1))))
        # yc.yc.tail(1).transpose().interpolate()
        #[(x[1].to_datetime().date() - date.today()).days / 365 for x in O.index.tolist()]

#Class Option from HW08. Computes BSM prices and implied volatilities for a given equity.
#Deleted the screen output methods from the original in this version.
#Instead, output the calls and puts on the market for a given equity in separate csv files in the working directory.
#Important: python requires write permission to the working directory.
class Option:
    """

    :Examples:

        >>> o = Option(symbol='AAPL', right='call')
        >>> o.O
        >>> o.plot3d(z_name='BS-IV')
        >>> o.plot3d(z_name='Last')
        >>> o.plot3d(z_name='IV')
        >>> o.plot3d(z_name='Vol')
        >>> o.plot3d(z_name='BS-IV - IV')
    """
    def __init__(self, symbol='AAPL', right='call'):
        """ Downloads options data from Yahoo!Finance.

        Use pandas.Options object to load market data for specified symbol, right.
        Then add computed fields
            ttm: years to maturity (computed from object's MultiIndex)
            K/S: moneyness (see p.439, table 20.2)
            r:  IR matching expiry of an option series (interpolated using YieldCurve.interp() method
            BS-IV: Black-Scholes model implied volatility
            BS-IV - IV: difference between BS-IV and IV.
                IV is computed by Options() class, but needs conversion to float (as a rate, not %)

        After all fields are tuned and added, save this DataFrame as self.O for later access by other class methods.

        :param symbol: stock symbol
        :type symbol: str
        :param right: call or put
        :type right:  str
        :return: __init__ always returns self
        :rtype: Option

        .. seealso::
            http://pandas.pydata.org/pandas-docs/stable/basics.html
            http://stackoverflow.com/questions/22015363/how-to-get-the-index-value-in-pandas-multiindex-data-frame
            http://stackoverflow.com/questions/3743222/how-do-i-convert-datetime-to-date-in-python
            http://stackoverflow.com/questions/24495695/pandas-get-unique-multiindex-level-values-by-label
            http://pandas.pydata.org/pandas-docs/stable/remote_data.html#yahoo-finance-options
            http://stackoverflow.com/questions/12432663/what-is-a-clean-way-to-convert-a-string-percent-to-a-float

        """
        from pandas.io.data import Options
        from datetime import date

        O = Options(symbol, 'yahoo').get_all_data()
        # data = aapl.get_call_data(expiry=date(2016, 1, 1))
        O = O.loc[(slice(None), slice(None), right), :]
        O = O.rename(columns={'Underlying_Price':'S'})
        # years to expiry, computed as difference between expiry (in MultiIndex) and today.
        O['ttm'] = [(x[1].to_datetime().date() - date.today()).days / 365 for x in O.index.tolist()]
        O['K/S'] = [x[0] / O.S[0] for x in O.index.tolist()]  # moneyness, see (see p.439, table 20.2)
        O['r'] = YieldCurve().interp(O['ttm'])  # appropriate interest rate (matching expiry), interpolated from yield curve

        # Black-Scholes model-implied volatility
        O['BS-IV'] = O.apply(lambda o: self.BS_IV(px=o.Last, S=o.S, r=o.r, q=0, T=o.ttm, K=o.name[0], right=o.name[2]),  axis=1)
        O.IV = tuple(float(x.strip('%'))/100 for x in O.IV)
        O['BS-IV - IV'] = O['BS-IV'] - O.IV   # difference between BS-implied volatility and imp.vol. provided by pandas.Options() class
        O.PctChg = tuple(float(x.strip('%'))/100 for x in O.PctChg)

        O.to_csv(right+'.csv')

        self.symbol, self.right, self.O = symbol, right, O

    def pxBS(self, right='call', S=586.08, K=585, T=.11, vol=.2, r=0.0002, q=0):
        """ Computes price from Black-Scholes model

        :param right: call or put
        :type right: str
        :param S: current (last) stock price of an underlying
        :type S: float
        :param K: option strike price
        :type K: float
        :param T: time to expiry (in years)
        :type T: float
        :param vol: volatility of an underlying (as a ratio, not %)
        :type vol: float
        :param r: interest rate (matching expiry, T), as a ration, not as %
        :type r: float
        :param q: dividend yield
        :type q: float
        :return: BSM price
        :rtype: float

        ..seealso::
            http://www.codeandfinance.com/finding-implied-vol.html
        """
        from scipy.stats import norm
        from math import sqrt, exp, log

        N = norm.cdf
        d1 = (log(S / K) + (r + vol * vol / 2.) * T)/(vol * sqrt(T))
        d2 = d1 - vol * sqrt(T)
        signCP = 1 if right.lower() == 'call' else -1
        return signCP * ( S * exp(-q * T) * N(signCP * d1) - K * exp(-r * T) * N(signCP * d2))
        # pxBS_eu()

    def BS_IV(self, px, right='call', S=586.08, K=585, T=.11, r=0.0002, q=0):
        """ Computes BS-implied volatility.

        Use scipy.optimize.minimize_scalar() to find volatility that gives the desired price px.
        Use BS model in this optimization.

        :param px: user provided price (market price, etc.)
        :type px: float
        :param right: call or put
        :type right: str
        :param S: current (last) stock price of an underlying
        :type S: float
        :param K: option strike price
        :type K: float
        :param T: time to expiry (in years)
        :type T: float
        :param r: interest rate (matching expiry, T), as a ration, not as %
        :type r: float
        :param q: dividend yield
        :type q: float
        :return: BS-implied volatility, as a ration (not as %)
        :rtype: float

        .. seealso::
            http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.root.html
            http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html#scipy.optimize.minimize_scalar
        """
        from scipy.optimize import minimize_scalar

        return minimize_scalar(
            lambda v: abs(self.pxBS(right, S, K, T, v, r, q) - px),
            bounds=(0., 10), method='bounded', options={'maxiter': 50, 'disp': 0}).x
        # imp_vol(px=17.5)

#This class computes PV_surface(strike volatility,T) of a variance swap on a given asset.
class varswp_px():
    
    def __init__(self, tkr='AAPL'):
    
        from pandas.io.data import read_csv
        from numpy import exp
        
        #Load Option market data for equity. 
        #Computes and appends interest rates and BS-IV for calls; puts.
        #Then writes data to file in working directory.
        #Important: python requires write permission to the working directory.
        Option(symbol=tkr,right='call')  
        Option(symbol=tkr,right='put') 

        #Reload put and call data separately from file.
        #Important: python requires read permission from the working directory.
        CDATA = read_csv('call.csv')
        PDATA = read_csv('put.csv')
        #Procedure of writing and reading to and from .csv will turn all MultiIndex labels into ordinary DataFrame columns.
        #i.e. Strike prices used to be a DataFrame index, but now it is stored as values in a DataFrame column.
        #We already eliminated the benefit of MultiIndex by storing calls and puts separately.
        #The index for CDATA; PDATA will be consecutive integers. 
        
        self.star = CDATA['S'][0] #The Underlying Px at time of assessing the market data is the same for all options available.
        self.ttm = CDATA['ttm'].unique() #Assess unique times to maturity (TTM) of Options on the market.

        self.evar = () #Stores tuple of expected variances for each unique TTM.        
        self.explist = () #Stores tuple of rate factors for each unique TTM.
        for i in range(len(self.ttm)): #Carry out operations and computations for each unique TTM.     
            #Initialize variables for operations.
            self.T = CDATA['ttm'].unique()[i] #Select a time from unique times.    
            self.r = CDATA['r'][CDATA['ttm']==self.T].values[0] #Compute the interest rate for the selected time.
            self.expe = exp(self.r*self.T) #Time exponent.
            self.explist = self.explist + (self.expe,)  #Store time exponent for future use.
            self.fz = self.star*self.expe #F_0 in Hull pp. 614.
            #Restore the DataFrames to pre-filtered state.
            self.CD = CDATA
            self.PD = PDATA
            #Operations on the Data.
            self.T_select() #Filter for TTM.
            self.R_star() #Compute the S*, the highest-Strike below the Future price of the asset.
            self.M_select() #Filter for rights on S*?K: From Hull pp. 614: call if K>S*; put if K<S*; .5(c+p) if K==S*. 
            self.O_count() #Counts the Options after filtering.
            self.pxCP() #Computes the average price of puts and calls for the special case of K==S* 
            self.evalint() #Evaluates the integral by summation as done on pp. 614.
            self.exvar() #Compute the expected variance of the asset price over the selected time and stores in tuple.
        
        self.plot_px() #Plots the present value (PV) surface on TTM and strike volatility.
            
    def T_select(self):
            
        self.CD = self.CD[self.CD['ttm']==self.T]
        self.PD = self.PD[self.PD['ttm']==self.T]
        
    def R_star(self):
        
        self.styx = max(self.CD['Strike'][self.CD['Strike']<=self.fz])

    def M_select(self):
        
        self.CP = self.CD[self.CD['Strike']==self.styx]
        self.PC = self.PD[self.PD['Strike']==self.styx]
        self.CD = self.CD[self.CD['Strike']>self.styx]
        self.PD = self.PD[self.PD['Strike']<self.styx]    
        
    def O_count(self):
        
        self.cind = self.CD.index
        self.pind = self.PD.index
            
    def pxCP(self):
        
        self.cppx = .5*(self.CP['Last'].values[0] + self.PC['Last'].values[0])
            
    def evalint(self):
        
        from pandas.io.data import Series
        
        cvind = Series(self.cind[1:],index = self.cind[:-1])
        pvind = Series(self.pind[1:],index = self.pind[:-1])
        
        cevil = sum(tuple(map(lambda i: (self.CD['Strike'][cvind[i]]-self.CD['Strike'][i])/ \
            self.CD['Strike'][i]**2*self.expe*self.CD['Last'][i], self.cind[:-1])))
        pevil = sum(tuple(map(lambda i: (self.PD['Strike'][pvind[i]]-self.PD['Strike'][i])/ \
            self.PD['Strike'][i]**2*self.expe*self.PD['Last'][i], self.pind[:-1])))        
        jevil = (self.styx-self.PD['Strike'][self.pind[len(self.pind)-1]])/ \
            self.PD['Strike'][self.pind[len(self.pind)-1]]**2* \
            self.expe*self.PD['Last'][self.pind[len(self.pind)-1]]
        kevil = (self.CD['Strike'][self.cind[0]]-self.styx)/self.styx**2* \
            self.expe*self.cppx
    
        self.evil = cevil + pevil + jevil + kevil
            
    def exvar(self):
        
        from numpy import log
        
        self.evar = self.evar + (2./self.T*log(self.fz/self.styx) - \
            2./self.T*(self.fz/self.styx-1.) + \
            2./self.T*self.evil,)
            
    def plot_px(self):
        
        import matplotlib.pyplot as plt
        from numpy import linspace, meshgrid, asarray
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        vol_K = linspace(.01,1,100)
        gvol_K, tmat = meshgrid(vol_K,self.ttm)
        gvol_K = asarray(gvol_K)
        tmat = asarray(tmat)
        
        payoff = ()
        for i in range(len(vol_K)):
            plist_K = ()
            for j in range(len(self.ttm)):
                plist_K = plist_K + ( 1/2/vol_K[i]*(self.evar[j] - vol_K[i]**2)/self.explist[j],)
            payoff = payoff + (plist_K,)
        payoff = asarray(payoff)

        ax.plot_surface(gvol_K,tmat,payoff.transpose(),\
            rstride=1, cstride=1, cmap='ocean', linewidth=0, antialiased=False)
        ax.set_xlabel('Strike volatility')
        ax.set_ylabel('ttm [yr]')
        ax.set_zlabel('Price [L_vol]')
        ax.set_title('PV of variance swap on '\
            +self.CD['Root'][self.cind[0]]+' stock')

        plt.show()

varswp_px()

