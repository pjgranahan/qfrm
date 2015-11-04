# -*- coding: utf-8 -*-

#This class shows the payoff from a realized (historical) volatility or variance swap.
class volswp_real():
    
    def __init__(self,tkr='GOOG',src='yahoo',start='2014-04-01',T=1./12.):
    
        from pandas.io.data import DataReader
        from datetime import datetime, timedelta
        
        deltaT = timedelta(days=T*365.25)
        sdate = datetime.strptime(start,'%Y-%m-%d')
        end = (sdate + deltaT).date().strftime('%Y-%m-%d')
        
        DATA = DataReader(tkr,src,start=start,end=end)
        date = DATA.index
        date = tuple(map(lambda i: date[i].date().strftime('%Y-%m-%d'), range(len(date))))
        price = DATA['Close'].values

        self.start = start
        self.end = end
        self.T = T
        self.ndays = len(date)    
        self.date = tuple(map(lambda i: date[i], range(self.ndays))) 
        self.price = tuple(map(lambda i: price[i], range(self.ndays)))  
        
        self.logreturns()
        self.realvol()
        
    def logreturns(self):
    
        from numpy import log
    
        self.lret = (0.0,) + tuple(map(lambda i: log(self.price[i+1]/self.price[i]) , range(self.ndays-1)))
        
    def realvol(self):
        
        self.vol_real = ((252./(self.ndays-2)) * sum(self.lret)**2.)**.5
        self.var_real = self.vol_real**2
     
    def getinfo(self):
        
        print('realized timespan in years = ' +str(round(self.T,4)))
        print('beginning date = ' +self.start)
        print('ending date = ' +self.end)
        print('number of trading days = ' +str(self.ndays))
        print('realized volatility = ' +str(self.vol_real))
        print('realized variance = ' +str(self.var_real))
        
    def getpayoff(self):
        
        import matplotlib.pyplot as plt
        from numpy import linspace

        vol_K = linspace(0,1,101)
        var_K = linspace(0,1,101)

        payvolswp = tuple(map(lambda i: self.vol_real - vol_K[i], range(len(vol_K))))     
        payvarswp = tuple(map(lambda i: (self.vol_real - var_K[i])/2/vol_K[i], range(len(var_K))))     
        
        fig = plt.figure()
        axvol = fig.add_subplot(121)
        axvol.plot(vol_K, payvolswp)
        axvol.set_xlabel('strike volatility')
        axvol.set_ylabel('payoff [L_vol]') 
        axvol.set_title('volatility swap payoff')
        axvol.grid()
        
        axvar = fig.add_subplot(122)
        axvar.plot(var_K, payvarswp)
        axvar.set_xlabel('strike variance')
        axvar.set_ylabel('payoff [L_vol]')
        axvar.set_title('variance swap payoff')
        axvar.grid()
        
        plt.tight_layout()
        plt.show()

vspr = volswp_real()
vspr.getinfo()
vspr.getpayoff()
