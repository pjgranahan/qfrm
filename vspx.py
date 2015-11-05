# -*- coding: utf-8 -*-

#This class computes the expected variance for options on a given asset of a given maturity.
class volswp_px():
    
    def __init__(self):
    
        from pandas.io.data import read_csv
        from numpy import exp
        
        CDATA = read_csv('call.csv')
        PDATA = read_csv('put.csv')
        
        self.T = CDATA['ttm'][0]        
        self.r = CDATA['r'][0]
        self.star = CDATA['Underlying_Price'][0]
        self.expe = exp(self.r*self.T)
        self.fz = self.star*self.expe

        self.CD = CDATA
        self.PD = PDATA
        
        self.T_select()
        self.R_star()
        self.M_select()
        self.O_count()
        #self.pxBS() # is inactive: using market prices
        self.pxCP()
        self.evalint()
        self.exvar()
            
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

    def pxBS(self):

        from pandas.io.data import Series
        from numpy import log, exp
        from scipy.stats import norm
        
        cd1 = Series(tuple(map(lambda i: (log(self.star/self.CD['Strike'][i]) + \
            (self.r + self.CD['BS-IV'][i]**2/2)*self.T)/self.CD['BS-IV'][i]*self.T**.5 \
            , self.cind)),index = self.cind)      
        pd1 = Series(tuple(map(lambda i: (log(self.star/self.PD['Strike'][i]) + \
            (self.r + self.PD['BS-IV'][i]**2/2)*self.T)/self.PD['BS-IV'][i]*self.T**.5 \
            , self.pind)),index = self.pind)                            
        cpd1 = log(self.star/self.styx) + (self.r + self.CP['BS-IV'].values[0]**2/2)* \
            self.CP['BS-IV'].values[0]*self.T**.5
            
        cd2 = Series(tuple(map(lambda i: (log(self.star/self.CD['Strike'][i]) + \
            (self.r - self.CD['BS-IV'][i]**2/2)*self.T)/self.CD['BS-IV'][i]*self.T**.5 \
            , self.cind)),index = self.cind)                
        pd2 = Series(tuple(map(lambda i: (log(self.star/self.PD['Strike'][i]) + \
            (self.r - self.PD['BS-IV'][i]**2/2)*self.T)/self.PD['BS-IV'][i]*self.T**.5 \
            , self.pind)),index = self.pind)  
        cpd2 = log(self.star/self.styx) + (self.r + self.CP['BS-IV'].values[0]**2/2)* \
            self.CP['BS-IV'].values[0]*self.T**.5
            
        cNd1 = Series(tuple(map(lambda i: norm.cdf(cd1[i]), self.cind)),index = self.cind)   
        pNd1 = Series(tuple(map(lambda i: norm.cdf(pd1[i]), self.pind)),index = self.pind)   
        cpNd1 = norm.cdf(cpd1)
        
        cNd2 = Series(tuple(map(lambda i: norm.cdf(cd2[i]), self.cind)),index = self.cind)   
        pNd2 = Series(tuple(map(lambda i: norm.cdf(pd2[i]), self.pind)),index = self.pind)   
        cpNd2 = norm.cdf(cpd2)
        
        self.PD['europx'] = Series(tuple(map(lambda i: self.star*pNd1[i] - \
            self.PD['Strike'][i]*exp(-self.r*self.T)*pNd2[i], self.pind)),index = self.pind)  
        self.CD['europx'] = Series(tuple(map(lambda i: -self.star*(1-cNd1[i]) + \
            self.CD['Strike'][i]*exp(-self.r*self.T)*(1-cNd2[i]), self.cind)),index = self.cind)  
        self.cppx = (self.star*cpNd1 - self.styx/self.expe*cpNd2 - \
            self.star*(1-cpNd1) + self.styx/self.expe*(1-cpNd2))/2.
            
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
        
        self.evar = 2./self.T*log(self.fz/self.styx) - \
            2./self.T*(self.fz/self.styx-1.) + \
            2./self.T*self.evil
            
print(volswp_px().evar)
        

