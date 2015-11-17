from qfrm import *

class VarianceSwap(OptionValuation):
    """ VarianceSwap option class.

    Inherits all methods and properties of OptionValuation class.
    """
    
    def calc_px(self, method='BS', K=(280.,300.,320.,340.,360.,380.,400.), vol=(0.2,0.2,0.2,0.3,0.3,0.3,0.3), 
                L_Var=10000000., Var_K=0.1, nsteps=None, npaths=None, keep_hist=False):
        """ Wrapper function that calls appropriate valuation method.

        User passes parameters to calc_px, which saves them to local PriceSpec object
        and calls specific pricing function (_calc_BS,...).
        This makes significantly less docstrings to write, since user is not interfacing pricing functions,
        but a wrapper function calc_px().

        Parameters
        ----------
        method : str
                Required. Indicates a valuation method to be used: 'BS', 'LT', 'MC', 'FD'
        K : float
                Required. Must be a vector (e.g. 1-D tuple, list, array, ...) of European strike prices to estimate variance of the underlying. 
        vol : float
                Required. Must be a vector (e.g. 1-D tuple, list, array, ...) of volatilities implied by European options with K strikes.
        L_Var : float
                Required. The variance notional, i.e. the size of the bet
        Var_K : float
                Required. The fixed variance rate against which the realized variance is swapped at maturity.
        nsteps : int
                LT, MC, FD methods require number of times steps
        npaths : int
                MC, FD methods require number of simulation paths
        keep_hist : bool
                If True, historical information (trees, simulations, grid) are saved in self.px_spec object.

        Returns
        -------
        self : VarianceSwap

        .. sectionauthor:: Oleg Melkinov; Andy Liao

        Notes
        -----

        Examples
        -------
        
        >>> #Pricing by BSM
        >>> s = Stock(355)
        >>> o = VarianceSwap(ref=s, rf_r=0.03, T=1.)
        >>> o.calc_px().px_spec.px
        -489162.761418
        >>> ##Changing the stock price
        >>> VarianceSwap(ref=Stock(310), rf_r=0.03, T=1.).calc_px().px_spec.px
        -504216.712984
        >>> VarianceSwap(ref=Stock(500), rf_r=0.03, T=1.).calc_px().px_spec.px
        -1404368.57684
        >>> ##Explicit input parameters
        >>> VarianceSwap(ref=Stock(290), rf_r=0.03, T=1.).calc_px(method='BS', K=(280.,300.,320.,340.,360.,380.,400.), \
        >>>     vol=(0.2,0.2,0.2,0.3,0.3,0.3,0.3), L_Var=10000000., Var_K=0.1).px_spec.px
        -560849.691576
        >>> ##Example 26.4 on Hull p 614
        >>> Karr = (800,850,900,950,1000,1050,1100,1150,1200)
        >>> varr = (.29,.28,.27,.26,.25,.24,.23,.22,.21)
        >>> VarianceSwap(ref=Stock(S0=1020,q=.01), rf_r=.04, T=.25).calc_px(method='BS', K=Karr, \
        >>>     vol=varr, L_Var=100., Var_K=.045).px_spec.px
        1.69073994549
        >>> ##Price vs. the strike volatility curve - example of vectorization of price calculation
        >>> import matplotlib.pyplot as plt
        >>> from numpy import linspace
        >>> s2K = linspace(0.01,0.2,200)
        >>> o = VarianceSwap(ref=Stock(355),rf_r=0.03, T=1.)
        >>> px = tuple(map(lambda i: o.calc_px(Var_K=s2K[i]**2).px_spec.px/1000, range(s2K.shape[0])))
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> ax.plot(s2K,px,label='Variance Swap')
        >>> ax.set_title('Price of Variance Swap vs Vol_K for L_Var = '+str(o.L_Var/1000)+'M')
        >>> ax.set_xlabel('Px [\'000]')
        >>> ax.set_xlabel('volatility strike')
        >>> ax.grid()
        >>> ax.legend()
        >>> plt.show()
        >>> px
        (480.31232659661396, 480.11816887427602, 479.90631812208477, ... 100.48162094750336, 96.801936344057935, 93.10455871075915)
         

        """
        
        #should override OptionValuation K; Stock vol with vector values
        self.K, self.ref.vol, self.L_Var, self.Var_K = K, vol, L_Var, Var_K

        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: VarianceSwap

        .. sectionauthor:: Andy Liao

        """
        from scipy.stats import norm
        from numpy import sqrt, exp, log, asarray, zeros

        _ = self
        d1 = tuple(map(lambda i: (log(_.ref.S0 / _.K[i]) + (_.rf_r + _.ref.vol[i] ** 2 / 2.) * _.T)/(_.ref.vol[i] * sqrt(_.T)), range(len(_.K))))
        d2 = tuple(map(lambda i: d1[i] - _.ref.vol[i] * sqrt(_.T), range(len(_.K)))) 

        px_call = tuple(map(lambda i: _.ref.S0 * exp(-_.ref.q * _.T) * norm.cdf(d1[i]) - _.K[i] * exp(-_.rf_r * _.T ) * norm.cdf(d2[i]), \
            range(len(_.K))))
        px_put = tuple(map(lambda i: -_.ref.S0 * exp(-_.ref.q * _.T) * norm.cdf(-d1[i]) + _.K[i] * exp(-_.rf_r * _.T) * norm.cdf(-d2[i]), \
            range(len(_.K))))

        #The machinery of the variance swap pricing, from Hull 9. ed. p. 613-614.
        Ka = asarray(_.K)
        cpxa = asarray(px_call)
        ppxa = asarray(px_put)
        expr = exp(_.rf_r * _.T)
        expq = exp(_.ref.q * _.T) 
        fz = _.ref.S0 * expr/expq         
        styx = asarray(Ka[Ka<=fz]).max()
        Qx = zeros(Ka.shape)
        Qx[Ka<styx] = ppxa[Ka<styx]
        Qx[Ka==styx] = .5 * (ppxa[Ka==styx] + cpxa[Ka==styx])
        Qx[Ka>styx] = cpxa[Ka>styx]
        Kint = sum(tuple(map(lambda i: .5*(Ka[i+1] - Ka[i-1])/Ka[i]**2 * expr * Qx[i], range(1,max(Ka.shape)-1))))
        Kint = Kint + (Ka[1] - Ka[0])/Ka[0]**2 * expr * Qx[0] + \
            (Ka[max(Ka.shape)-1] - Ka[max(Ka.shape)-2])/Ka[max(Ka.shape)-1]**2 * expr * Qx[max(Ka.shape)-1]
        Var_E = (2./_.T) * (log(fz/styx) - (fz/styx - 1.) + Kint)
        px = _.L_Var * (Var_E - _.Var_K)/expr

        #adds the BSM price of calls and puts from the vector K,vol to px_call; px_put, and the price of the variance swap to px.
        self.px_spec.add(px=px, sub_method='Hull p. 613-614', px_call=px_call, px_put=px_put, d1=d1, d2=d2)
        
        return self

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: VarianceSwap

        .. sectionauthor::
        
        """
        return self

    def _calc_MC(self, nsteps=3, npaths=4, keep_hist=False):
        """ Internal function for option valuation.

        Returns
        -------
        self: VarianceSwap

        .. sectionauthor::

        """
        return self

    def _calc_FD(self, nsteps=3, npaths=4, keep_hist=False):
        """ Internal function for option valuation.

        Returns
        -------
        self: VarianceSwap

        .. sectionauthor::

        """
        return self
        
#Pricing by BSM
s = Stock(355)
o = VarianceSwap(ref=s, rf_r=0.03, T=1.)
print(o.calc_px().px_spec.px)
##Changing the stock price
print(VarianceSwap(ref=Stock(310),rf_r=0.03, T=1.).calc_px().px_spec.px)
print(VarianceSwap(ref=Stock(500),rf_r=0.03, T=1.).calc_px().px_spec.px)
##Explicit input parameters
print(VarianceSwap(ref=Stock(290),rf_r=0.03, T=1.).calc_px(method='BS', K=(280.,300.,320.,340.,360.,380.,400.), \
                vol=(0.2,0.2,0.2,0.3,0.3,0.3,0.3), L_Var=10000000., Var_K=0.1).px_spec.px)
##Example 26.4 on Hull p 614
Karr = (800,850,900,950,1000,1050,1100,1150,1200)
varr = (.29,.28,.27,.26,.25,.24,.23,.22,.21)
print(VarianceSwap(ref=Stock(S0=1020,q=.01), rf_r=.04, T=.25).calc_px(method='BS', K=Karr, \
    vol=varr, L_Var=100., Var_K=.045).px_spec.px)
##Price vs. the strike volatility curve - example of vectorization of price calculation
import matplotlib.pyplot as plt
from numpy import linspace
s2K = linspace(0.01,0.20,200)
o = VarianceSwap(ref=Stock(355),rf_r=0.03, T=1.)
px = tuple(map(lambda i: o.calc_px(Var_K=s2K[i]**2).px_spec.px/1000, range(s2K.shape[0])))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(s2K,px,label='Variance Swap')
ax.set_title('Price of Variance Swap vs Vol_K for L_Var = '+str(o.L_Var/1000)+'M')
ax.set_ylabel('Px [\'000]')
ax.set_xlabel('volatility strike')
ax.grid()
ax.legend()
plt.show()
#print(px)
