import matplotlib.pyplot as plt
import numpy as np

try: from qfrm.OptionValuation import *  # production:  if qfrm package is installed
except:   from OptionValuation import *  # development: if not installed and running from source


class VarianceSwap(OptionValuation):
    """ VarianceSwap option class.

    Inherits all methods and properties of OptionValuation class.
    The Variance Swap's conceptual cousins have little to do with what is thought of as "options," they include the
    forward-rate agreement. Like a forward-rate agreement on interest rates, variance swaps exchange, at the
    contracted maturity date, the realized (volatility rate)**2 with a fixed (volatility rate)**2. The dollar value 
    is proportional to the principle of the "rate" agreement.The bigger the principle, the bigger the magnitude of 
    the money exchanged for the "bet" at maturity. As a type of FRA more so than an "option", the price can be 
    positive or negative depending on the size of the agreed-to variance versus the expected variance over (0,T). 
    This isn't an error of pricing formulae, but rather, like other types of forwards, this directly illustrates 
    the zero-sum nature of this option. If the price is negative for you receiving the fixed variance, it must be 
    positive for the counterparty receiving the realized variance, and vice-versa.      
    """
    
    def calc_px(self, method='BS', K=(280.,300.,320.,340.,360.,380.,400.), vol=(0.2,0.2,0.2,0.3,0.3,0.3,0.3), 
                L_Var=10000000., Var_K=0.1, nsteps=None, npaths=None, keep_hist=False):
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

        K : float
                Required. Must be a vector (e.g. 1-D tuple, list, array, ...) of European strike prices to estimate 
                variance of the underlying. 
        vol : float
                Required. Must be a vector (e.g. 1-D tuple, list, array, ...) of volatilities implied by European 
                options with K strikes.
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
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object).



        Notes
        -----

        **BS Notes**
        Referenced example is example 26.4 on pages 613-614 in OFOD, Hull, 9e.


        Examples
        --------
        
        **BS**

        Pricing by BSM

        >>> s = Stock(355)
        >>> o = VarianceSwap(ref=s, rf_r=0.03, T=1.)
        >>> o.pxBS() # doctest: +ELLIPSIS
        -489162.761...
        
        Changing the stock price

        >>> VarianceSwap(ref=Stock(310), rf_r=0.03, T=1.).pxBS() # doctest: +ELLIPSIS
        -504216.712...
        
        >>> VarianceSwap(ref=Stock(500), rf_r=0.03, T=1.).pxBS() # doctest: +ELLIPSIS
        -1404368.57...
        
        Explicit input parameters

        >>> VarianceSwap(ref=Stock(290), rf_r=0.03, T=1.).pxBS(K=(280.,300.,320.,340.,360.,380.,400.),
        ... vol=(0.2,0.2,0.2,0.3,0.3,0.3,0.3), L_Var=10000000., Var_K=0.01) # doctest: +ELLIPSIS
        312551.288...
        
        Referenced example

        >>> Karr = (800,850,900,950,1000,1050,1100,1150,1200)
        >>> varr = (.29,.28,.27,.26,.25,.24,.23,.22,.21)
        >>> VarianceSwap(ref=Stock(S0=1020,q=.01), rf_r=.04, T=.25).pxBS(K=Karr, vol=varr, L_Var=100.,
        ... Var_K=.045) # doctest: +ELLIPSIS
        1.69073994...
        
        Price vs. the strike volatility curve - example of vectorization of price calculation

        >>> s2K = np.linspace(0.01,0.2,200)
        >>> o = VarianceSwap(ref=Stock(355),rf_r=0.03, T=1.)
        >>> px = tuple(map(lambda i: o.pxBS(Var_K=s2K[i]**2)/1000, range(s2K.shape[0])))
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> ax.plot(s2K,px,label='Variance Swap') # doctest: +ELLIPSIS
        [<matplotlib.lines.Line2D object at 0x...>]
        >>> ax.set_title('Price of Variance Swap vs Vol_K for L_Var = '+str(o.L_Var/1000)+'M');  # doctest: +ELLIPSIS
        <matplotlib.text.Text object at 0x0...>
        >>> ax.set_ylabel('Px [x000]');           # doctest: +ELLIPSIS
        <matplotlib.text.Text object at 0x0...>
        >>> ax.set_xlabel('volatility strike');   # doctest: +ELLIPSIS
        <matplotlib.text.Text object at 0x0...>
        >>> ax.grid(); ax.legend(); plt.show();   # doctest: +ELLIPSIS
        <matplotlib.legend.Legend object at 0x...>

        :Authors:
            Andy Liao <Andy.Liao@rice.edu>
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

        :Authors: 
            Andy Liao <Andy.Liao@rice.edu>

        """

        _ = self
        N = Util.norm_cdf

        d1 = tuple(map(lambda i: (np.log(_.ref.S0 / _.K[i]) + (_.rf_r + _.ref.vol[i] ** 2 / 2.) * _.T)/(_.ref.vol[i] * \
            np.sqrt(_.T)), range(len(_.K))))
        d2 = tuple(map(lambda i: d1[i] - _.ref.vol[i] * np.sqrt(_.T), range(len(_.K)))) 

        #Compute the call and put prices for the sequence of options with strikes K_i with BSM.
        px_call = tuple(map(lambda i: _.ref.S0 * np.exp(-_.ref.q * _.T) * N(d1[i]) - _.K[i] * np.exp(-_.rf_r * _.T )
            * N(d2[i]), range(len(_.K))))
        px_put = tuple(map(lambda i: -_.ref.S0 * np.exp(-_.ref.q * _.T) * N(-d1[i]) + _.K[i] * np.exp(-_.rf_r * _.T) \
            * N(-d2[i]), range(len(_.K))))

        #The machinery of the variance swap pricing, from Hull 9. ed. p. 613-614.
        Ka = np.asarray(_.K)
        cpxa = np.asarray(px_call)
        ppxa = np.asarray(px_put)
        expr = np.exp(_.rf_r * _.T)
        expq = np.exp(_.ref.q * _.T) 
        fz = _.ref.S0 * expr/expq         
        styx = np.asarray(Ka[Ka<=fz]).max()
        Qx = np.zeros(Ka.shape)
        Qx[Ka<styx] = ppxa[Ka<styx]
        Qx[Ka==styx] = .5 * (ppxa[Ka==styx] + cpxa[Ka==styx])
        Qx[Ka>styx] = cpxa[Ka>styx]
        Kint = sum(tuple(map(lambda i: .5*(Ka[i+1] - Ka[i-1])/Ka[i]**2 * expr * Qx[i], range(1,max(Ka.shape)-1))))
        Kint = Kint + (Ka[1] - Ka[0])/Ka[0]**2 * expr * Qx[0] + \
            (Ka[max(Ka.shape)-1] - Ka[max(Ka.shape)-2])/Ka[max(Ka.shape)-1]**2 * expr * Qx[max(Ka.shape)-1]
        Var_E = (2./_.T) * (np.log(fz/styx) - (fz/styx - 1.) + Kint)
        px = _.L_Var * (Var_E - _.Var_K)/expr

        #adds the BSM price of calls and puts from the vector K,vol to px_call; px_put, and the price of the variance 
        #swap to px.
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
        

