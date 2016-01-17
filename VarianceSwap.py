import numpy as np

try: from qfrm.European import *  # production:  if qfrm package is installed
except:   from European import *  # development: if not installed and running from source


class VarianceSwap(European):
    """ `VarianceSwap <https://en.wikipedia.org/wiki/Variance_swap>`_ exotic option class.

    The Variance Swap's conceptual cousins have little to do with what is thought of as "options," they include the
    forward-rate agreement. Like a forward-rate agreement on interest rates, variance swaps exchange, at the
    contracted maturity date, the realized (volatility rate)**2 with a fixed (volatility rate)**2. The dollar value 
    is proportional to the principle of the "rate" agreement.The bigger the principle, the bigger the magnitude of 
    the money exchanged for the "bet" at maturity. As a type of FRA more so than an "option", the price can be 
    positive or negative depending on the size of the agreed-to variance versus the expected variance over (0,T). 
    This isn't an error of pricing formulae, but rather, like other types of forwards, this directly illustrates 
    the zero-sum nature of this option. If the price is negative for you receiving the fixed variance, it must be 
    positive for the counterparty receiving the realized variance, and vice-versa.

    Notes
    -----

    Strike prices ``K`` and corresponding volatilities ``vol`` passed to object constructor need to be same-size iterables of numbers.

    - ``K``: European strike prices to estimate variance of the underlying.

    - ``vol``: volatilities implied by European options with K strikes.
    """
    
    # def calc_px(self, K=(280,300,320,340,360,380,400), vol=(.2,.2,.2,.3,.3,.3,.3), L_Var=10**7, Var_K=.1, **kwargs):
    def calc_px(self, L_Var=10**7, Var_K=.1, **kwargs):
        """ Wrapper function that calls appropriate valuation method.

        Parameters
        ----------
        L_Var : number
                Required. The variance notional amount, i.e. the size of the bet.
        Var_K : number
                Required. The fixed variance rate against which the realized variance is swapped at maturity.
        kwargs : dict
            Keyword arguments (``method``, ``nsteps``, ``npaths``, ``keep_hist``, ``rng_seed``, ..)
            are passed to the parent. See ``European.calc_px()`` for details.

        Returns
        -------
        self : VarianceSwap
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object).


        Notes
        -----

        *References:*

        - See example 26.4 on pp.613-614, `OFOD <http://www-2.rotman.utoronto.ca/~hull/ofod/index.html>`_, J.C.Hull, 9e, 2014
        - Valuation of a Variance Swap, `Technical Note #22, J.C.Hull <http://www-2.rotman.utoronto.ca/~hull/technicalnotes/TechnicalNote22.pdf>`_

        Examples
        --------
        **BS**

        >>> s = Stock(S0=355, vol=(.2,.2,.2,.3,.3,.3,.3))
        >>> o = VarianceSwap(ref=s, rf_r=0.03, T=1, K=(280,300,320,340,360,380,400))
        >>> o.pxBS() # doctest: +ELLIPSIS
        -489162.761...
        
        Changing the stock price

        >>> s.S0 = 310
        >>> VarianceSwap(clone=o, ref=s).pxBS() # doctest: +ELLIPSIS
        -504216.712...

        >>> s.S0 = 500
        >>> VarianceSwap(clone=o, ref=s).pxBS() # doctest: +ELLIPSIS
        -1404368.57...
        
        Explicit input parameters

        >>> s = Stock(S0=290, vol=(0.2,.2,.2,.3,.3,.3,.3))
        >>> o = VarianceSwap(ref=s, rf_r=0.03, T=1, K=(280,300,320,340,360,380,400))
        >>> o.pxBS(L_Var=10**7, Var_K=.01); o # doctest: +ELLIPSIS
        312551.288...
        
        Referenced example

        >>> s = Stock(S0=1020, q=.01, vol=(.29,.28,.27,.26,.25,.24,.23,.22,.21))
        >>> o = VarianceSwap(ref=s, rf_r=.04, T=.25, K=(800,850,900,950,1000,1050,1100,1150,1200))
        >>> o.pxBS(L_Var=100, Var_K=.045); o # doctest: +ELLIPSIS
        1.69073994...
        
        Price vs. the strike volatility curve - example of vectorization of price calculation

        >>> from pandas import Series
        >>> s2K = np.linspace(0.01,.2,200)
        >>> s = Stock(S0=355, vol=(.2,.2,.2,.3,.3,.3,.3))
        >>> o = VarianceSwap(ref=s, rf_r=0.03, T=1, K=(280,300,320,340,360,380,400))
        >>> px = Series(map(lambda i: o.pxBS(Var_K=s2K[i]**2) / 1000, range(s2K.shape[0])), range(s2K.shape[0]))
        >>> px.plot(grid=True, title='BS price vs Vol_K for ' + o.specs + ', L_Var=' + str(str(o.px_spec.L_Var/1000)))  # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at 0x...>

        :Authors:
            Andy Liao <Andy.Liao@rice.edu>
        """
        self.save2px_spec(L_Var=L_Var, Var_K=Var_K, **kwargs)
        return getattr(self, '_calc_' + self.px_spec.method.upper())()


    def _calc_BS(self):
        """ Internal function for option valuation.

        :Authors: 
            Andy Liao <Andy.Liao@rice.edu>
        """
        _ = self;               T, K, rf_r, net_r = _.T, _.K, _.rf_r, _.net_r
        _ = self.ref;           S0, vol, q = _.S0, _.vol, _.q
        _ = self.px_spec;       L_Var, Var_K = _.L_Var, _.Var_K

        _ = self
        N = Util.norm_cdf

        d1 = tuple(map(lambda i: (np.log(S0 / K[i]) + (rf_r + vol[i] ** 2 / 2) * T)/(vol[i] * np.sqrt(T)), range(len(K))))
        d2 = tuple(map(lambda i: d1[i] - vol[i] * np.sqrt(T), range(len(K))))

        #Compute the call and put prices for the sequence of options with strikes K_i with BSM.
        px_call = tuple(map(lambda i: S0 * np.exp(-q * T) * N(d1[i]) - K[i] * np.exp(-rf_r * T )
            * N(d2[i]), range(len(K))))
        px_put = tuple(map(lambda i: -S0 * np.exp(-q * T) * N(-d1[i]) + K[i] * np.exp(-rf_r * T) * N(-d2[i]), range(len(K))))

        #The machinery of the variance swap pricing, from Hull 9. ed. p. 613-614.
        Ka = np.asarray(K)
        cpxa = np.asarray(px_call)
        ppxa = np.asarray(px_put)
        expr = np.exp(rf_r * T)
        expq = np.exp(q * T)
        fz = S0 * expr/expq         
        styx = np.asarray(Ka[Ka<=fz]).max()
        Qx = np.zeros(Ka.shape)
        Qx[Ka<styx] = ppxa[Ka<styx]
        Qx[Ka==styx] = .5 * (ppxa[Ka==styx] + cpxa[Ka==styx])
        Qx[Ka>styx] = cpxa[Ka>styx]
        Kint = sum(tuple(map(lambda i: .5*(Ka[i+1] - Ka[i-1])/Ka[i]**2 * expr * Qx[i], range(1,max(Ka.shape)-1))))
        Kint = Kint + (Ka[1] - Ka[0])/Ka[0]**2 * expr * Qx[0] + (Ka[max(Ka.shape)-1] - Ka[max(Ka.shape)-2])/Ka[max(Ka.shape)-1]**2 * expr * Qx[max(Ka.shape)-1]
        Var_E = (2./T) * (np.log(fz/styx) - (fz/styx - 1) + Kint)
        px = L_Var * (Var_E - Var_K)/expr

        #adds the BSM price of calls and puts from the vector K,vol to px_call; px_put, and the price of the variance 
        #swap to px.
        self.px_spec.add(px=px, sub_method='Hull p. 613-614', px_call=px_call, px_put=px_put, d1=d1, d2=d2)
        return self

    def _calc_LT(self):
        """ Internal function for option valuation.        """
        return self

    def _calc_MC(self, nsteps=3, npaths=4, keep_hist=False):
        """ Internal function for option valuation.        """
        return self

    def _calc_FD(self, nsteps=3, npaths=4, keep_hist=False):
        """ Internal function for option valuation.        """
        return self
        

