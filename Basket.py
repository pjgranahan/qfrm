import numpy.random
import numpy as np

try: from qfrm.European import *  # production:  if qfrm package is installed
except:   from European import *  # development: if not installed and running from source


class Basket(European):
    """ European option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, mu = (0.1,0.2,0.5), weight = (0.5,0.3,0.2), corr = [[1,0,0],[0,1,0],[0,0,1]], **kwargs):
        """ Wrapper function that calls appropriate valuation method.

        Parameters
        ----------
        mu : tuple
                Expected return of assets in a basket
        weight: tuple
                 Weights of assets in a basket
        Corr: list
                 Correlation Matrix of assets in a basket
        kwargs : dict
            Keyword arguments (``method``, ``nsteps``, ``npaths``, ``keep_hist``, ``rng_seed``, ...)
            are passed to the parent. See ``European.calc_px()`` for details.

        Returns
        -------
        self : Basket
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object).


        Notes
        -----

        *References:*

        - Basket Options (Lecture 3, MFE5010 at NUS), `Lim Tiong Wee, 2001 <http://1drv.ms/1NFu0uR>`_
        - Calculation of Moments for Valuing Basket Options, `Technical Note #28, J.C.Hull <http://www-2.rotman.utoronto.ca/~hull/technicalnotes/TechnicalNote28.pdf>`_
        - Verify examples with `Online option pricer <http://www.infres.enst.fr/~decreuse/pricer/en/index.php?page=panier.html>`_.
        - Pricing of basket options. `Online tool. <http://www.infres.enst.fr/~decreuse/pricer/en/index.php?page=panier.html>`_

        Examples
        -------

        >>> s = Stock(S0=(42,55,75), vol=(.20,.30,.50))
        >>> o = Basket(ref=s, right='call', K=40, T=.5, rf_r=.1, desc='Hull p.612')

        >>> o.calc_px(method='MC',mu=(0.05,0.1,0.05),weight=(0.3,0.5,0.2),corr=[[1,0,0],[0,1,0],[0,0,1]],\
        npaths=10,nsteps=10).px_spec # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px: 14.394869309...

        >>> s = Stock(S0=(50,85,65,80,75), vol=(.20,.10,.05,.20,.30))
        >>> o = Basket(ref=s, right='put', K=80, T=1, rf_r=.05, desc='Hull p.612')

        >>> o.pxMC(mu=(0.05,0,0.1,0,0),weight=(0.2,0.2,0.2,0.2,0.2),corr=[[1,0,0,0.9,0],\
        [0,1,0,0,0],[0,0,1,-0.1,0],[0.9,0,-0.1,1,0],[0,0,0,0,1]],\
        npaths=10,nsteps=10)   # save interim results to self.px_spec. Equivalent to repr(o)
        5.865304293

        >>> s = Stock(S0=(30,50), vol=(.20,.15))
        >>> o = Basket(ref=s, right='put', K=55, T=3, rf_r=.05, desc='Hull p.612')

        >>> o.pxMC(mu=(0.06,0.05), weight=(0.4,0.6),corr=[[1,0.7],[0.7,1]], npaths=10, nsteps=10)
        6.147189494

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> O = Series([o.update(T=t).calc_px(method='MC',mu=(0.06,0.05),weight=(0.4,0.6),\
        corr=[[1,0.7],[0.7,1]],npaths=2,nsteps=3).px_spec.px for t in expiries], expiries)
        >>> O.plot(grid=1, title='Price vs expiry (in years)') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()

        :Authors:
          Hanting Li <hl45@rice.edu>

        """

        self.save2px_spec(mu=mu, weight=weight, corr=corr, **kwargs)
        return getattr(self, '_calc_' + self.px_spec.method.upper())()

    def _calc_BS(self):
        """ Internal function for option valuation.  See ``calc_px()`` for complete documentation. """
        return self

    def _calc_LT(self):
        """ Internal function for option valuation. See ``calc_px()`` for complete documentation.       """
        return self

    def _calc_MC(self, keep_hist=False):
        """ Internal function for option valuation. See ``calc_px()`` for complete documentation.

        :Authors:
          Hanting Li <hl45@rice.edu>
        """
        _ = self

        # Define the parameters
        S0, vol = _.ref.S0, _.ref.vol
        mu, corr, n, m = _.px_spec.mu, _.px_spec.corr, _.px_spec.nsteps, _.px_spec.npaths

        # Compute Deltat and number of assets
        deltat = _.T/n
        Nasset = len(vol)

        # Compute the stock price at t
        def calS(St,mu,sigma,param):
            deltaS = mu*St*deltat + sigma*St*param*np.sqrt(deltat)
            S_update = St+deltaS
            return(S_update.item())

        # Generate one path
        def one_path(S0,mu,vol,param):
            S0 = (S0,)
            for i in range(n):
                parami = param[i]
                S0 = S0 + (calS(S0[len(S0)-1],mu,vol,parami),)
            return(S0)

        # Define n paths matrix
        priceNpath = ()

        # Compute covariance matrix from correlation matrix
        covM = np.dot(np.dot(np.diag(vol),(corr)),np.diag(vol))

        # Set seed
        numpy.random.seed(10987)
        # Generate random numbers
        param = numpy.random.multivariate_normal(np.repeat(0,Nasset),covM,n)
        param = tuple(zip(*param))

        # Generate N paths
        for i in range(m):
            price = list(map(one_path,S0,mu,vol,param))
            wprice = np.transpose(np.matrix(price))*np.transpose(np.matrix(_.px_spec.weight))
            wprice = tuple(wprice.ravel().tolist()[0])
            priceNpath = priceNpath + (wprice,)

        # Terminal Payoff
        payoff = max(0,_.signCP*(np.mean(tuple(zip(*priceNpath))[n])-_.K))

        self.px_spec.add(px=float(payoff*np.exp(-_.rf_r*_.T)), sub_method='standard; Hull p.612')

        return self

    def _calc_FD(self, nsteps=3, npaths=4, keep_hist=False):
        """ Internal function for option valuation. See ``calc_px()`` for complete documentation.      """
        return self
