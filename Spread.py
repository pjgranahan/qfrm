import math
from scipy import stats
import numpy.random
import numpy as np

try: from qfrm.OptionValuation import *  # production:  if qfrm package is installed
except:   from OptionValuation import *  # development: if not installed and running from source



class Spread(OptionValuation):
    """ Spread option class.

    Inherits all methods and properties of OptionValuation class.
    """


    def calc_px(self, method='BS', S2 = None, rho = .5, nsteps=None, npaths=None, keep_hist=False):
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

        S2 : Stock
                Required. Indicated the second stock used in the spread option
        rho : float
                The correlation between the reference stock and S2
        nsteps : int
                LT, MC, FD methods require number of times steps
        npaths : int
                MC, FD methods require number of simulation paths
        keep_hist : bool
                If True, historical information (trees, simulations, grid) are saved in self.px_spec object.

        Returns
        -------
        self : Spread
            Returned object contains specifications and calculated price in embedded ``PriceSpec`` object.


        Notes
        ---------
        Verify Examples: http://www.fintools.com/resources/online-calculators/exotics-calculators/spread/


        Examples
        ---------------

        >>> s1 = Stock(S0=30.,q=0.,vol=.2)
        >>> s2 = Stock(S0=31.,q=0.,vol=.3)
        >>> o = Spread(ref = s1, rf_r = .05, right='call', K=0., T=2., seed0 = 0)
        >>> o.calc_px(method='BS',S2 = s2,rho=.4).px_spec.px
        5.409907579760095

        >>> s1 = Stock(S0=30.,q=0.,vol=.2)
        >>> s2 = Stock(S0=31.,q=0.,vol=.3)
        >>> o = Spread(ref = s1, rf_r = .05, right='put', K=0., T=2., seed0 = 0)
        >>> from pandas import Series;  exps = range(1,10)
        >>> O = Series([o.update(T=t).calc_px(method='BS',S2=s2, rho=.4, nsteps = 100, npaths=100).px_spec.px \
        for t in exps], exps)
        >>> O.plot(grid=1, title='Price vs Time to Expiry') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> # import matplotlib.pyplot as plt
        >>> # plt.show() # run last two lines to show plot

        >>> s1 = Stock(S0=30.,q=0.,vol=.2)
        >>> s2 = Stock(S0=31.,q=0.,vol=.3)
        >>> o = Spread(ref = s1, rf_r = .05, right='call', K=0., T=2., seed0 = 0)
        >>> o.calc_px(method='MC',S2 = s2,rho=.4,nsteps=1000,npaths=1000).px_spec.px
        5.644473359622987

        >>> s1 = Stock(S0=30.,q=0.,vol=.2)
        >>> s2 = Stock(S0=31.,q=0.,vol=.3)
        >>> o = Spread(ref = s1, rf_r = .05, right='put', K=2., T=2., seed0 = 0)
        >>> o.calc_px(method='MC',S2 = s2,rho=.4,nsteps=1000,npaths=1000).px_spec.px
        5.262902444782136

        >>> s1 = Stock(S0=30.,q=0.,vol=.2)
        >>> s2 = Stock(S0=30.,q=0.,vol=.2)
        >>> o = Spread(ref = s1, rf_r = .05, right='put', K=1., T=2., seed0 = 2, \
        desc = 'Perfectly correlated -- present value of 1')
        >>> o.calc_px(method='MC',S2 = s2,rho=1.,nsteps=1000,npaths=1000).px_spec
        PriceSpec
        keep_hist: false
        method: MC
        npaths: 1000
        nsteps: 1000
        px: 0.904837418



        >>> s1 = Stock(S0=30.,q=0.,vol=.2)
        >>> s2 = Stock(S0=31.,q=0.,vol=.3)
        >>> o = Spread(ref = s1, rf_r = .05, right='put', K=2., T=2., seed0 = 0)
        >>> from pandas import Series;  exps = range(1,10)
        >>> O = Series([o.update(T=t).calc_px(method='MC',S2=s2, rho=.4, nsteps = 100, npaths=100).px_spec.px \
        for t in exps], exps)
        >>> O.plot(grid=1, title='Price vs Time to Expiry') # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>

        :Authors:
            Scott Morgan
       """

        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        self.rho = rho
        self.S2 = S2
        return getattr(self, '_calc_' + method.upper())()

    def _calc_LT(self):
        """ Internal function for option valuation.


        """

        return self


    def _calc_BS(self):
        """ Internal function for option valuation using the Black-Scholes Method

        _calc_BS uses a Black-Scholes based analytical solution, but it is not exact and is only
        valid when K = 0. Thus, it does not even factor in K at all and should only be used to price
        spreads with K = 0.

        Notes
        -----

        HUGE NOTE: Black-Scholes Method only works when K = 0

        :Authors:
            Scott Morgan

        """


        vol = math.sqrt(self.ref.vol**2 - 2*self.rho*self.ref.vol*self.S2.vol + self.S2.vol**2)
        d1 = (1./(vol*math.sqrt(self.T)))*math.log((self.S2.S0*math.exp(-self.S2.q*self.T))/(self.ref.S0*math.exp(-self.ref.q*self.T)))
        d2 = d1 - (vol*math.sqrt(self.T)/2.)
        d1 = d1 + (vol*math.sqrt(self.T)/2.)
        p = self.S2.S0*math.exp(-self.S2.q*self.T)*stats.norm.cdf(d1)
        p = p - self.ref.S0*math.exp(-self.ref.q*self.T)*stats.norm.cdf(d2)

        self.px_spec.add(px=float(p), method='BS')

        return self


    def _calc_MC(self):
        """ Internal function for option valuation using Monte-Carlo simulation


        _calc_MC uses Monte-Carlo simulation to price European Spread Options
        It computes correlated paths and computes the average present value of
        the spread at expiry

        Returns
        -------
        self: Spread

        .. sectionauthor:: Scott Morgan

        Note
        ----

        """

        _ = self.px_spec
        npaths = getattr(_, 'npaths', 3)
        nsteps = getattr(_, 'nsteps', 3)

        __ = self.LT_specs(npaths)

        opt_vals = list()

        if self.seed0 is not None:
            numpy.random.seed(self.seed0)


        for path in range(0,npaths):

            ## Generate correlated Wiener Processes
            u = numpy.random.normal(size=nsteps)
            v = numpy.random.normal(size=nsteps)
            v = self.rho*u + math.sqrt(1-self.rho**2)*v
            u = u*math.sqrt(__['dt'])
            v = v*math.sqrt(__['dt'])

            ## Simulate the paths
            S1 = [self.ref.S0]
            S2 = [self.S2.S0]
            mu_1 = (self.rf_r-self.ref.q)*__['dt']
            mu_2 = (self.rf_r-self.S2.q)*__['dt']

            for t in range(0,len(u)):
                S1.append(S1[-1]*(mu_1 + self.ref.vol*u[t]) + S1[-1])
                S2.append(S2[-1]*(mu_2 + self.S2.vol*v[t]) + S2[-1])

            ## Calculate the Payoff
            val = np.maximum(self.signCP*(S2[-1] - S1[-1] - self.K),0.0)*math.exp(-self.rf_r*self.T)
            opt_vals.append((val))

        self.px_spec.add(px=float(np.mean(opt_vals)), method='MC')

        return self

    def _calc_FD(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Spread

        .. sectionauthor::

        Note
        ----

        """

        return self




