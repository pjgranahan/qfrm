from OptionValuation import *

class Spread(OptionValuation):
    """ Asian option class.

    Inherits all methods and properties of OptionValuation class.
    """


    def calc_px(self, method='BS', S2 = None, rho = .5, nsteps=None, npaths=None, keep_hist=False):
        """ Wrapper function that calls appropriate valuation method.

        User passes parameters to calc_px, which saves them to local PriceSpec object
        and calls specific pricing function (_calc_BS,...).
        This makes significantly less docstrings to write, since user is not interfacing pricing functions,
        but a wrapper function calc_px().

        Parameters
        ----------
        method : str
                Required. Indicates a valuation method to be used: 'BS', 'LT', 'MC', 'FD'
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

        .. sectionauthor:: Scott Morgan

        Notes
        -----


        Examples
        -------




       """

        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        self.rho = rho
        self.S2 = S2
        return getattr(self, '_calc_' + method.upper())()

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Spread

        .. sectionauthor::

        Note
        ----

        Formulae:


        """

        return self


    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Spread

        .. sectionauthor:: Scott Morgan

        Note
        ----

        Formulae:

        """

        return self


    def _calc_MC(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Spread

        .. sectionauthor:: Scott Morgan

        Note
        ----

        """

        from numpy.random import normal
        from numpy import maximum, mean
        from math import sqrt, exp

        _ = self.px_spec
        npaths = getattr(_, 'npaths', 3)
        nsteps = getattr(_, 'nsteps', 3)

        __ = self.LT_specs(npaths)

        opt_vals = list()

        for path in range(0,npaths):

            ## Generate correlated Wiener Processes
            u = normal(size=nsteps)
            v = normal(size=nsteps)
            v = self.rho*u + sqrt(1-self.rho**2)*v
            u = u*sqrt(__['dt'])
            v = v*sqrt(__['dt'])

            ## Simulate the paths
            S1 = [self.ref.S0]
            S2 = [self.S2.S0]
            mu_1 = (self.rf_r-self.ref.q)*__['dt']
            mu_2 = (self.rf_r-self.S2.q)*__['dt']

            for t in range(0,len(u)):
                S1.append(S1[-1]*(mu_1 + self.ref.vol*u[t]) + S1[-1])
                S2.append(S2[-1]*(mu_2 + self.S2.vol*v[t]) + S2[-1])

            ## Calculate the Payoff
            val = maximum(self.signCP*(S2[-1] - S1[-1] - self.K),0.0)*exp(-self.rf_r*self.T)
            opt_vals.append((val))

        self.px_spec.add(px=float(mean(opt_vals)), method='MC')

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



