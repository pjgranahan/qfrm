from OptionValuation import *

class Spread(OptionValuation):
    """ Asian option class.

    Inherits all methods and properties of OptionValuation class.
    """


    def calc_px(self, method='BS', on = 'put', nsteps=None, npaths=None, keep_hist=False):
        """ Wrapper function that calls appropriate valuation method.

        User passes parameters to calc_px, which saves them to local PriceSpec object
        and calls specific pricing function (_calc_BS,...).
        This makes significantly less docstrings to write, since user is not interfacing pricing functions,
        but a wrapper function calc_px().

        Parameters
        ----------
        method : str
                Required. Indicates a valuation method to be used: 'BS', 'LT', 'MC', 'FD'
        on : str
                Either 'call' or 'put'. i.e. 'call' is for call on call or put on call (depends on right)
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
        self.on = on
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
        self: Compound

        .. sectionauthor::

        Note
        ----

        Formulae:

        """


        return self


    def _calc_MC(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Compound

        .. sectionauthor::

        Note
        ----

        """

        return self

    def _calc_FD(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Compound

        .. sectionauthor:: Scott Morgan

        Note
        ----

        """
        from numpy import arange, zeros
        from pandas import DataFrame

        _ = self.px_spec

        npaths = getattr(_, 'npaths', 3)
        nsteps = getattr(_, 'nsteps', 3)
        grid = DataFrame(zeros([npaths,nsteps]))
        





        return self
