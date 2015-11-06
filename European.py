from qfrm import *
class European(OptionValuation):
    """ European option class.
    Inherits all methods and properties of OptionValuation class.
    """
    @property
    def pxBS(self):
        """ Option valuation via BSM.

        Use BS_params method to draw computed parameters.
        They are also used by other exotic options.
        It's basically a one-liner.

        :return: price of a put or call European option
        :rtype: float

        :Example:


        """
        from math import exp
        _ = self.BS_params
        c = (self.ref.S0 * _['Nd1'] - self.K * exp(-self.r * self.T) * _['Nd2'])
        return c if self.right == 'call' else c - self.ref.S0 + exp(-self.r * self.T) * self.K

    def _pxLT(self, nsteps=3, return_tree=False):
        """ Option valuation via binomial (lattice) tree

        This method is not called directly. Instead, OptionValuation calls it via (vectorized) method pxLT()
        See Ch. 13 for numerous examples and theory.

        :param nsteps: number of time steps in the tree
        :type nsteps: int
        :param return_tree: indicates whether a full tree needs to be returned
        :type return_tree: bool
        :return: option price or a chronological tree of stock and option prices
        :rtype:  float|tuple of tuples

        :Example:

        >>> European().pxLT()  # produce lattice tree pricing with default parameters

        >>> a = European(ref=Stock(S0=810, vol=.2, q=.02), right='call', K=800, T=.5, r=.05)   # 53.39, p.291
        >>> a.pxLT(2)
        53.394716374961348
        >>> a.pxLT((2,20,200))
        (7.42840190270483, 7.5113077715410839, 7.4772083289361388)
        >>> a.pxLT(2, return_tree=True)
        (((27.44058, 50.0, 91.10594), (24.55942, 2.0, 0.0)),    # stock and option values for step 2
        ((37.04091, 67.49294), (14.95909, 0.9327)),             # stock and option values for step 1
        ((50.0,), (7.4284,)))                                   # stock and option values for step 0 (now)
        """
        # http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1341181
        # def pxLT_(nsteps):
        from numpy import cumsum, log, arange, insert, exp, sqrt, sum, maximum, vectorize

        _ = self.LT_params(nsteps)
        S = self.ref.S0 * _['d'] ** arange(nsteps, -1, -1) * _['u'] ** arange(0, nsteps + 1)
        O = maximum(self.signCP * (S - self.K), 0)          # terminal option payouts
        tree = ((S, O),)

        if return_tree:
            for i in range(nsteps, 0, -1):
                O = _['df_dt'] * ((1 - _['p']) * O[:i] + ( _['p']) * O[1:])  #prior option prices (@time step=i-1)
                S = _['d'] * S[1:i+1]                   # prior stock prices (@time step=i-1)
                tree = tree + ((S, O),)
            out = Util.round(tree, to_tuple=True)
        else:
            csl = insert(cumsum(log(arange(nsteps)+1)), 0, 0)         # logs avoid overflow & truncation
            tmp = csl[nsteps] - csl - csl[::-1] + log(_['p'])*arange(nsteps+1) + log(1-_['p'])*arange(nsteps+1)[::-1]
            out = (_['df_T'] * sum(exp(tmp) * tuple(O)))
        return out
