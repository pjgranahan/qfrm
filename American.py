from qfrm import *

class American(OptionValuation):
    def calc_LT(self, nsteps, save_tree=False):
        """  Computes option price via binomial (lattice) tree.

        This method is not called directly. Instead, OptionValuation calls it via (vectorized) method pxLT()

        :param nsteps: number of time steps for which to build a tree
        :type nsteps:  int
        :param save_tree: indicates whether to return the full tree with stock and option prices.
        :type save_tree: bool
        :return:  option price, if return_tree is False, or a full tree, if return_tree is True.
        :rtype:  float | tuple of tuples

        :Example:
        >>> s = Stock(S0=50, vol=.3)
        >>> o = American(ref=s, right='put', K=52, T=2, rf_r=.05, desc='7.42840, Hull p.288')
        >>> o.calc_LT(2, False).px_spec.px
        7.42840190270483
        >>> o.calc_LT(2, True).px_spec.ref_tree
        >>> o

        """
        from numpy import arange, maximum, log, exp, sqrt

        _ = self.LT_specs(nsteps)
        S = self.ref.S0 * _['d'] ** arange(nsteps, -1, -1) * _['u'] ** arange(0, nsteps + 1)  # terminal stock prices
        O = maximum(self.signCP * (S - self.K), 0)          # terminal option payouts
        # tree = ((S, O),)
        S_tree = (tuple([float(s) for s in S]),)  # use tuples of floats (instead of numpy.float)
        O_tree = (tuple([float(o) for o in O]),)
        # tree = ([float(s) for s in S], [float(o) for o in O],)

        for i in range(nsteps, 0, -1):
            O = _['df_dt'] * ((1 - _['p']) * O[:i] + ( _['p']) * O[1:])  #prior option prices (@time step=i-1)
            S = _['d'] * S[1:i+1]                   # prior stock prices (@time step=i-1)
            Payout = maximum(self.signCP * (S - self.K), 0)   # payout at time step i-1 (moving backward in time)
            O = maximum(O, Payout)
            # tree = tree + ((S, O),)
            S_tree = (tuple([float(s) for s in S]),) + S_tree
            O_tree = (tuple([float(o) for o in O]),) + O_tree
            # tree = tree + ([float(s) for s in S], [float(o) for o in O],)

        # self.px = Price(px=float(Util.demote(O)), method='LT', sub_method='binomial tree; Hull Ch.13', LT_specs=_, tree=Util.to_tuple(tree, leaf_as_float=True) if save_tree else None)
        self.px_spec = PriceSpec(px=float(Util.demote(O)), method='LT', sub_method='binomial tree; Hull Ch.13',
                        LT_specs=_, ref_tree = S_tree if save_tree else None, opt_tree = O_tree if save_tree else None)
        return self

    def calc_BS(self):
        """ Currently not implemented.

        There is a way to approximate American option's price via BSM. We'll cover it in later chapters.

        :return: price for an American option estimated with BSM and other parameters.
        :rtype: None
        """
        # self.px_spec = PriceSpec(px=None, desc='Not yet implemented. TODO');     return self
        return self

    def calc_MC(self):
        # self.px_spec = PriceSpec(px=None, desc='Not yet implemented. TODO');     return self
        return self

    def calc_FD(self):
        # self.px_spec = PriceSpec(px=None, desc='Not yet implemented. TODO');     return self
        return self