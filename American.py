class American(OptionValuation):
    def _pxLT(self, nsteps, return_tree=False):
        """  Computes option price via binomial (lattice) tree.

        This method is not called directly. Instead, OptionValuation calls it via (vectorized) method pxLT()

        :param nsteps: number of time steps for which to build a tree
        :type nsteps:  int
        :param return_tree: indicates whether to return the full tree with stock and option prices.
        :type return_tree: bool
        :return:  option price, if return_tree is False, or a full tree, if return_tree is True.
        :rtype:  float | tuple of tuples
        """
        from numpy import arange, maximum,cumsum, log,  insert, exp, sqrt, sum

        _ = self.LT_params(nsteps)
        S = self.ref.S0 * _['d'] ** arange(nsteps, -1, -1) * _['u'] ** arange(0, nsteps + 1) # terminal stock prices
        O = maximum(self.signCP * (S - self.K), 0)          # terminal option payouts
        tree = ((S, O),)

        for i in range(nsteps, 0, -1):
            O = _['df_dt'] * ((1 - _['p']) * O[:i] + ( _['p']) * O[1:])  #prior option prices (@time step=i-1)
            S = _['d'] * S[1:i+1]                   # prior stock prices (@time step=i-1)
            Payout = maximum(self.signCP * (S - self.K), 0)   # payout at time step i-1 (moving backward in time)
            O = maximum(O, Payout)
            tree = tree + ((S, O),)

        return Util.round(tree, to_tuple=True) if return_tree else Util.demote(O)

    @property
    def pxBS(self):
        """ Currently not implemented.

        There is a way to approximate American option's price via BSM. We'll cover it in later chapters.

        :return: price for an American option estimated with BSM and other parameters.
        :rtype: None
        """
        pass
