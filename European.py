from qfrm import *

class European(OptionValuation):
    """ European option class.
    Inherits all methods and properties of OptionValuation class.
    """
    def calc_BS(self):
        """ Option valuation via BSM.

        Use BS_params method to draw computed parameters.
        They are also used by other exotic options.

        :return: self
        :rtype: European

        :Example:

        >>> s = Stock(S0=42, vol=.20)
        >>> o = European(ref=s, right='put', K=40, T=.5, rf_r=.1, desc='call @0.81, put @4.76, Hull p.339')
        >>> o.calc_BS()      # saves interim results to self and prints out BS price. Equivalent to repr(o)
        >>> (o.px.px, o.px.d1, o.px.d2, o.px.method)  # alternative way to retrieve attributes
        >>> o.update(right='call').calc_BS()  # change option object to a put
        >>> print(European(clone=o, K=41, desc='Ex. copy params to new option, but with a new strike.').calc_BS())

        """
        from scipy.stats import norm
        from math import sqrt, exp, log

        _ = self
        d1 = (log(_.ref.S0 / _.K) + (_.rf_r + _.ref.vol ** 2 / 2.) * _.T)/(_.ref.vol * sqrt(_.T))
        d2 = d1 - _.ref.vol * sqrt(_.T)

        # if calc of both prices is cheap, do both and include them into Price object.
        # Price.px should always point to the price of interest to the user
        # Save values as basic data types (int, floats, str), instead of numpy.array
        px_call = float(_.ref.S0 * exp(-_.ref.q * _.T) * norm.cdf(d1) - _.K * exp(-_.rf_r * _.T) * norm.cdf(d2))
        px_put = float(- _.ref.S0 * exp(-_.ref.q * _.T) * norm.cdf(-d1) + _.K * exp(-_.rf_r * _.T) * norm.cdf(-d2))
        px = px_call if _.signCP == 1 else px_put if _.signCP == -1 else None

        self.px = Price(px=px, px_call=px_call, px_put=px_put, d1=d1, d2=d2, method='BS', sub_method='standard; Hull p.335')
        return self

    def calc_LT(self, nsteps=3, save_tree=False):
        """ Option valuation via binomial (lattice) tree

        This method is not called directly. Instead, OptionValuation calls it via (vectorized) method pxLT()
        See Ch. 13 for numerous examples and theory.

        .. sectionauthor:: Oleg Melnikov

        :param nsteps: number of time steps in the tree
        :type nsteps: int
        :param return_tree: indicates whether a full tree needs to be returned
        :type return_tree: bool
        :return: self
        :rtype:  European

        .. seealso::

        Implementing Binomial Trees:   http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1341181

        :Example:

        >>> s = Stock(S0=810, vol=.2, q=.02)
        >>> o = European(ref=s, right='call', K=800, T=.5, rf_r=.05, desc='53.39, Hull p.291')
        >>> o.calc_LT(3).px.px  # option price from a 3-step tree (that's 2 time intervals)
        59.867529937506426
        >>> o.calc_LT(2, True).px.opt_tree
        (((663.17191000000003, 810.0, 989.33623), (0.0, 10.0, 189.33623)),
        ((732.91831000000002, 895.18844000000001), (5.0623199999999997, 100.66143)),
        ((810.0,), (53.39472,)))
        >>> o.calc_LT(2)
        European
        K: 800
        T: 0.5
        _right: call
        _signCP: 1
        frf_r: 0
        px: qfrm.Price
          LT_specs:
            a: 1.0075281954445339
            d: 0.9048374180359595
            df_T: 0.9753099120283326
            df_dt: 0.9875778004938814
            dt: 0.25
            p: 0.5125991278953855
            u: 1.1051709180756477
          method: LT
          px: 53.39471637496135
          sub_method: binomial tree; Hull Ch.13
        ref: qfrm.Stock
          S0: 810
          curr: null
          desc: null
          q: 0.02
          tkr: null
          vol: 0.2
        rf_r: 0.05
        seed0: null
        """
        from numpy import cumsum, log, arange, insert, exp, sqrt, sum, maximum

        _ = self.LT_specs(nsteps)
        S = self.ref.S0 * _['d'] ** arange(nsteps, -1, -1) * _['u'] ** arange(0, nsteps + 1)
        O = maximum(self.signCP * (S - self.K), 0)          # terminal option payouts
        S_tree, O_tree = None, None
        # tree = ((S, O),) if save_tree else None

        if save_tree:
            S_tree = (tuple([float(s) for s in S]),)
            O_tree = (tuple([float(o) for o in O]),)

            for i in range(nsteps, 0, -1):
                O = _['df_dt'] * ((1 - _['p']) * O[:i] + ( _['p']) * O[1:])  #prior option prices (@time step=i-1)
                S = _['d'] * S[1:i+1]                   # prior stock prices (@time step=i-1)
                # tree = tree + ((S, O),)
                S_tree = (tuple([float(s) for s in S]),) + S_tree
                O_tree = (tuple([float(o) for o in O]),) + O_tree

            out = O_tree[0][0]
            # tree = Util.round(tree, to_tuple=True)
        else:
            csl = insert(cumsum(log(arange(nsteps)+1)), 0, 0)         # logs avoid overflow & truncation
            tmp = csl[nsteps] - csl - csl[::-1] + log(_['p'])*arange(nsteps+1) + log(1-_['p'])*arange(nsteps+1)[::-1]
            out = (_['df_T'] * sum(exp(tmp) * tuple(O)))

        self.px = Price(px=float(out), method='LT', sub_method='binomial tree; Hull Ch.13',
                        LT_specs=_, ref_tree=S_tree, opt_tree=O_tree)
        return self
