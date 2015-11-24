from OptionValuation import *

class Basket(OptionValuation):
    """ European option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, method='BS', mu = (0.1,0.2,0.5), weight = (0.5,0.3,0.2),
                corr = [[1,0,0],[0,1,0],[0,0,1]], nsteps=None, npaths=None, keep_hist=False):
        """ Wrapper function that calls appropriate valuation method.

        User passes parameters to calc_px, which saves them to local PriceSpec object
        and calls specific pricing function (_calc_BS,...).
        This makes significantly less docstrings to write, since user is not interfacing pricing functions,
        but a wrapper function calc_px().

        Parameters
        ----------
        method : str
                Required. Indicates a valuation method to be used: 'BS', 'LT', 'MC', 'FD'
        nsteps : int
                LT, MC, FD methods require number of times steps
        npaths : int
                MC, FD methods require number of simulation paths
        keep_hist : bool
                If True, historical information (trees, simulations, grid) are saved in self.px_spec object.

        Returns
        -------
        self : Basket

        .. sectionauthor:: Hanting Li

        Notes
        -----

        Examples
        -------

        >>> s = Stock(S0=(42,55,75), vol=(.20,.30,.50))
        >>> o = Basket(ref=s, right='call', K=40, T=.5, rf_r=.1, desc='Hull p.612')

        >>> o.calc_px(method='MC',mu=(.1,.2,.5),weight=(0.3,0.5,0.2),corr=[[1,0,0],[0,1,0],[0,0,1]],npaths=10000,nsteps=100).px_spec   # save interim results to self.px_spec. Equivalent to repr(o)
        PriceSpec
        keep_hist: false
        method: MC
        npaths: 10000
        nsteps: 100
        px: 19.29183765961456
        sub_method: standard; Hull p.612

        >>> (o.px_spec.px, o.px_spec.d1, o.px_spec.d2, o.px_spec.method)  # alternative attribute access
        (0.8085993729000922, 0.7692626281060315, 0.627841271868722, 'BS')

        >>> o.update(right='call').calc_px().px_spec.px  # change option object to a put
        4.759422392871532

        >>> European(clone=o, K=41, desc='Ex. copy params; new strike.').calc_px(method='LT').px_spec.px
        4.2270039114413125

        >>> s = Stock(S0=810, vol=.2, q=.02)
        >>> o = European(ref=s, right='call', K=800, T=.5, rf_r=.05, desc='53.39, Hull p.291')
        >>> o.calc_px(method='LT', nsteps=3, keep_hist=True).px_spec.px  # option price from a 3-step tree (that's 2 time intervals)
        59.867529937506426

        >>> o.px_spec.ref_tree  # prints reference tree
        ((810.0,),
         (746.4917680871579, 878.9112325795882),
         (687.9629133603595, 810.0, 953.6851293266307),
         (634.0230266330457, 746.491768087158, 878.9112325795882, 1034.8204598880159))

        >>> o.calc_px(method='LT', nsteps=2, keep_hist=True).px_spec.opt_tree
        ((53.39471637496134,),
         (5.062315192620067, 100.66143225703827),
         (0.0, 10.0, 189.3362341097378))

        >>> o.calc_px(method='LT', nsteps=2)
        European
        K: 800
        T: 0.5
        _right: call
        _signCP: 1
        desc: 53.39, Hull p.291
        frf_r: 0
        px_spec: qfrm.PriceSpec
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
        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        self.mu = mu
        self.weight = weight
        self.corr = corr
        self.npaths = npaths
        self.nsteps = nsteps
        return getattr(self, '_calc_' + method.upper())()

    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: European

        .. sectionauthor:: Oleg Melnikov

        """

        return self

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: European

        .. sectionauthor:: Oleg Melnikov

        .. note::
        Implementing Binomial Trees:   http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1341181

        """

        return self

    def _calc_MC(self, keep_hist=False):
        """ Internal function for option valuation.

        Returns
        -------
        self: European

        .. sectionauthor:: Hanting Li

        Notes
        -----
        Implementing Binomial Trees:   http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1341181

        """

        from numpy.random import multivariate_normal, seed
        from numpy import sqrt, mean, matrix, transpose, diag, dot, repeat, exp

        _ = self

        S0 = _.ref.S0
        vol = _.ref.vol
        mu = _.mu
        corrM = _.corr
        nsteps = _.nsteps
        npaths = _.npaths

        deltat = _.T/nsteps
        Nasset = len(vol)

        def calS(St,mu,sigma,param):
            deltaS = mu*St*deltat + sigma*St*param*sqrt(deltat)
            S_update = St+deltaS
            return(S_update.item())

        def one_path(S0,mu,vol,param):
            S0 = (S0,)
            for i in range(nsteps):
                parami = param[i]
                S0 = S0 + (calS(S0[len(S0)-1],mu,vol,parami),)
            return(S0)

        priceNpath = ()


        covM = dot(dot(diag(vol),(corrM)),diag(vol))

        seed(111)
        param = multivariate_normal(repeat(0,Nasset),covM,nsteps)
        param = tuple(zip(*param))

        for i in range(npaths):
            price = list(map(one_path,S0,mu,vol,param))
            wprice = transpose(matrix(price))*transpose(matrix(_.weight))
            wprice = tuple(wprice.ravel().tolist()[0])
            priceNpath = priceNpath + (wprice,)

        payoff = max(0,_.signCP*(mean(tuple(zip(*priceNpath))[nsteps])-_.K))


        self.px_spec.add(px=float(payoff*exp(-_.rf_r*_.T)), sub_method='standard; Hull p.612')

        return self


    def _calc_FD(self, nsteps=3, npaths=4, keep_hist=False):
        """ Internal function for option valuation.

        Returns
        -------
        self: European

        .. sectionauthor::

        """
        return self
