from qfrm import *

class Bermudan(OptionValuation):
    """ Bermudan option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, method='LT', tex=(.12,.24,.46,.9,.91,.92,.93,.94,.95,.96,.97,.98,.99, 1.), nsteps=None, npaths=None, keep_hist=False):
        """ Wrapper function that calls appropriate valuation method.

        User passes parameters to calc_px, which saves them to local PriceSpec object
        and calls specific pricing function (_calc_BS,...).
        This makes significantly less docstrings to write, since user is not interfacing pricing functions,
        but a wrapper function calc_px().

        Parameters
        ----------
        method : str
                Required. Indicates a valuation method to be used: 'BS', 'LT', 'MC', 'FD'
        tex : float
                Required. Must be a vector (tuple; list; array, ...) of times to exercisability. 
                For Bermudan, assume that exercisability is for discrete tex times only.
                This also needs to be sorted ascending and the final value is the corresponding vanilla maturity.
                If T is not equal the the final value of T, then
                    the T will take precedence: if T < max(tex) then tex will be truncated to tex[tex < T] and will be appended to tex.
                    If T > max(tex) then the largest value of tex will be replaced with T.
        nsteps : int
                MC, FD methods require number of times steps. 
                Optional if using LT: n_steps = <integer> * <length of tex>. Will fill in the spaces between steps implied by tex. 
                Useful if tex is regular or sparse to improve accuracy. Otherwise leave as None.
        npaths : int
                MC, FD methods require number of simulation paths
        keep_hist : bool
                If True, historical information (trees, simulations, grid) are saved in self.px_spec object.

        Returns
        -------
        self : Bermudan

        .. sectionauthor:: Oleg Melkinov; Andy Liao


        Notes
        -----
        The Bermudan option is a modified American with restricted early-exercise dates. Due to this restriction, 
        Bermudans are named as such as they are "between" American and European options in exercisability, and as this module demonstrates,
        in price.
        

        Examples
        -------

        >>> #LT pricing of Bermudan options
        >>> s = Stock(S0=50, vol=.3)
        >>> o = Bermudan(ref=s, right='put', K=52, T=2, rf_r=.05)
        >>> o.calc_px(method='LT', keep_hist=True).px_spec.px
        7.251410363950508
        >>> ##Changing the maturity
        >>> o = Bermudan(ref=s, right='put', K=52, T=1, rf_r=.05)
        >>> o.calc_px(method='LT', keep_hist=True).px_spec.px
        5.9168269657242
        >>> o = Bermudan(ref=s, right='put', K=52, T=.5, rf_r=.05)
        >>> o.calc_px(method='LT', keep_hist=True).px_spec.px        
        4.705110748543638
        >>> ##Explicit input of exercise schedule
        >>> ##Explicit input of exercise schedule
        >>> from numpy.random import normal, seed
        >>> seed(12345678)
        >>> rlist = normal(1,1,20)
        >>> times = tuple(map(lambda i: float(str(round(abs(rlist[i]),2))), range(20)))
        >>> o = Bermudan(ref=s, right='put', K=52, T=1., rf_r=.05)
        >>> o.calc_px(method='LT', tex=times, keep_hist=True).px_spec.px
        5.8246496768398055
        >>> ##Example from p. 9 of http://janroman.dhis.org/stud/I2012/Bermuda/reportFinal.pdf
        >>> times = (3/12,6/12,9/12,12/12,15/12,18/12,21/12,24/12)
        >>> o = Bermudan(ref=Stock(50, vol=.6), right='put', K=52, T=2, rf_r=0.1)
        >>> o.calc_px(method='LT', tex=times, nsteps=40, keep_hist=False).px_spec.px       
        13.206509995991107
        >>> ##Price vs. strike curve - example of vectorization of price calculation
        >>> import matplotlib.pyplot as plt
        >>> from numpy import linspace
        >>> Karr = linspace(30,70,101)
        >>> px = tuple(map(lambda i:  Bermudan(ref=Stock(50, vol=.6), right='put', K=Karr[i], T=2, rf_r=0.1).calc_px(tex=times, nsteps=20).px_spec.px, range(Karr.shape[0])))
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> ax.plot(Karr,px,label='Bermudan put')
        >>> ax.set_title('Price of Bermudan put vs K')
        >>> ax.set_ylabel('Px')
        >>> ax.set_xlabel('K')
        >>> ax.grid()
        >>> ax.legend()
        >>> plt.show()        
        
        """

        from numpy import asarray
        T = max(tex)
        if self.T < T:
            tex = tuple(asarray(tex)[asarray(tex) < self.T]) + (self.T,)
        if self.T > T:
            tex = tex[:-1] + (self.T,)            
        self.tex = tex
        knsteps = max(tuple(map(lambda i: int(T/(tex[i+1]-tex[i])), range(len(tex)-1))))
        if nsteps != None:
            knsteps = knsteps * nsteps 
        nsteps = knsteps
        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Bermudan

        .. sectionauthor:: Oleg Melnikov; Andy Liao

        """
        from numpy import arange, maximum, log, exp, sqrt

        keep_hist = getattr(self.px_spec, 'keep_hist', False)
        n = getattr(self.px_spec, 'nsteps', 3)
        _ = self.LT_specs(n)
        
        #Redo tree steps 

        S = self.ref.S0 * _['d'] ** arange(n, -1, -1) * _['u'] ** arange(0, n + 1)  # terminal stock prices
        O = maximum(self.signCP * (S - self.K), 0)          # terminal option payouts
        # tree = ((S, O),)
        S_tree = (tuple([float(s) for s in S]),)  # use tuples of floats (instead of numpy.float)
        O_tree = (tuple([float(o) for o in O]),)
        # tree = ([float(s) for s in S], [float(o) for o in O],)

        for i in range(n, 0, -1):
            O = _['df_dt'] * ((1 - _['p']) * O[:i] + ( _['p']) * O[1:])  #prior option prices (@time step=i-1)
            S = _['d'] * S[1:i+1]                   # prior stock prices (@time step=i-1)
            Payout = maximum(self.signCP * (S - self.K), 0)   # payout at time step i-1 (moving backward in time)
            if i*_['dt'] in self.tex:   #The Bermudan condition: exercise only at scheduled times         
                O = maximum(O, Payout)
            # tree = tree + ((S, O),)
            S_tree = (tuple([float(s) for s in S]),) + S_tree
            O_tree = (tuple([float(o) for o in O]),) + O_tree
            # tree = tree + ([float(s) for s in S], [float(o) for o in O],)

        self.px_spec.add(px=float(Util.demote(O)), method='LT', sub_method='binomial tree; Hull Ch.13',
                        LT_specs=_, ref_tree = S_tree if keep_hist else None, opt_tree = O_tree if keep_hist else None)

        # self.px_spec = PriceSpec(px=float(Util.demote(O)), method='LT', sub_method='binomial tree; Hull Ch.13',
        #                 LT_specs=_, ref_tree = S_tree if save_tree else None, opt_tree = O_tree if save_tree else None)
        return self

    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Bermudan

        .. sectionauthor:: 

        Note
        ----

        """
        return self

    def _calc_MC(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Bermudan

        .. sectionauthor:: 

        Note
        ----

        """
        return self

    def _calc_FD(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Bermudan

        .. sectionauthor:: 

        Note
        ----

        """

        return self


