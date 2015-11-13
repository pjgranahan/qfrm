def Barrier_BS(S0,K,r,q,sigma,T,H,Right,knock,dir):

    from scipy.stats import norm
    from numpy import exp, log, sqrt

    # Compute Parameters
    d1 = (log(S0/K) + (r-q+(sigma**2)/2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    c = S0*exp(-q*T)*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
    p = K*exp(-r*T)*norm.cdf(-d2) - S0*exp(-q*T)*norm.cdf(-d1)

    l = (r-q+sigma**2)/(sigma**2)
    y = log((H**2)/(S0*K))/(sigma*sqrt(T)) + l*sigma*sqrt(T)
    x1 = log(S0/H)/(sigma*sqrt(T)) + l*sigma*sqrt(T)
    y1 = log(H/S0)/(sigma*sqrt(T)) + l*sigma*sqrt(T)

    # Consider Call Option
    # Two Situations: H<=K vs H>K
    if (Right == 'call'):
        if (H<=K):
            cdi = S0*exp(-q*T)*((H/S0)**(2*l))*norm.cdf(y) - K*exp(-r*T)*((H/S0)**(2*l-2))*norm.cdf(y-sigma*sqrt(T))
            cdo = S0*norm.cdf(x1)*exp(-q*T) - K*exp(-r*T)*norm.cdf(x1-sigma*sqrt(T)) \
                  - S0*exp(-q*T)*((H/S0)**(2*l))*norm.cdf(y1) + K*exp(-r*T)*((H/S0)**(2*l-2))*norm.cdf(y1-sigma*sqrt(T))
            cdo = c - cdi
            cuo = 0
            cui = c
        else:
            cdo = S0*norm.cdf(x1)*exp(-q*T) - K*exp(-r*T)*norm.cdf(x1-sigma*sqrt(T)) \
                  - S0*exp(-q*T)*((H/S0)**(2*l))*norm.cdf(y1) + K*exp(-r*T)*((H/S0)**(2*l-2))*norm.cdf(y1-sigma*sqrt(T))
            cdi = c - cdo
            cui = S0*norm.cdf(x1)*exp(-q*T) - K*exp(-r*T)*norm.cdf(x1-sigma*sqrt(T)) - \
                  S0*exp(-q*T)*((H/S0)**(2*l))*(norm.cdf(-y)-norm.cdf(-y1)) + \
                  K*exp(-r*T)*((H/S0)**(2*l-2))*(norm.cdf(-y+sigma*sqrt(T))-norm.cdf(-y1+sigma*sqrt(T)))
            cuo = c - cui
    # Consider Put Option
    # Two Situations: H<=K vs H>K
    else:
        if (H>K):
            pui = -S0*exp(-q*T)*((H/S0)**(2*l))*norm.cdf(-y) + K*exp(-r*T)*((H/S0)*(2*l-2))*norm.cdf(-y+sigma*sqrt(T))
            puo = p - pui
            pdo = 0
            pdi = p
        else:
            puo = -S0*norm.cdf(-x1)*exp(-q*T) + K*exp(-r*T)*norm.cdf(-x1+sigma*sqrt(T)) + \
                  S0*exp(-q*T)*((H/S0)**(2*l))*norm.cdf(-y1) - K*exp(-r*T)*((H/S0)**(2*l-2))*norm.cdf(-y1+sigma*sqrt(T))
            pui = p - puo
            pdi = -S0*norm.cdf(-x1)*exp(-q*T) + K*exp(-r*T)*norm.cdf(-x1+sigma*sqrt(T)) + \
                  S0*exp(-q*T)*((H/S0)**(2*l))*(norm.cdf(y)-norm.cdf(y1)) - \
                  K*exp(-r*T)*((H/S0)**(2*l-2))*(norm.cdf(y-sigma*sqrt(T)) - norm.cdf(y1-sigma*sqrt(T)))
            pdo = p - pdi

    if (Right == 'call'):
        if (knock == 'down'):
            if (dir == 'in'):
                return(cdi)
            else:
                return(cdo)
        else:
            if (dir == 'in'):
                return(cui)
            else:
                return(cdi)
    else:
        if (knock == 'down'):
            if (dir == 'in'):
                return(pdi)
            else:
                return(pdi)
        else:
            if (dir == 'in'):
                return(pui)
            else:
                return(pdi)

'''
#print(Barrier_BS(50,40,0.01,0.03,0.15,1,60,'call','down','in'))
Barrier_BS(50,40,0.01,0.03,0.15,0.5,60,'call','up','in')
Barrier_BS(60,40,0.01,0.03,0.15,0.75,30,'put','up','in')
Barrier_BS(50,30,0.01,0.03,0.15,10,60,'put','down','out')
Barrier_BS(90,100,0.05,0.45,0.15,3,72,'call','up','out')
Barrier_BS(100,80,0.1,0.15,0.2,5,60,'put','down','out')
'''


from qfrm import *

class Barrier(OptionValuation):
    """ European option class.

    Inherits all methods and properties of OptionValuation class.
    """

    def __init__(self, H = 10., knock = 'down', dir = 'out', *args, **kwargs):

        """ Constructor for Barrier class

        Passes additional arguments to OptionValuation class

        Parameters
        ----------
        H : int
                The barrier used to price the barrier option
        knock : string
                'down' or 'up'
        dir : string
                'in' or 'out'
        *args, **kwargs: varies
                arguments required by the constructor of OptionValuation class


        Returns
        -------
        self : Barrier

        .. sectionauthor:: Scott Morgan

       """

        self.H = H
        self.knock = knock
        self.dir = dir
        super().__init__(*args,**kwargs)

    def calc_px(self, method='BS', nsteps=None, npaths=None, keep_hist=False):
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
        self : Barrier

        .. sectionauthor:: Scott Morgan

       """


        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_BS(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Barrier

        .. sectionauthor::

        """

        return self

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Barrier

        .. sectionauthor:: Scott Morgan

        .. note::
        Binomial Trees for Barrier Options:   http://homepage.ntu.edu.tw/~jryanwang/course/Financial%20Computation%20or%20Financial%20Engineering%20(graduate%20level)/FE_Ch08%20Barrier%20Option.pdf
        In-Out Parity: http://www.iam.uni-bonn.de/people/ankirchner/lectures/OP_WS1314/OP_chap_nine.pdf
        Verify Examples: http://www.fintools.com/resources/online-calculators/exotics-calculators/exoticscalc-barrier/


        Examples
        -------

        >>> s = Stock(S0=95., vol=.25, q=.00)
        >>> o = Barrier(H=90.,knock='down',dir='in',ref=s, right='put', K=100., T=1., rf_r=.1, desc='down and in put')
        >>> print(o.calc_px(method='LT', nsteps=1050, keep_hist=False).px_spec.px)
        >>> print(o.px_spec)

        7.104101924957116

        qfrm.PriceSpec
        LT_specs:
          a: 1.0000952426305294
          d: 0.9923145180146982
          df_T: 0.9048374180359595
          df_dt: 0.9999047664397653
          dt: 0.0009523809523809524
          p: 0.5042435843778115
          u: 1.0077450060900832
        keep_hist: false
        method: LT
        nsteps: 1050
        px: 7.104101924957116
        sub_method: in out parity


        >>> s = Stock(S0=95., vol=.25, q=.00)
        >>> o = Barrier(H=87.,knock='down',dir='out',ref=s, right='call', K=100., T=2., rf_r=.1, desc='down and out call')
        >>> print(o.calc_px(method='LT', nsteps=1050, keep_hist=False).px_spec.px)

        11.549805549495334

        >>> s = Stock(S0=95., vol=.25, q=.00)
        >>> o = Barrier(H=105.,knock='up',dir='out',ref=s, right='put', K=100., T=2., rf_r=.1, desc='up and out put')
        >>> print(o.calc_px(method='LT', nsteps=1050, keep_hist=False).px_spec.px)

        3.2607593764427434

        >>> s = Stock(S0=95., vol=.25, q=.00)
        >>> o = Barrier(H=105.,knock='up',dir='in',ref=s, right='call', K=100., T=2., rf_r=.1, desc='up and in call')
        >>> print(o.calc_px(method='LT', nsteps=1050, keep_hist=False).px_spec.px)

        20.037733657756565

        >>> s = Stock(S0=95., vol=.25, q=.00)
        >>> o = Barrier(H=105.,knock='up',dir='in',ref=s, right='call', K=100., T=2., rf_r=.1, desc='up and in call')
        >>> print(o.calc_px(method='LT', nsteps=10, keep_hist=False).px_spec.px)

        20.040606033552542

        """

        if self.knock == 'down':
            s = 1
        elif self.knock == 'up':
            s = -1

        from numpy import arange, maximum, log, exp, sqrt, minimum

        keep_hist = getattr(self.px_spec, 'keep_hist', False)
        n = getattr(self.px_spec, 'nsteps', 3)
        _ = self.LT_specs(n)

        S = self.ref.S0 * _['d'] ** arange(n, -1, -1) * _['u'] ** arange(0, n + 1)  # terminal stock prices
        S2 = maximum(s*(S - self.H),0) # Find where crossed the barrier
        S2 = minimum(S2,1)  # 0 when across the barrier, 1 otherwise
        O = maximum(self.signCP * (S - self.K), 0)
        O = O * S2        # terminal option payouts
        # tree = ((S, O),)
        S_tree = (tuple([float(s) for s in S]),)  # use tuples of floats (instead of numpy.float)
        O_tree = (tuple([float(o) for o in O]),)
        # tree = ([float(s) for s in S], [float(o) for o in O],)

        for i in range(n, 0, -1):
            O = _['df_dt'] * ((1 - _['p']) * O[:i] + ( _['p']) * O[1:])  #prior option prices (@time step=i-1)
            S = _['d'] * S[1:i+1]                   # prior stock prices (@time step=i-1)
            S2 = maximum(s*(S - self.H),0)
            S2 = minimum(S2,1)
            O = O * S2
            S_tree = (tuple([float(s) for s in S]),) + S_tree
            O_tree = (tuple([float(o) for o in O]),) + O_tree

        out_px = float(Util.demote(O))

        # self.px_spec = PriceSpec(px=float(Util.demote(O)), method='LT', sub_method='binomial tree; Hull Ch.13',
        #                 LT_specs=_, ref_tree = S_tree if save_tree else None, opt_tree = O_tree if save_tree else None)

        if self.dir == 'out':

            self.px_spec.add(px=out_px, method='LT', sub_method='binomial tree; biased',
                        LT_specs=_, ref_tree = S_tree if keep_hist else None, opt_tree = O_tree if keep_hist else None)

            return self


        from sympy import binomial
        from math import ceil, floor

        k = int(ceil(log(self.K/(self.ref.S0*_['d']**n))/log(_['u']/_['d'])))
        h = int(floor(log(self.H/(self.ref.S0*_['d']**n))/log(_['u']/_['d'])))
        l = list(map(lambda j: binomial(n,n-2*h+j)*(_['p']**j)*((1-_['p'])**(n-j))*(self.ref.S0*(_['u']**j)*(_['d']**(n-j))-self.K),range(k,n+1)))
        down_in_call = exp(-self.rf_r*self.T)*sum(l)



        if self.dir == 'in' and self.right == 'call' and self.knock == 'down':
            self.px_spec.add(px=down_in_call, method='LT', sub_method='combinatorial',
                        LT_specs=_, ref_tree = None, opt_tree =  None)

        elif self.dir == 'in' and self.right == 'call' and self.knock == 'up':

            from European import European
            o = European(ref=self.ref, right='call', K=self.K, T=self.T, rf_r=self.rf_r, desc='reference')
            call_px = o.calc_px(method='BS').px_spec.px   # save interim results to self.px_spec. Equivalent to repr(o)
            in_px = call_px - out_px
            self.px_spec.add(px=in_px, method='LT', sub_method='in out parity',
                        LT_specs=_, ref_tree = None, opt_tree =  None)


        elif self.dir == 'in' and self.right == 'put' and self.knock == 'up':

            from European import European
            o = European(ref=self.ref, right='put', K=self.K, T=self.T, rf_r=self.rf_r, desc='reference')
            put_px = o.calc_px(method='BS').px_spec.px   # save interim results to self.px_spec. Equivalent to repr(o)
            in_px = put_px - out_px
            self.px_spec.add(px=in_px, method='LT', sub_method='in out parity',
                        LT_specs=_, ref_tree = None, opt_tree =  None)

        elif self.dir == 'in' and self.right == 'put' and self.knock == 'down':

            from European import European
            o = European(ref=self.ref, right='put', K=self.K, T=self.T, rf_r=self.rf_r, desc='reference')
            put_px = o.calc_px(method='BS').px_spec.px   # save interim results to self.px_spec. Equivalent to repr(o)
            in_px = put_px - out_px
            self.px_spec.add(px=in_px, method='LT', sub_method='in out parity',
                        LT_specs=_, ref_tree = None, opt_tree =  None)

        return self


    def _calc_MC(self, nsteps=3, npaths=4, keep_hist=False):
        """ Internal function for option valuation.

        Returns
        -------
        self: Barrier

        .. sectionauthor::

        Notes
        -----


        """
        return self

    def _calc_FD(self, nsteps=3, npaths=4, keep_hist=False):
        """ Internal function for option valuation.

        Returns
        -------
        self: Barrier

        .. sectionauthor::

        """
        return self

