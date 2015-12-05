import math

import numpy as np
import scipy.linalg as la

try: from qfrm.OptionValuation import *  # production:  if qfrm package is installed
except:   from qfrm.option import *  # development: if not installed and running from source


class Chooser(OptionValuation):
    """ Chooser option class.

    Inherits all methods and properties of OptionValuation class.
    An option contract that allows the holder to decide whether it is a call or put prior to
    the expiration date. Chooser options usually have the same exercise price and expiration
    date regardless of what decision the holder ultimately makes.
    """

    def calc_px(self, method='BS', nsteps=None, npaths=None, keep_hist=False, tau=None):
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
        nsteps : int
                LT, MC, FD methods require number of times steps
        npaths : int
                MC, FD methods require number of simulation paths
        keep_hist : bool
                If True, historical information (trees, simulations, grid) are saved in self.px_spec object.
        tau : float
                Time to choose whether this option is a call or put.

        Returns
        -------
        self : Chooser
            Returned object contains specifications and calculated price in embedded ``PriceSpec`` object.


        Notes
        ------


        **FD Notes**

        Mathworks Chooser BSM result gives a price of 8.9308 for the first FD example, below.
        See: `MathWorks chooserbybls() documentation
        <http://www.mathworks.com/help/fininst/chooserbybls.html>`_. The difference between the
        BSM result and the FD result is a discretization error.

        Examples
        --------

        **BS Examples**

        EXOTIC OPTIONS: A CHOOSER OPTION AND ITS PRICING by Raimonda Martinkkute-Kauliene (Dec 2012)
        https://www.dropbox.com/s/r9lvi0uzdehwlm4/101-330-1-PB%20%284%29.pdf?dl=0

        >>> s = Stock(S0=50, vol=0.2, q=0.05)
        >>> o = Chooser(ref=s, right='put', K=50, T=1, rf_r=.1, desc= 'Exotic options paper page 297 Table 2 time 0.5')
        >>> o.pxBS(tau=6/12)
        6.587896324

        >>> s = Stock(S0=50, vol=0.2, q=0.05)
        >>> o = Chooser(ref=s, right='put', K=50, T=1, rf_r=.1, desc= 'Exotic options paper page 297 Table 2 time 1.00')
        >>> o.pxBS(tau=12/12)
        7.621302274

        >>> s = Stock(S0=50, vol=0.25, q=0.08)
        >>> o = Chooser(ref=s, right='put', K=50, T=.5, rf_r=.08)
        >>> o.pxBS(tau=3/12)
        5.777578344

        **LT Examples**

        >>> s = Stock(S0=50, vol=0.2, q=0.05)
        >>> o = Chooser(ref=s, right='put', K=50, T=1, rf_r=.1, desc= 'Exotic options paper page 297 Table 2 time 0.5')
        >>> o.pxLT(tau=3/12, nsteps=2)
        6.755605275

        >>> o.calc_px(tau=3/12, method='LT', nsteps=2, keep_hist=True).px_spec.ref_tree
        ((50.0,), (43.40617226972924, 57.595495508445445), (37.68191582218824, 49.99999999999999, 66.3448220572672))

        >>> o.px_spec
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        PriceSpec...px: 6.755605275...

        >>> from pandas import Series
        >>> expiries = range(1,11)
        >>> o = Series([o.update(T=t).pxLT(tau=3/12, nsteps=2) for t in expiries], expiries)
        >>> o.plot(grid=1, title='LT Price vs expiry (in years)')# doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at ...>
        >>> import matplotlib.pyplot as plt
        >>> plt.show()
        
        **FD Examples**

        First example: see referenced result for comparison

        >>> s = Stock(S0=50, vol=0.2, q=0.05)
        >>> o = Chooser(ref=s, right='put', K=60, T=6/12, rf_r=.1, desc= 'Mathworks example')
        >>> o.pxFD(tau=3/12,nsteps=100,npaths=100) # doctest: +ELLIPSIS
        8.94395152...
        
        Second example: coarsen the grid to increase deviation from the BSM price

        >>> s = Stock(S0=50, vol=0.2, q=0.05)
        >>> o = Chooser(ref=s, right='put', K=60, T=6/12, rf_r=.1, desc= 'Mathworks example')
        >>> o.pxFD(tau=3/12,nsteps=10,npaths=10) # doctest: +ELLIPSIS
        9.49812976...
        
        Third example: Change the maturity

        >>> s = Stock(S0=50, vol=0.2, q=0.05)
        >>> o = Chooser(ref=s, right='put', K=60, T=12/12, rf_r=.1, desc= 'Mathworks example')
        >>> o.pxFD(tau=3/12,nsteps=100,npaths=100) # doctest: +ELLIPSIS
        8.49747834...
        
        Fourth example: make choice at t=0: price collapses to a European call.

        >>> s = Stock(S0=50, vol=0.2, q=0.05)
        >>> o = Chooser(ref=s, right='put', K=60, T=12/12, rf_r=.1, desc= 'Mathworks example')
        >>> o.pxFD(tau=0/12,nsteps=100,npaths=100) # doctest: +ELLIPSIS
        8.27396786...
        
        Vectorization example with plot: exploration of tau-space.

        >>> tarr = np.linspace(0,12/12,11)
        >>> px = tuple(map(lambda i: Chooser(ref=s, right='put', K=60, T=12/12, rf_r=.1).pxFD(tau=tarr[i],nsteps=100,
        ... npaths=100), range(tarr.shape[0])))
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111) 
        >>> ax.plot(tarr,px,label='Chooser') # doctest: +ELLIPSIS
        [<...>]
        >>> ax.set_title('Price of Chooser vs tau') # doctest: +ELLIPSIS
        <...>
        >>> ax.set_ylabel('Px') # doctest: +ELLIPSIS
        <...>
        >>> ax.set_xlabel('tau') # doctest: +ELLIPSIS 
        <...>
        >>> ax.grid()
        >>> ax.legend() # doctest: +ELLIPSIS
        <...>
        >>> plt.show()
        

        See Also
        ---------

        Hull, John C.,Options, Futures and Other Derivatives, 9ed, 2014. Prentice Hall. ISBN 978-0-13-345631-8.
        http://www-2.rotman.utoronto.ca/~hull/ofod/index.html

        Huang Espen G., Option Pricing Formulas, 2ed.
        http://down.cenet.org.cn/upfile/10/20083212958160.pdf

        Wee, Lim Tiong, MFE5010 Exotic Options,Notes for Lecture 4 Chooser option.
        http://www.stat.nus.edu.sg/~stalimtw/MFE5010/PDF/L4chooser.pdf

        Humphreys, Natalia A., ACTS 4302 Principles of Actuarial Models: Financial Economics.
        Lesson 14: All-or-nothing, Gap, Exchange and Chooser Options.

        Implementing Binomial Trees:   http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1341181

        :Authors:
            Thawda Aung,
            Yen-fei Chen <yensfly@gmail.com>,
            Andy Liao <Andy.Liao@rice.edu>

        """
        self.tau = float(tau)
        return super().calc_px(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)

    def _calc_BS(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Thawda Aung
        """

        _ = self
        N = Util.norm_cdf

        d2 = (math.log(_.ref.S0/_.K) + ((_.rf_r - _.ref.q  - _.ref.vol**2/2)*_.T) ) / ( _.ref.vol * math.sqrt(_.T))
        d1 =  d2 + _.ref.vol * math.sqrt(_.T)

        d2n = (math.log(_.ref.S0/_.K) + (_.rf_r - _.ref.q) * _.T - _.ref.vol**2 * _.tau /2) / ( _.ref.vol * math.sqrt(_.tau))
        d1n = d2n + _.ref.vol * math.sqrt(_.tau)

        px = _.ref.S0 * math.exp(-_.ref.q * _.T) * N(d1) - _.K* math.exp(-_.rf_r * _.T ) * N(d2) +\
             _.K* math.exp(-_.rf_r * _.T ) * N(-d2n)  - _.ref.S0* math.exp(-_.ref.q * _.T) * N(-d1n)
        self.px_spec.add(px=px, d1=d1, d2=d2)

        return self

    def _calc_LT(self):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Yen-fei Chen <yensfly@gmail.com>
        """

        n = getattr(self.px_spec, 'nsteps', 3)
        _ = self.LT_specs(n)

        S = self.ref.S0 * _['d'] ** np.arange(n, -1, -1) * _['u'] ** np.arange(0, n + 1)
        O = np.maximum(np.maximum((S - self.K), 0), np.maximum(-1*(S - self.K), 0))
        S_tree, O_tree = None, None

        if getattr(self.px_spec, 'keep_hist', False):
            S_tree = (tuple([float(s) for s in S]),)
            O_tree = (tuple([float(o) for o in O]),)

            for i in range(n, 0, -1):
                O = _['df_dt'] * ((1 - _['p']) * O[:i] + (_['p']) * O[1:])  # prior option prices (@time step=i-1)
                S = _['d'] * S[1:i + 1]  # prior stock prices (@time step=i-1)

                S_tree = (tuple([float(s) for s in S]),) + S_tree
                O_tree = (tuple([float(o) for o in O]),) + O_tree

            out = O_tree[0][0]
        else:
            csl = np.insert(np.cumsum(np.log(np.arange(n) + 1)), 0, 0)  # logs avoid overflow & truncation
            tmp = csl[n] - csl - csl[::-1] + np.log(_['p']) * np.arange(n + 1) + np.log(1 - _['p']) * np.arange(n + 1)[::-1]
            out = (_['df_T'] * sum(np.exp(tmp) * tuple(O)))

        self.px_spec.add(px=float(out), sub_method='binomial tree; Hull Ch.135',
                         LT_specs=_, ref_tree=S_tree, opt_tree=O_tree)

        return self

    def _calc_MC(self, nsteps=3, npaths=4, keep_hist=False):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Andrew Weatherly <amw13@rice.edu>
        """
        n = getattr(self.px_spec, 'nsteps', 3)

        return self

    def _calc_FD(self, nsteps=3, npaths=4, keep_hist=False):
        """ Internal function for option valuation.

        See ``calc_px()`` for complete documentation.

        :Authors:
            Andy Liao <Andy.Liao@rice.edu>
        """

        #List all the parameters used in calculation
        _ = self
        
        S0 = _.ref.S0
        vol = _.ref.vol
        q = _.ref.q
        r = _.rf_r
        K = _.K
        tau = _.tau
        T = _.T
        N = _.px_spec.nsteps
        M = _.px_spec.npaths
        
        #Grid parameters
        Smax = 2*K
        dt = T/float(N)
        dS = Smax/float(M)
        ival = np.arange(M)
        jval = np.arange(N)
        
        #Set up grid
        grid = np.zeros(shape=(M+1,N+1))
        bcs = np.linspace(0,Smax,M+1)
        
        #Calculation as given in Hull 9e. section 26.8, page 604.
        #We use the Crank-Nicholson method to:        
        #Compute the price of a European call with maturity T and strike K
        #Set up BC's
        grid[:,-1] = np.maximum(bcs-K,0)
        grid[-1,:-1] = (Smax-K)*np.exp(-r*dt*(N-jval))
        #Set up coefficients
        alpha = .25*dt*(vol**2*ival**2-(r-q)*ival)
        beta = -dt*.5*(vol**2*ival**2+r)
        gamma = .25*dt*(vol**2*ival**2+(r-q)*ival)
        M1 = -np.diag(alpha[2:M],-1)+np.diag(1-beta[1:M])-np.diag(gamma[1:M-1],1)
        M2 = np.diag(alpha[2:M],-1)+np.diag(1+beta[1:M])+np.diag(gamma[1:M-1],1)
        #Populate grid
        P, L, U = la.lu(M1)
        for j in reversed(range(N)):
            x1 = la.solve(L, np.dot(M2,grid[1:M,j+1]))
            x2 = la.solve(U, x1)
            grid[1:M,j] = x2
        cpx = np.interp(S0,bcs,grid[:,0])
        
        #Compute the price of a European put with maturity tau and strike K*exp(-(r-q)*(T-tau))
        grid = np.zeros(shape=(M+1,N+1))
        Kau = K*np.exp(-r*(T-tau))
        S1 = S0*np.exp(-q*(T-tau))
        Smau = 2*Kau
        dt = tau/float(N)
        dS = Smau/float(M)
        bcs = np.linspace(0,Smau,M+1)
        grid[:,-1] = np.maximum(Kau-bcs,0)
        grid[-1,:-1] = (Kau-Smau)*np.exp(-r*dt*(N-jval))
        #Set up coefficients
        alpha = .25*dt*(vol**2*ival**2-(r-q)*ival)
        beta = -dt*.5*(vol**2*ival**2+r)
        gamma = .25*dt*(vol**2*ival**2+(r-q)*ival)
        M1 = -np.diag(alpha[2:M],-1)+np.diag(1-beta[1:M])-np.diag(gamma[1:M-1],1)
        M2 = np.diag(alpha[2:M],-1)+np.diag(1+beta[1:M])+np.diag(gamma[1:M-1],1)
        #Populate grid
        P, L, U = la.lu(M1)
        for j in reversed(range(N)):
            x1 = la.solve(L, np.dot(M2,grid[1:M,j+1]))
            x2 = la.solve(U, x1)
            grid[1:M,j] = x2
        ppx = np.interp(S1,bcs,grid[:,0])        

        #Sum of above call and put prices is the Chooser price.        
        _.px_spec.add(px=cpx+ppx)

        return self
