from OptionValuation import *
from European import European
from American import American
import numpy as np

class Compound(OptionValuation):
    """ Asian option class.

    Inherits all methods and properties of OptionValuation class.
    """


    def calc_px(self, option, method='BS', on = 'put', nsteps=None, npaths=None, keep_hist=False):
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
        self.o = option
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

        _ = self.px_spec

        M = getattr(_, 'npaths', 3)
        N = getattr(_, 'nsteps', 3)

        signCP = 1 if self.right.lower() == 'call' else -1
        x = np.matrix(np.zeros(shape=(N+1,M+1)))

        T = self.T
        T_left = self.o.T - self.T
        orig_T = self.o.T
        vol = self.o.ref.vol
        Smax = 2*self.o.ref.S0
        r = self.o.rf_r
        q = .0
        dt = T/(N+0.0)
        dS = Smax/(M+0.0)
        N = int(T/dt)
        M = int(Smax/dS)
        K = self.K
        S0 = self.o.ref.S0

        def a(j):
            return .5*dt*((vol**2)*(j**2)-(r-q)*j)
        def b(j):
            return 1 - dt*((vol**2)*(j**2)+r)
        def c(j):
            return .5*dt*((vol**2)*(j**2)+(r-q)*j)

        #print(self.o)
        #print(self.o.calc_px(method='LT',nsteps=20).px_spec.px)

        self.o.T = T_left
        self.o.ref.S0 = Smax
        #print(self.o.T)
        Smax_price = self.o.calc_px(method='LT',nsteps = 30).px_spec.px
        #print(Smax)
        #print(Smax_price)
        self.o.ref.S0 = 0
        Smin_price = self.o.calc_px(method='LT',nsteps = 30).px_spec.px

        # replace with LT Price for each one
        for i in range(0,M+1):
            self.o.ref.S0 = dS*i
            x[N,i] = np.maximum(signCP*(self.o.calc_px(method='LT',nsteps = 30).px_spec.px-K),0)
            #print(x[N,i])

        #x[N,:] = np.matrix(np.maximum(signCP*(np.linspace(dS*M,0,num=M+1,endpoint=True) - K),0) ).\
            #transpose().transpose()

        # replace with LT price for each one
        if signCP == 1:
            x[:,0] = np.matrix(list(map(lambda i: (np.maximum(signCP*(Smax_price - K),0)*np.exp(-r*(N-i)*dt)), \
                                        range(0,N+1)))).transpose()
        else:
            x[:,M] = np.matrix(list(map(lambda i: (np.maximum(signCP*(Smin_price - K),0)*np.exp(-r*(N-i)*dt)), \
                                        range(0,N+1)))).transpose()

        for i in np.arange(N-1,-1,-1):
            for k in range(1,M):
                j = M-k
                x[i,k] = a(j)*x[i+1,k+1] + b(j)*x[i+1,k] + c(j)*x[i+1,k-1]

        self.o.ref.S0 = S0
        self.o.T = orig_T
        print(x[0,M-self.o.ref.S0/dS])
        return self

s = Stock(S0=55., vol=.40, q=.00)
o = European(ref=s, right='put', K=50., T=12./12., rf_r=.1, desc='7.42840, Hull p.288')
o2 = Compound(right='call',T=5./12., K = 5.)
o2.calc_px(method='FD',option=o,npaths=100,nsteps = 700)