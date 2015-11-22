import numpy as np
import math
from OptionValuation import *

class Boston(OptionValuation):
    """ Boston Option Valuation Class

    Inherits all methods and properties of OptionValuation class.
    """

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
        self : Boston

        .. sectionauthor:: Andrew Weatherly

        Notes
        -----
        A Boston option is simply an American option except that the premium is paid at the end of the contract instead of
        the beginning as done in a normal American option.  Because of this, the price of the option has to be calculated
        as the American price NPV.

        See page 598 and 599 in Hull for explanation.

        Examples
        -------
        >>> s = Stock(S0=50, vol=.3)
        >>> o = Boston(ref=s, right='put', K=52, T=2, rf_r=.05)
        >>> o.calc_px(method='LT', nsteps=2, keep_hist=True).px_spec.px
        8.209653750647185

        >>> o.px_spec.ref_tree
        ((50.000000000000014,), (37.0409110340859, 67.49294037880017), (27.440581804701324, 50.00000000000001, 91.10594001952546))

        >>> o.calc_px(method='LT', nsteps=2, keep_hist=False)
        Boston
        K: 52
        T: 2
        _right: put
        _signCP: -1
        frf_r: 0
        px_spec: PriceSpec
          LT_specs:
            a: 1.0512710963760241
            d: 0.7408182206817179
            df_T: 0.9048374180359595
            df_dt: 0.951229424500714
            dt: 1.0
            p: 0.5097408651817704
            u: 1.3498588075760032
          keep_hist: false
          method: LT
          nsteps: 2
          px: 8.209653750647185
          sub_method: Binomial Tree
        ref: Stock
          S0: 50
          curr: -
          desc: -
          q: 0
          tkr: -
          vol: 0.3
        rf_r: 0.05
        seed0: -
        <BLANKLINE>
        """
        self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        return getattr(self, '_calc_' + method.upper())()

    def _calc_LT(self):
        """ Internal function for option valuation.

        Returns
        -------
        self: Boston

        .. sectionauthor:: Andrew Weatherly

        Examples
        -------
        >>> s = Stock(S0=45, vol=.3, q=.02)
        >>> o = Boston(ref=s, right='call', K=52, T=3, rf_r=.05)
        >>> o.calc_px(method='LT', nsteps=10, keep_hist=True).px_spec.px
        9.272539685915113
        >>> o.calc_px(method='LT', nsteps=10, keep_hist=False)
        Boston
        K: 52
        T: 3
        _right: call
        _signCP: 1
        frf_r: 0
        px_spec: PriceSpec
          LT_specs:
            a: 1.0090406217738679
            d: 0.8484732107801852
            df_T: 0.8607079764250578
            df_dt: 0.9851119396030626
            dt: 0.3
            p: 0.4863993185207596
            u: 1.1785875939211838
          keep_hist: false
          method: LT
          nsteps: 10
          px: 9.272539685915113
          sub_method: Binomial Tree
        ref: Stock
          S0: 45
          curr: -
          desc: -
          q: 0.02
          tkr: -
          vol: 0.3
        rf_r: 0.05
        seed0: -
        <BLANKLINE>

        >>> s = Stock(S0=100, vol=.4)
        >>> o = Boston(ref=s, right='put', K=100, T=1, rf_r=.08)
        >>> o.calc_px(method='LT', nsteps=5, keep_hist=True).px_spec.px
        14.256042662176432
        >>> o.calc_px(method='LT', nsteps=5, keep_hist=False)
        Boston
        K: 100
        T: 1
        _right: put
        _signCP: -1
        frf_r: 0
        px_spec: PriceSpec
          LT_specs:
            a: 1.016128685406095
            d: 0.8362016906807812
            df_T: 0.9231163463866358
            df_dt: 0.9841273200552851
            dt: 0.2
            p: 0.5002390255615027
            u: 1.1958837337268056
          keep_hist: false
          method: LT
          nsteps: 5
          px: 14.256042662176432
          sub_method: Binomial Tree
        ref: Stock
          S0: 100
          curr: -
          desc: -
          q: 0
          tkr: -
          vol: 0.4
        rf_r: 0.08
        seed0: -
        <BLANKLINE>
        """

        #Verify Input
        assert self.right in ['call', 'put'], 'right must be "call" or "put" '
        assert self.ref.vol > 0, 'vol must be >=0'
        assert self.K > 0, 'K must be > 0'
        assert self.T > 0, 'T must be > 0'
        assert self.ref.S0 >= 0, 'S must be >= 0'
        assert self.rf_r >= 0, 'r must be >= 0'

        keep_hist = getattr(self.px_spec, 'keep_hist', False)
        n = getattr(self.px_spec, 'nsteps', 3)
        _ = self.LT_specs(n)

        S = self.ref.S0 * _['d'] ** np.arange(n, -1, -1) * _['u'] ** np.arange(0, n + 1)  # terminal stock prices
        O = np.maximum(self.signCP * (S - self.K), 0)          # terminal option payouts
        # tree = ((S, O),)
        S_tree = (tuple([float(s) for s in S]),)  # use tuples of floats (instead of numpy.float)
        O_tree = (tuple([float(o) for o in O]),)
        # tree = ([float(s) for s in S], [float(o) for o in O],)

        for i in range(n, 0, -1):
            O = _['df_dt'] * ((1 - _['p']) * O[:i] + ( _['p']) * O[1:])  #prior option prices (@time step=i-1)
            S = _['d'] * S[1:i + 1]                   # prior stock prices (@time step=i-1)
            Payout = np.maximum(self.signCP * (S - self.K), 0)   # payout at time step i-1 (moving backward in time)
            O = np.maximum(O, Payout)
            # tree = tree + ((S, O),)
            S_tree = (tuple([float(s) for s in S]),) + S_tree
            O_tree = (tuple([float(o) for o in O]),) + O_tree
            # tree = tree + ([float(s) for s in S], [float(o) for o in O],)
        O *= math.exp(self.rf_r * self.T)

        self.px_spec.add(px=float(Util.demote(O)), method='LT', sub_method='Binomial Tree',
                        LT_specs=_, ref_tree = S_tree if keep_hist else None, opt_tree = O_tree if keep_hist else None)

        # self.px_spec = PriceSpec(px=float(Util.demote(O)), method='LT', sub_method='binomial tree; Hull Ch.13',
        #                 LT_specs=_, ref_tree = S_tree if save_tree else None, opt_tree = O_tree if save_tree else None)
        return self


if __name__ == "__main__":
    import doctest
    doctest.testmod()
"""
s = Stock(S0=100, vol=.4)
o = Boston(ref=s, right='put', K=100, T=1, rf_r=.08)
print(o.calc_px(method='LT', nsteps=5, keep_hist=True).px_spec.px)
print(o.calc_px(method='LT', nsteps=5, keep_hist=False))
"""
