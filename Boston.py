import math
import numpy as np

try: from qfrm.European import *  # production:  if qfrm package is installed
except:   from European import *  # development: if not installed and running from source


class Boston(European):
    """ Boston Option Valuation Class

    Inherits all methods and properties of OptionValuation class.
    """

    def calc_px(self, **kwargs):
        """ Wrapper function that calls appropriate valuation method.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments (``method``, ``nsteps``, ``npaths``, ``keep_hist``, ``rng_seed``, ...)
            are passed to the parent. See ``European.calc_px()`` for details.

        Returns
        -------
        self : Boston
            Returned object contains specifications and calculated price in  ``px_spec`` variable (``PriceSpec`` object).


        Notes
        -----
        A Boston option is simply an American option except that the premium is paid at the end of the contract instead of
        the beginning as done in a normal American option.  Because of this, the price of the option has to be calculated
        as the American price NPV.

        *References:*

        - Exotic Options (Ch.19 Lecture Slides), `John J. Stansfield, 2002 <http://1drv.ms/1NEhkjz>`_, see slide 4
        - Pay Later Option - A very simple Structured Product, `Team Latte, 2007 <http://1drv.ms/1NEgVxi>`_
        - pp.589-599 in `OFOD <http://www-2.rotman.utoronto.ca/~hull/ofod/index.html>`_, J.C.Hull, 9ed, 2014


        Examples
        -------
        >>> s = Stock(S0=50, vol=.3)
        >>> o = Boston(ref=s, right='put', K=52, T=2, rf_r=.05)
        >>> o.pxLT(nsteps=2, keep_hist=True)
        8.209653751

        >>> o.px_spec.ref_tree
        ((50.000000000000014,), (37.0409110340859, 67.49294037880017), (27.440581804701324, 50.00000000000001, 91.10594001952546))

        >>> o.calc_px(method='LT', nsteps=2, keep_hist=False)  # doctest: +ELLIPSIS
        Boston...px: 8.209653751...

        >>> s = Stock(S0=45, vol=.3, q=.02)
        >>> o = Boston(ref=s, right='call', K=52, T=3, rf_r=.05)
        >>> o.pxLT(nsteps=10, keep_hist=True)
        9.272539686

        >>> o   # display all specifications    # doctest: +ELLIPSIS
        Boston...px: 9.272539686...

        >>> s = Stock(S0=100, vol=.4)
        >>> o = Boston(ref=s, right='put', K=100, T=1, rf_r=.08)
        >>> o.pxLT(nsteps=5, keep_hist=True)
        14.256042662


        :Authors:
            Andrew Weatherly
        """

        self.save2px_spec(**kwargs)
        return getattr(self, '_calc_' + self.px_spec.method.upper())()
        # self.px_spec = PriceSpec(method=method, nsteps=nsteps, npaths=npaths, keep_hist=keep_hist)
        # return getattr(self, '_calc_' + method.upper())()

    def _calc_LT(self):
        """ Internal function for option valuation. See ``calc_px()`` for complete documentation.

        :Authors:
            Andrew Weatherly
        """

        #Verify Input
        assert self.right in ['call', 'put'], 'right must be "call" or "put" '
        assert self.ref.vol > 0, 'vol must be >=0'
        assert self.K > 0, 'K must be > 0'
        assert self.T > 0, 'T must be > 0'
        assert self.ref.S0 >= 0, 'S must be >= 0'
        assert self.rf_r >= 0, 'r must be >= 0'

        keep_hist = self.px_spec.keep_hist
        n = self.px_spec.nsteps
        _ = self._LT_specs()

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
        #O_tree = Util.demote([i - O for i in O_tree])
        self.px_spec.add(px=float(Util.demote(O)), method='LT', sub_method='Binomial Tree',
                        LT_specs=_, ref_tree = S_tree if keep_hist else None, opt_tree = O_tree if keep_hist else None)

        # self.px_spec = PriceSpec(px=float(Util.demote(O)), method='LT', sub_method='binomial tree; Hull Ch.13',
        #                 LT_specs=_, ref_tree = S_tree if save_tree else None, opt_tree = O_tree if save_tree else None)
        return self


