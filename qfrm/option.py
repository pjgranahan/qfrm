import itertools
import math
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from pricespec import PriceSpec

from qfrm.optionseries import OptionSeries

try: from qfrm.Util import *  # production:  if qfrm package is installed
except:   from qfrm.util import *  # development: if not installed and running from source


class OptionValuation(OptionSeries):
    """ Adds interest rates and some methods shared by subclasses.

    The class inherits from a simpler class that describes an option.
    """
    def __init__(self, rf_r=None, frf_r=0, seed0=None, *args, **kwargs):
        """ Constructor saves all identified arguments and passes others to the base (parent) class, OptionSeries.

        It also calculates net_r, the rate used in computing growth factor a (p.452) for options
        with dividends and foreign risk free rates.

        Parameters
        ----------
        rf_r : float
            risk free rate. required, unless clone object supplies it (see OptionSeries constructor).
                number in (0,1) interval
        frf_r : float, optional
            foreign risk free rate.
        seed0 : int, None, optional
            None or positive integer to seed random number generator (rng).
        precision : {None, int}, optional
            indicates desired floating number precision of calculated prices.
            Assists with doctesting due to rounding errors near digits in 10^-12 placements
            If value is None, then precision is ignored and default machine precision is used
        args : object, optional
            arguments to be passed to base class constructor. see base class for types of its arguments
        kwargs : object, optional
            keyword arguments to be passed to base class constructor. see base class for types of its arguments

        Returns
        -------
        __main__.OptionValuation
            __init__() method always implicitly returns self, i.e. a reference to this object

        Examples
        --------

        >>> OptionValuation(ref=Stock(S0=50), rf_r=.05, frf_r=.01)
        OptionValuation
        frf_r: 0.01
        px_spec: PriceSpec{}
        ref: Stock
          S0: 50
          q: 0
        rf_r: 0.05


        """
        # Todo: OptionValuation.__init__(print_precision=4) doesn't work.
        self.rf_r, self.frf_r, self.seed0 = rf_r, frf_r, seed0
        super().__init__(*args, **kwargs)  # pass remaining arguments to base (parent) class
        self.reset()

    def LT_specs(self, nsteps=2):
        """ Calculates a collection of specs/parameters needed for lattice tree pricing.

        parameters returned:
            dt: time interval between consequtive two time steps
            u: Stock price up move factor
            d: Stock price down move factor
            a: growth factor, p.452
            p: probability of up move over one time interval dt
            df_T: discount factor over full time interval dt, i.e. per life of an option
            df_dt: discount factor over one time interval dt, i.e. per step

        Parameters
        ----------
        nsteps : int
            number of steps in a tree, positive number. required.

        Returns
        -------
        dict
            A dictionary of parameters required for lattice tree pricing.

        Examples
        --------
        >>> from pprint import pprint
        >>> pprint(OptionValuation(ref=Stock(S0=42, vol=.2), right='call', K=40, T=.5, rf_r=.1).LT_specs(2))
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        {'a': 1.025315120...'d': 0.904837418...'df_T': 0.951229424...
         'df_dt': 0.975309912...'dt': 0.25, 'p': 0.601385701...'u': 1.105170918...}

        >>> s = Stock(S0=50, vol=.3)
        >>> pprint(OptionValuation(ref=s,right='put', K=52, T=2, rf_r=.05, desc={'See Hull p.288'}).LT_specs(3))
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        {'a': 1.033895113...'d': 0.782744477...'df_T': 0.904837418...
         'df_dt': 0.967216100...'dt': 0.666...'p': 0.507568158...'u': 1.277556123...}

         """
        assert isinstance(nsteps, int), 'nsteps must be an integer, >2'

        sp = {'dt': self.T / nsteps}
        sp['u'] = math.exp(self.ref.vol * math.sqrt(sp['dt']))
        sp['d'] = 1 / sp['u']
        sp['a'] = math.exp(self.net_r * sp['dt'])   # growth factor, p.452
        sp['p'] = (sp['a'] - sp['d']) / (sp['u'] - sp['d'])
        sp['df_T'] = math.exp(-self.rf_r * self.T)
        sp['df_dt'] = math.exp(-self.rf_r * sp['dt'])

        return sp

    def plot_bt(self, bt=None, ax=None, title=''):
        """ Plots recombining binary tree

        Parameters
        ----------
        bt : tuple[tuple[long,...], ...]
            binomial tree
        ax : matplotlib.axes._subplots.axessubplot, optional
            Plot object on which to plot the data.
        vs : object, optional
            another option object (i.e. subclass of OptionValuation such as European, American,...)

        Examples
        --------
        See J.C.Hull, OFOD, 9ed, p.289, Fig 13.10, 2-step Binomial tree for American put option.
        The following produces two trees: stock price progressions and option proce backward induction.

        >>> from qfrm import *;
        >>> s = Stock(S0=50, vol=.3)
        >>> a = American(ref=s, right='put', K=52, T=2, rf_r=.05, desc={'px=$7.42840, See Hull p.289'})
        >>> ref_tree = a.calc_px(method='LT', nsteps=10, keep_hist=True).px_spec.ref_tree

        >>> a.plot_bt(bt=ref_tree, title='Binary tree of stock prices; ' + a.specs)
        >>> a.plot_bt(bt=a.px_spec.opt_tree, title='Binary tree of option prices; ' + a.specs)

        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """
        if ax is None: fig, ax = plt.subplots()
        if 'fig' in locals():  # assures tight layout even when plot is manually resized
            def onresize(event):
                try: plt.tight_layout()
                except: pass
            cid = fig.canvas.mpl_connect('resize_event', onresize)  # tighten layout on resize event

        def edges(t):
            # pairs takes a binary tree:
            #          import itertools; ax = None; bt = ((4,), (3, 5), (2, 4, 6), (1, 3, 5, 7))
            # and produces a list of coordinates for each edge, connecting a pair tree nodes:
            # [[(0, 4), (1, 3)], [(0, 4), (1, 5)], [(1, 3), (2, 2)], [(1, 3), (2, 4)],
            #  [(1, 5), (2, 4)], [(1, 5), (2, 6)], [(2, 2), (3, 1)], [(2, 2), (3, 3)],
            #  [(2, 4), (3, 3)], [(2, 4), (3, 5)], [(2, 6), (3, 5)], [(2, 6), (3, 7)]]
            a, b = itertools.tee(t)
            next(b)
            for ind, (t1, t2) in enumerate(zip(a, b)):
                it = iter(t2)
                nxt = next(it)
                for ele in t1:
                    n = next(it)
                    yield [(ind, ele), (ind + 1, nxt)]
                    yield [(ind, ele), (ind + 1, n)]
                    nxt = n

        points = set()
        for l in edges(bt):
            d = [[p[0] for p in l], [p[1] for p in l]]
            ax.plot(d[0], d[1], 'k--', color='.5')
            for p in l:
                points.add(p)

        for p in points:
            ax.annotate(str(round(p[1],2)), xy=p, horizontalalignment='center', color='0')

        plt.grid()
        # plt.title(title)
        plt.text(x=0, y=max(bt[len(bt)-1]), s=title)
        plt.tight_layout()
        plt.show()

    def plot_px_convergence(self, nsteps_max=50, ax=None, vs=None):
        """ Plots convergence of an option price for different `nsteps` values.

        If `vs` object is provided, its plot is added,
        i.e. execute `vs.plot_px_convergence(...)` to add a plot of the benchmark option.
        This is helpful to compare the convergence of lt price for European versus American options.
        BSM price (a constant line) is also plotted, if available.
        If `ax` argument is omitted, a new `matplotlib.axes` object is created for plotting.

        Parameters
        ----------
        nsteps_max : int
            sets the range of `nsteps`, so that the lt price can be computed for each time step.
            i.e. this is the maximum range of the x-axis on the resulting plot.
            `calc_px` is called with `range(1, nsteps_max)`. Required. positive integer.
        ax : matplotlib.axes._subplots.axessubplot, optional
            Plot object on which to plot the data.
        vs : object, optional
            another option object (i.e. subclass of OptionValuation such as European, American,...)

        Examples
        --------
        >>> from qfrm import *;  s = Stock(S0=50, vol=.3);
        >>> a = American(ref=s, right='put', K=52, T=2, rf_r=.05, desc={'$7.42840, See Hull p.288'})
        >>> e = European(clone=a)
        >>> e.plot_px_convergence(nsteps_max=10)
        >>> a.plot_px_convergence(nsteps_max=10, vs=e)

        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """
        if ax is None: fig, ax = plt.subplots()
        if 'fig' in locals():  # assures tight layout even when plot is manually resized
            def onresize(event):  plt.tight_layout()
            try: cid = fig.canvas.mpl_connect('resize_event', onresize)  # tighten layout on resize event
            except: pass

        LT_prices = [self.pxLT(nsteps=n) for n in range(1, nsteps_max + 1)]

        pd.DataFrame({'LT price for ' + self.specs: LT_prices,
                   'BS price for ' + self.specs: self.pxBS()}) \
            .plot(ax=ax, grid=1, title='Option price convergence: price vs number of steps')

        if vs is not None: vs.plot_px_convergence(nsteps_max=nsteps_max, ax=ax)

        try: plt.tight_layout()
        except: pass
        plt.show()

    def plot(self, nsteps_max=10):
        """ Plot multiple subplots in a single panel.

        Parameters
        ----------
        nsteps_max : int
            Indicates max number of steps for plotting price vs. number of steps

        Examples
        --------
        >>> from qfrm import *; s = Stock(S0=50, vol=.3)
        >>> a = American(ref=s, right='put', K=52, T=2, rf_r=.05, desc={'$7.42840, See Hull p.288'})
        >>> a.plot(nsteps_max=5)
        ... # See J.C.Hull, OFOD, 9ed, p.288-289.
        ... # This example demonstrates change in (convergence of) price with increasing number of steps of binary tree.

        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """
        fig = plt.figure()
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(224)

        if 'fig' in locals():  # assures tight layout even when plot is manually resized
            def onresize(event):  plt.tight_layout()
            try: cid = fig.canvas.mpl_connect('resize_event', onresize)  # tighten layout on resize event
            except: pass

        self.plot_px_convergence(nsteps_max=nsteps_max, ax=ax1)

        if getattr(self.px_spec, 'ref_tree', None) is None:
            self.calc_px(method='LT', nsteps=nsteps_max, keep_hist=True)

        self.plot_bt(bt=self.px_spec.ref_tree, ax=ax2, title='Binary tree of stock prices; ' + self.specs)
        self.plot_bt(bt=self.px_spec.opt_tree, ax=ax3, title='Binary tree of option prices; ' + self.specs)
        # fig, ax = plt.subplots()
        # def onresize(event):  fig.tight_layout()
        # cid = fig.canvas.mpl_connect('resize_event', onresize)  # tighten layout on resize event
        # self.plot_px_convergence(nsteps_max=nsteps_max, ax=ax)

        try: plt.tight_layout()
        except: pass
        plt.show()

    @property
    def net_r(self):
        """ Computes net risk free rate.

        Returns
        -------
        float
            Net value of interest rate used to price this option

        Examples
        --------
        >>> from pprint import pprint; from qfrm import *
        >>> o = OptionValuation(rf_r=0.05); pprint(vars(o))
        {'frf_r': 0, 'px_spec': PriceSpec{}, 'rf_r': 0.05, 'seed0': None}

        >>> o.update(rf_r=0.04)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        OptionValuation...frf_r: 0...rf_r: 0.04...

        >>> o.update(ref=Stock(q=0.01))  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        OptionValuation...frf_r: 0...q: 0.01...rf_r: 0.04...

        >>> o.net_r   # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        0.03

        """
        try: q = 0 if self.ref.q is None else self.ref.q
        except: q = 0

        frf_r = 0 if self.frf_r is None else self.frf_r
        rf_r = 0 if self.rf_r is None else self.rf_r

        return rf_r - q - frf_r   # calculate RFR net of yield and foreign RFR

    def calc_px(self, **kwargs):
        """ Wrapper pricing function.

        Each exotic option overloads `calc_px()` to accept exotic-specific parameters from user.
        Then child's `calc_px()` calls `OptionValuation.calc_px()` to check basic pricing parameters
        and to call the appropriate pricing method.

        Returns
        -------
        self, None
            Returns None, if called on OptionValuation object.
            Returns self (sub-class), if called on class that inherited OptionValuation (these are exotic classes)

        Examples
        --------

        >>> OptionValuation().calc_px()  # prints a UserWarning and returns None
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE, +IGNORE_EXCEPTION_DETAIL

        >>> from qfrm import *; European(ref=Stock(S0=50, vol=.2), rf_r=.05, K=50, T=0.5).calc_px()
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
            assert getattr(self, '_signCP') is not None, 'Ooops. Please supply option right: call, put, ...'
        AttributeError: 'European' object has no attribute '_signCP'

        >>> from qfrm import *; European(ref=Stock(S0=50, vol=.2), rf_r=.05, K=50, T=0.5, right='call').calc_px()
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        European ... px: 3.444364289 ...

        """
        if self.style is None:
            warnings.warn('Assure that calc_px() is overloaded by exotic option class.', UserWarning)
            return None

        else:
            self.px_spec = PriceSpec(**kwargs)
            assert getattr(self, 'ref') is not None, 'Ooops. Please supply referenced (underlying) asset, `ref`'
            assert getattr(self, 'rf_r') is not None, 'Ooops. Please supply risk free rate `rf_r`'
            assert getattr(self, 'K') is not None, 'Ooops. Please supply strike `K`'
            assert getattr(self, 'T') is not None, 'Ooops. Please supply time to expiry (in years) `T`'
            assert getattr(self, '_signCP') is not None, 'Ooops. Please supply option right: call, put, ...'

            return getattr(self, '_calc_' + self.px_spec.method.upper())()

        return None

    def pxBS(self, **kwargs):
        """ Calls exotic pricing method `calc_px()`

        This property calls `calc_px()` method which should be overloaded
        by each exotic option class (inheriting OptionValuation)

        Parameters
        ----------
        kwargs
            Pricing parameters required to price this exotic option. See `calc_px()` for specifics and examples.

        Returns
        -------
        float
            price of the exotic option

        Examples
        --------
        >>> from qfrm import *
        >>> European(ref=Stock(S0=50, vol=.2), rf_r=.05, K=50, T=0.5, right='call').pxBS()
        3.444364289

        """
        return self.print_value(self.calc_px(method='BS', **kwargs).px_spec.px)

    def pxLT(self, **kwargs):
        """ Calls exotic pricing method `calc_px()`

        This property calls `calc_px()` method which should be overloaded
        by each exotic option class (inheriting OptionValuation)

        Parameters
        ----------
        kwargs
            Pricing parameters required to price this exotic option. See `calc_px()` for specifics and examples.

        Returns
        -------
        float
            price of the exotic option

        Examples
        --------
        >>> from qfrm import *
        >>> European(ref=Stock(S0=50, vol=.2), rf_r=.05, K=50, T=0.5, right='call').pxLT()
        3.669370702

        """
        return self.print_value(self.calc_px(method='LT', **kwargs).px_spec.px)

    def pxMC(self, **kwargs):
        """ Calls exotic pricing method `calc_px()`

        This property calls `calc_px()` method which should be overloaded
        by each exotic option class (inheriting OptionValuation)

        Parameters
        ----------
        kwargs
            Pricing parameters required to price this exotic option. See `calc_px()` for specifics and examples.

        Returns
        -------
        float
            price of the exotic option

        Examples
        --------
        >>> from qfrm import *
        >>> European(ref=Stock(S0=50, vol=.2), rf_r=.05, K=50, T=0.5, right='call').pxMC()

        """
        return self.print_value(self.calc_px(method='MC', **kwargs).px_spec.px)

    def pxFD(self, **kwargs):
        """ Calls exotic pricing method `calc_px()`

        This property calls `calc_px()` method which should be overloaded
        by each exotic option class (inheriting OptionValuation)

        Parameters
        ----------
        kwargs
            Pricing parameters required to price this exotic option. See `calc_px()` for specifics and examples.

        Returns
        -------
        float
            price of the exotic option


        Examples
        --------
        >>> from qfrm import *
        >>> European(ref=Stock(S0=50, vol=.2), rf_r=.05, K=50, T=0.5, right='call').pxFD()

        """
        return self.print_value(self.calc_px(method='FD', **kwargs).px_spec.px)


