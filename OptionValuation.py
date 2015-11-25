import math
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import itertools
from Util import *


class PriceSpec(SpecPrinter):
    """ Object for storing calculated price and related intermediate parameters.

    Use this object to store the price, sub/method and any intermediate results in your option object.

    """
    px = None  # use float data type
    method = None  # 'BS', 'LT', 'MC', 'FD'
    sub_method = None   # indicate specifics about pricing method. ex: 'lsm' or 'naive' for mc pricing of American

    def __init__(self, **kwargs):
        """ Constructor.

        Calls add() method to save named input variables.

        Parameters
        ----------
        kwargs : object, optional
            any named input (key=value, key=value,...) that needs to be stored at PriceSpec

        """
        self.add(**kwargs)

    def add(self, **kwargs):
        """ Adds all key/value input arguments as class variables

        Parameters
        ----------
        kwargs : object, optional
            any named input (key=value, key=value,...) that needs to be stored at PriceSpec

        Returns
        -------
        self : PriceSpec

        """
        for K, v in kwargs.items():
            if v is not None:  setattr(self, K, v)
        return self

    # def __repr__(self):
    #     """ Compiles a printable representation of the object
    #
    #     Returns
    #     -------
    #     str
    #         Return a string containing a printable representation of an object.
    #
    #     Examples
    #     --------
    #     >>> PriceSpec(a=1, b='2', c=2.0)  # instantiates an object, saves its input, prints out structure
    #     OptionValuation.PriceSpec
    #     a: 1
    #     b: '2'
    #     c: 2.0
    #     <BLANKLINE>
    #
    #     """
    #     s = yaml.dump(self, default_flow_style=0).replace('!!python/object:','').replace('!!python/tuple','')
    #     s = s.replace('__main__.','')
    #     return s


class Stock(SpecPrinter):
    """ Object for storing parameters of an underlying (referenced) asset.

    .. sectionauthor:: Oleg Melnikov

    Sets parameters of an equity stock share: S0, vol, ticker, dividend yield, curr, tkr ...

    """
    def __init__(self, S0=None, vol=None, q=0, curr=None, tkr=None, desc=None):
        """ Constructor.

        Parameters
        ----------
        S0 : float
            stock price today ( or at the time of evaluation), positive number. used in pricing options.
        vol : float
            volatility of this stock as a rate, positive number. used in pricing options.
            ex. if volatility is 30%, enter vol=.3
        q : float
            dividend yield rate, usually used with equity indices. optional
        curr : str
            currency name/symbol of this stock... optional
        tkr : str
            stock ticker. optional.
        desc : dict
            any additional information related to the stock.

        Examples
        --------
        >>> Stock(S0=50, vol=0.2, tkr='MSFT')   # doctest: +NORMALIZE_WHITESPACE
        Stock
        S0: 50
        curr: -
        desc: -
        q: 0
        tkr: MSFT
        vol: 0.2

        """
        self.S0, self.vol, self.q, self.curr, self.tkr, self.desc = S0, vol, q, curr, tkr, desc


class OptionSeries(SpecPrinter):
    """ Object representing an option series.

    This class describes the option specs outside of valuation. so, it doesn't contain interest rates needed for pricing.
    This class can be used for plotting and evaluating option packages (strategies like bull spread, straddle, ...).
    It can also be inherited by classes that require an important extension - option valuation.

    Sets option series specifications: ref, K, T, .... this is a ligth object with only a few methods.

    .. sectionauthor:: Oleg Melnikov

    """
    def __init__(self, ref=None, right=None, K=None, T=None, clone=None, desc=None):
        r""" Constructor.

        If clone object is supplied, its specs are used.

        Parameters
        ----------
        ref : object
            any suitable object of an underlying instrument (must have S0 & vol variables).
                required, if clone = None.
        right : {'call', 'put', 'other'}
            'call', 'put', and 'other' (for some exotic instruments). required, if clone = None.
        K : float
            strike price, positive number. required, if clone = None.
        T : float
            time to maturity, in years, positive number. required, if clone = None.
        clone : OptionValuation, European, American, any child of OptionValuation, optional
            another option object from which this object will inherit specifications. optional.
            this is useful if you want to price European option as (for example) American.
            then European option's specs will be used to create a new American option. just makes things simple.
        desc : dict
            any number of describing variables. optional.

        Examples
        --------
        Examples show different ways of printing specs (parameters) of the objects

        >>> OptionSeries(ref=Stock(S0=50, vol=0.3), K=51, right='call').full_spec(new_line=False)
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        'OptionSeries {K:51, _right:call, _signCP:1,...ref:Stock {S0:50, curr:-, desc:-, q:0, ,  tkr:-, vol:0.3}},'

        >>> print(OptionSeries(ref=Stock(S0=50, vol=0.3, tkr='ibm', curr='usd'), K=51, right='call').full_spec(True))
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        OptionSeries
        K: 51
        _right: call
        _signCP: 1
        px_spec: PriceSpec {}
        ref: Stock
          S0: 50
          curr: usd
          desc: -
          q: 0
          tkr: ibm
          vol: 0.3

        >>> o = OptionSeries(ref=Stock(S0=50,vol=.03)); repr(o)
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        'OptionSeries\npx_spec: PriceSpec {}\nref: Stock\n  S0: 50\n
        curr: -\n  desc: -\n  q: 0\n  tkr: -\n  vol: 0.03\n'
        >>> o  # equivalent to print(repr(o))
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        OptionSeries
        px_spec: PriceSpec {}
        ref: Stock
          S0: 50
          curr: -
          desc: -
          q: 0
          tkr: -
          vol: 0.03

        >>> o = OptionSeries(ref=Stock(S0=50,vol=.03));  str(o)
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        'OptionSeries\npx_spec: PriceSpec {}\nref: Stock\n  S0: 50\n
        curr: -\n  desc: -\n  q: 0\n  tkr: -\n  vol: 0.03\n'
        >>> print(str(o))
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        OptionSeries
        px_spec: PriceSpec {}
        ref: Stock
          S0: 50
          curr: -
          desc: -
          q: 0
          tkr: -
          vol: 0.03

        """
        self.update(ref=ref, right=right, K=K, T=T, clone=clone, desc=desc)

    def update(self, **kwargs):
        """ Updates current objects' parameters

        Use this method to add/update any specification for the current option.

        Parameters
        ----------
        **kwargs :
            parameters (key=value,  key=value, ...) that needs to be updated

        Examples
        --------

        >>> o = OptionSeries(ref=Stock(S0=50, vol=.3), right='put', K=52, T=2).update(K=53) # sets new strike
        >>> o   # prints structure of the object to screen
        OptionSeries
        K: 53
        T: 2
        _right: put
        _signCP: -1
        px_spec: PriceSpec {}
        ref: Stock
          S0: 50
          curr: -
          desc: -
          q: 0
          tkr: -
          vol: 0.3
        <BLANKLINE>

        >>> OptionSeries(clone=o, K=54).update(right='call')  # copy parameters from o; changes strike & right
        OptionSeries
        K: 54
        T: 2
        _right: call
        _signCP: 1
        px_spec: PriceSpec {}
        ref: Stock
          S0: 50
          curr: -
          desc: -
          q: 0
          tkr: -
          vol: 0.3
        <BLANKLINE>

        """
        self.reset()   # delete old calculations, before updating parameters

        # First, clone an object, then update remaining parameters
        if 'clone' in kwargs:
            self.clone = kwargs['clone']
            del kwargs['clone']

        for K, v in kwargs.items():
            if v is not None: setattr(self, K, v)

        return self

    def get_right(self):
        """ Returns option's right as a string.

        This is a getter method that hides direct access to the right attribute.

        Returns
        -------
        str
            'call', 'put', or 'other'

        """
        if getattr(self, '_right') is None:
            warnings.warn('Hmmm... I will use "call" right, since you did not provide any', UserWarning)
            self._right = 'call'

        return self._right
        # try:
        #     return self._right
        # except:
        #     warnings.warn('Cannot access self._right in OptionValuation.get_right() method', UserWarning)
        #     return ''

    def set_right(self, right='call'):
        """ Sets option's right to a new string.

        This is a setter method that hides direct access to the right attribute.

        Parameters
        ----------
        right : str
            Right of the option: 'cal', 'put', or other valid options.

        Returns
        -------
        self : OptionSeries

        """
        if right is not None:
            self._right = right.lower()
            self._signCP = 1 if self._right == 'call' else -1 if self._right == 'put' else 0  # 0 for other rights
        return self

    right = property(get_right, set_right, None, 'option\'s right (str): call or put')

    @property
    def signCP(self):
        """ Identifies a sign (+/-) indicating the right of the option.

        This property is convenient in calculations, which have parts with sign depending on the option's right.
        There is no setter property for `signCP`, instead it must be set via `right` property.

        Returns
        -------
        int
            +1 if the option is a call
            -1 if the option is a put
            0 for other rights of the option

        """
        return self._signCP   # defines a getter attribute (property)

    @property
    def style(self):
        """ Returns option style (European, American, bermudan, Asian, Binary,...) as a string.

        It first checks whether this object inherits 'OptionValuation' class,
        i.e. whether this is an exotic option object.
        Option style can be drawn from the class name. see example.

        Returns
        -------
        str, None
            Option style for objects inheriting OptionValuation

        Examples
        --------

        >>> from qfrm import *; American().style
        'American'
        >>> from qfrm import *; European().style
        'European'
        >>> OptionSeries().style  # returns None
        """
        if any('OptionValuation' == i.__name__ for i in self.__class__.__bases__):
            return type(self).__name__
        else:
            return None

    @property
    def series(self):
        """ Compiles option series name.

        Compiles an option series name (as a string), including option style (European, American, ...)

        Returns
        -------
        str
            Option series name

        Examples
        --------
        >>> from qfrm import *
        >>> OptionSeries(ref=Stock(S0=50, vol=0.3), K=51, right='call').series
        '51 call'
        >>> OptionSeries(ref=Stock(S0=50, vol=0.3, tkr='IBM'), K=51, right='call').series
        'IBM 51 call'
        >>> OptionSeries(ref=Stock(S0=50, vol=0.3, tkr='IBM'), K=51, T=2, right='call').series
        'IBM 51 2yr call'
        >>> American(ref=Stock(S0=50, vol=0.3), K=51, right='call').series
        '51 American call'

        """
        try: tkr = self.ref.tkr + ' '
        except: tkr=''

        K = '' if getattr(self, 'K', None) is None else str(self.K) + ' '
        T = '' if getattr(self, 'T', None) is None else str(self.T) + 'yr '
        style = '' if self.style is None else self.style + ' '
        right = '' if getattr(self, 'right', None) is None else str(self.right) + ' '

        return (tkr + K + T + style + str(right)).rstrip()  # strip trailing spaces

    @property
    def specs(self):
        """ Compile option series, rfr, foreign rfr, volatility, dividend yield

        Returns
        -------
        str
            Option pricing specifications, including interest rates, volatility, ...

        Examples
        --------

        >>> from qfrm import *; s = Stock(S0=50, vol=0.3, tkr='IBM')
        >>> OptionSeries(ref=s, K=51, right='call').specs
        'IBM 51 call, Stock {S0:50, curr:-, desc:-, q:0, tkr:IBM, , vol:0.3},'
        >>> American(ref=Stock(S0=50, vol=0.3), K=51, right='call').specs
        '51 American call, Stock {S0:50, curr:-, desc:-, q:0, tkr:-, , vol:0.3}, rf_r=None frf_r=0'

        """
        try: ref = self.ref.full_spec(new_line=False)
        except: ref = ''

        frf_r = (' frf_r=' + str(self.frf_r)) if hasattr(self, 'frf_r') else ''
        rf_r = (' rf_r=' + str(self.rf_r)) if hasattr(self, 'rf_r') else ''

        return self.series + ', ' + ref + rf_r + frf_r

    @property
    def clone(self):  return self

    @clone.setter
    def clone(self, clone=None):
        """ Inherits parameters from specified (cloned) option.

        All parameters will be copied into this (current) option object.

        Parameters
        ----------
        clone : OptionSeries, OptionValuation, European, American, ...
            Target option object that needs to be duplicated.

        Examples
        --------

        >>> o = OptionSeries(right='call');
        >>> OptionSeries(clone=o).right  # create new option similar to o
        'call'

        >>> from qfrm import *; American(clone=European(frf_r=.05))  # create American similar to European
        ... # doctest: +ELLIPSES, +NORMALIZE_WHITESPACE
        American
        frf_r: 0.05
        px_spec: OptionValuation.PriceSpec {}
        rf_r: -
        seed0: -

        """
        # copy specs from supplied object
        if clone is not None:  [setattr(self, v, getattr(clone, v)) for v in vars(clone)]

    def reset(self):
        """ Erase calculated parameters.

        Returns
        -------
        self : OptionValuation

        """
        self.px_spec = PriceSpec(px=None)
        return self


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
        px_spec: PriceSpec {}
        ref: Stock
          S0: 50
          curr: -
          desc: -
          q: 0
          tkr: -
          vol: -
        rf_r: 0.05
        seed0: -
        <BLANKLINE>
        """
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
        {'a': 1.0253151205244289, 'd': 0.9048374180359595, 'df_T': 0.951229424500714,
         'df_dt': 0.9753099120283326, 'dt': 0.25, 'p': 0.60138570166548, 'u': 1.1051709180756477}

        >>> s = Stock(S0=50, vol=.3)
        >>> pprint(OptionValuation(ref=s,right='put', K=52, T=2, rf_r=.05, desc={'See Hull p.288'}).LT_specs(3))
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        {'a': 1.033895113513574, 'd': 0.7827444773247475, 'df_T': 0.9048374180359595,
         'df_dt': 0.9672161004820059, 'dt': 0.6666666666666666, 'p': 0.5075681589595774, 'u': 1.2775561233185384}

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
        :return : None
            Plot the price convergence.

        Examples
        --------

        See J.C.Hull OFOD, 9ed, p.289, Fig 13.10, 2-step Binomial tree for American put option.
        The following produces two trees: stock price progressions and option proce backward induction.
        >>> from qfrm import *;  s = Stock(S0=50, vol=.3)
        >>> a = American(ref=s, right='put', K=52, T=2, rf_r=.05, desc={'$7.42840, See Hull p.289'})
        >>> ref_tree = a.calc_px(method='LT', nsteps=20, keep_hist=True).px_spec.ref_tree
        >>> a.plot_bt(bt=ref_tree, title='Binary tree of stock prices; ' + a.specs) # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at 0x...>
        >>> a.plot_bt(bt=a.px_spec.opt_tree, title='Binary tree of option prices; ' + a.specs)# doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at 0x...>

        """
        # import itertools; ax = None; bt = ((4,), (3, 5), (2, 4, 6), (1, 3, 5, 7))

        if ax is None: fig, ax = plt.subplots()
        if 'fig' in locals():
            def onresize(event):
                try: plt.tight_layout()
                except: pass
            cid = fig.canvas.mpl_connect('resize_event', onresize)  # tighten layout on resize event

        def pairs(t):
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

        annotated = set()
        for l in list(pairs(bt)):
            d = [[p[0] for p in l], [p[1] for p in l]]
            ax.plot(d[0], d[1], 'k--', color='.5')
            for p in l:
                annotated.add(p)

        for p in annotated:
            ax.annotate(str(round(p[1],2)), xy=p, horizontalalignment='center', color='0')
        # from pprint import pprint as pp
        # pp(list(pairs(t)),compact=1)
        plt.grid()
        plt.tight_layout();
        plt.show()
        plt.title(title)

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
        :return : None
            Plot the price convergence.

        Examples
        --------

        >>> from qfrm import *;  s = Stock(S0=50, vol=.3);
        >>> a = American(ref=s, right='put', K=52, T=2, rf_r=.05, desc={'$7.42840, See Hull p.288'})
        >>> e = European(clone=a)
        >>> e.plot_px_convergence(nsteps_max=10)  # doctest: +ELLIPSIS
        >>> a.plot_px_convergence(nsteps_max=10, vs=e)  # doctest: +ELLIPSIS

        """
        if ax is None: fig, ax = plt.subplots()
        if 'fig' in locals():
            def onresize(event):  plt.tight_layout()
            cid = fig.canvas.mpl_connect('resize_event', onresize)  # tighten layout on resize event

        LT_prices = [self.pxLT(nsteps=n) for n in range(1, nsteps_max + 1)]

        pd.DataFrame({'LT price for ' + self.specs: LT_prices,
                   'BS price for ' + self.specs: self.pxBS()}) \
            .plot(ax=ax, grid=1, title='Option price convergence: price vs number of steps')

        if vs is not None: vs.plot_px_convergence(nsteps_max=nsteps_max, ax=ax)

        plt.tight_layout();
        plt.show()

    def plot(self, nsteps_max=10):
        """ Plot multiple subplots in a single panel.

        Parameters
        ----------
        nsteps_max : int
            Indicates max number of steps for plotting price vs. number of steps

        Examples
        --------
        See J.C.Hull, OFOD, 9ed, p.288-289.
        This example demonstrates change in (convergence of) price with increasing number of steps of binary tree.
        >>> from qfrm import *; s = Stock(S0=50, vol=.3)
        >>> a = American(ref=s, right='put', K=52, T=2, rf_r=.05, desc={'$7.42840, See Hull p.288'})
        >>> a.plot(nsteps_max=5) # doctest: +ELLIPSIS
        <matplotlib.axes._subplots.AxesSubplot object at 0x...>

        """
        fig, ax = plt.subplots()
        def onresize(event):  fig.tight_layout()
        cid = fig.canvas.mpl_connect('resize_event', onresize)  # tighten layout on resize event

        self.plot_px_convergence(nsteps_max=nsteps_max, ax=ax)
        plt.tight_layout();
        plt.show()

    @property
    def net_r(self):
        """

        Returns
        -------
        float
            Net value of interest rate used to price this option

        Examples
        --------
        >>> from pprint import pprint; from qfrm import *
        >>> o = OptionValuation(rf_r=0.05); pprint(vars(o))  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        {'frf_r': 0,...'rf_r': 0.05,...}
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
        European ... px: 3.444364288840312 ...

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

        >>> from qfrm import *; European(ref=Stock(S0=50, vol=.2), rf_r=.05, K=50, T=0.5, right='call').pxBS()
        3.444364288840312

        """
        return self.calc_px(method='BS', **kwargs).px_spec.px

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

        >>> from qfrm import *; European(ref=Stock(S0=50, vol=.2), rf_r=.05, K=50, T=0.5, right='call').pxLT()
        3.6693707022743633

        """
        return self.calc_px(method='LT', **kwargs).px_spec.px

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

        >>> from qfrm import *; European(ref=Stock(S0=50, vol=.2), rf_r=.05, K=50, T=0.5, right='call').pxMC()

        """
        return self.calc_px(method='MC', **kwargs).px_spec.px

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

        >>> from qfrm import *; European(ref=Stock(S0=50, vol=.2), rf_r=.05, K=50, T=0.5, right='call').pxFD()

        """
        return self.calc_px(method='FD', **kwargs).px_spec.px


