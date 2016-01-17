import math
import re
import warnings
import itertools
import pandas as pd

# TravisCI doesn't have an Xwindows backend, causing tests to fail on plot generation.
# This forces matplotlib to not use any Xwindows backend.
# See http://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

try: from qfrm.Util import *  # production:  if qfrm package is installed
except:   from Util import *  # development: if not installed and running from source


class PriceSpec(SpecPrinter):
    """ PriceSpec verifies and saves calculated price and intermediate calculations.

    Use this object to store the price, sub/method and any intermediate results in your option object.

    A typical structure for option with price computed by **LT** method:

    .. code::

          LT_specs:
            a: 1.025315121
            d: 0.904837418
            df_T: 0.951229425
            df_dt: 0.975309912
            dt: 0.25
            p: 0.601385702
            u: 1.105170918
          keep_hist: false
          method: LT
          nsteps: 2
          px: 4.799241115
          sub_method: binary tree; Hull p.135

    A typical structure for option with price computed by **BS** method:

    .. code::

          BS_specs:
            N_d1: 0.220868709
            N_d2: 0.265053963
            Nd1: 0.779131291
            Nd2: 0.734946037
            d1: 0.769262628
            d2: 0.627841272
          keep_hist: false
          method: BS
          px: 4.759422393
          px_call: 4.759422393
          px_put: 0.808599373
          sub_method: standard; Hull p.335

    :Authors:
        Oleg Melnikov <xisreal@gmail.com>
    """
    px = None  # use float data type
    method = None  # 'BS', 'LT', 'MC', 'FD'
    sub_method = None   # indicate specifics about pricing method. ex: 'lsm' or 'naive' for mc pricing of American

    def __init__(self, print_precision=9, **kwargs):
        """ Constructor.

        Calls ``add()`` method to save named input variables.
        See ``add()`` method for further details.

        Parameters
        ----------
        print_precision : int, optional
            Sets number of decimal digits to which printed output is rounded;
            used with whole object print out and with print out of some calculated values (``px``, ...)
            Default 9 digits. If set to ``None``, machine precision is used.
        kwargs : object, optional
            any named input (key=value, key=value,...) that needs to be stored at ``PriceSpec``

        Examples
        --------

        Default ``print_precision = 9`` is used
        """
        SpecPrinter.print_precision = print_precision
        self.add(**kwargs)

    def add_verify(self, dtype=None, min=None, max=None, dflt=None, **kwargs):
        """ Asserts the type and range of passed ``kwargs`` parameter *key*=*value*.

        Use this function to validate and save user's input.
        If assertion fails, default value is used and message is saved into PriceSpec variable as ``[key]_warning``.
        Only the first kwargs argument will be saved.
        Use this method once for each *key*=*value* pair.

        Parameters
        ----------
        dtype : {None, int, long, ...}
            Specifies the type of the input variable

            ``None`` results in no constraint on type of *value*
        min : {None, number}
            Specifies the minimum of the range for the *value*. min must work with operator >=

            ``None`` results in no constraint on minimum of *value*
        max : {None, number}
            Specifies the maximum of the range for the *value*. min must work with operator >=

            ``None`` results in no constraint on maximum of *value*
        dflt : object
            If range/type assertions failed, this (default) value will be used.
        kwargs :
            A single *key*=*value* pair that needs to be validated and stored.

        Returns
        -------


        Examples
        --------

        Here we add ``nsteps=5`` assuring that 1 < 5 and 5 is of type ``int``.
        Conditions are satisfied and value is added to ``PriceSpec`` object.

        >>> ps = PriceSpec()
        >>> ps.add_verify(dtype=int, min=1, max=None, dflt=3, nsteps=5); ps
        PriceSpec
        nsteps: 5

        >>> ps.add_verify(dtype=int, min=1, max=10, dflt=3, nsteps=11); ps
        PriceSpec
        nsteps: 3
        nsteps_warning: bad spec nsteps=11. Must be 1 <= int <= 10. Using default 3

        >>> ps.add_verify(dtype=float, min=1, max=float("inf"), dflt=3, nsteps=11); ps
        PriceSpec
        nsteps: 3
        nsteps_warning: bad spec nsteps=11. Must be 1 <= float <= inf. Using default 3

        >>> ps.add_verify(dtype=float, min=float("-inf"), max=float("inf"), dflt=5, nsteps='bla'); ps
        PriceSpec
        nsteps: 5
        nsteps_warning: bad spec nsteps=bla. Must be -inf <= float <= inf. Using default 5

        >>> ps.add_verify(dtype=float, min=float("-inf"), max=float("inf"), dflt=5, nsteps=None); ps
        PriceSpec
        nsteps: 5
        nsteps_warning: bad spec nsteps=None. Must be -inf <= float <= inf. Using default 5
        """
        k, v = tuple(kwargs.keys())[0], tuple(kwargs.values())[0]

        use_default = v is None
        if not (use_default or dtype is None):
            if not isinstance(v, dtype): use_default = True
        if not (use_default or min is None):
            if v < min: use_default = True
        if not (use_default or max is None):
            if v > max: use_default = True

        if use_default:
            msg = 'bad spec ' + k + '=' + str(v) \
                + '. Must be ' + str(min) + ' <= ' + dtype().__class__.__name__ + ' <= ' + str(max)\
                + '. Using default ' + str(dflt)
            # msg = '\nOooops! PriceSpec.add_verify() says: \n\tinput ' + k + '=' + str(v) + ' must be of type ' + str(dtype) \
            #     + ' with min=' + str(min) + ', max=' + str(max) + ', default=' + str(dflt) \
            #     + '. Using default.'
            # warnings.warn(msg, UserWarning)
            v = dflt
            setattr(self, k + '_warning', msg)

        setattr(self, k, v)

    def add(self, **kwargs):
        """ Adds all key/value input arguments as class variables

        Parameters
        ----------
        kwargs : optional
            any named input (key=value, key=value,...) that needs to be stored at PriceSpec

        Returns
        -------
        self : PriceSpec

        """
        for K, v in kwargs.items():
            if v is not None:  setattr(self, K, v)
        return self


class Stock(SpecPrinter):
    """ Object for storing parameters of an underlying (referenced) asset.

    Sets parameters of an equity stock share: S0, vol, ticker, dividend yield, curr, tkr ...

    :Authors:
        Oleg Melnikov <xisreal@gmail.com>
    """
    def __init__(self, S0=None, vol=None, q=0, curr=None, tkr=None, desc=None, print_precision=9):
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
        print_precision : int, optional
            Sets number of decimal digits to which printed output is rounded;
            used with whole object print out and with print out of some calculated values (``px``, ...)
            Default 9 digits. If set to ``None``, machine precision is used.


        Examples
        --------
        >>> Stock(S0=50, vol=1/7, tkr='MSFT')  # uses default print_precision of 9 digits
        Stock
        S0: 50
        q: 0
        tkr: MSFT
        vol: 0.142857143

        >>> Stock(S0=50, vol=1/7, tkr='MSFT', print_precision=4) # doctest: +ELLIPSIS
        Stock...vol: 0.1429

        """
        self.S0, self.vol, self.q, self.curr, self.tkr, self.desc = S0, vol, q, curr, tkr, desc
        # if 'print_precision' in kwargs: super().__init__(print_precision=kwargs['print_precision'])
        # super().__init__(print_precision=print_precision)
        SpecPrinter.print_precision = print_precision


class OptionSeries(SpecPrinter):
    """ Object representing an option series.

    This class describes the option specs outside of valuation.
    So, it doesn't contain interest rates needed for pricing.
    This class can be used for plotting and evaluating option packages (strategies like bull spread, straddle, ...).
    It can also be inherited by classes that require an important extension - option valuation.

    Sets option series specifications: ``ref``, ``K``, ``T``, .... this is a ligth object with only a few methods.

    :Authors:
        Oleg Melnikov <xisreal@gmail.com>
    """
    def __init__(self, ref=None, right=None, K=None, T=None, clone=None, desc=None, print_precision=9):
        r""" Constructor.

        If clone object is supplied, its specs are used.

        Parameters
        ----------
        ref : object
            any suitable object of an underlying instrument (must have ``S0`` & ``vol`` variables).
                required, if ``clone = None``.
        right : {'call', 'put', 'other'}
            'call', 'put', and 'other' (for some exotic instruments). required, if ``clone = None``.
        K : float
            strike price, positive number. required, if ``clone = None``.
        T : float
            time to maturity, in years, positive number. required, if ``clone = None``.
        clone : OptionValuation, European, American, any child of OptionValuation, optional
            another option object from which this object will inherit specifications.
            this is useful if you want to price European option as (for example) American.
            then European option's specs will be used to create a new American option. just makes things simple.
        desc : dict, optional
            any number of describing variables.
        print_precision : int, optional
            Sets number of decimal digits to which printed output is rounded;
            used with whole object print out and with print out of some calculated values (``px``, ...)
            Default 9 digits. If set to ``None``, machine precision is used.


        Examples
        --------
        Various ways of printing specifications (parameters) of the objects (which inherit ``SpecPrinter``).

        The default (floating point number) precision of printed values (9 decimals) is used.
        Note precision of ``vol`` variable:

        >>> OptionSeries(ref=Stock(S0=50, vol=1/7, tkr='IBM', curr='USD'), K=51, right='call')
        OptionSeries
        K: 51
        _right: call
        _signCP: 1
        px_spec: PriceSpec{}
        ref: Stock
          S0: 50
          curr: USD
          q: 0
          tkr: IBM
          vol: 0.142857143

        The following uses built-in ``repr()`` function,
        which calls object's ``__repr__()`` method.

        >>> repr(OptionSeries(ref=Stock(S0=50,vol=1/7)))
        'OptionSeries\npx_spec: PriceSpec{}\nref: Stock\n  S0: 50\n  q: 0\n  vol: 0.142857143'

        The following shows how to control precision temporarily.
        If needed, default precision can be changed in ``SpecPrinter.full_spec()`` definition.

        >>> OptionSeries(ref=Stock(S0=50, vol=1/7), K=51, print_precision=2).full_spec(print_as_line=True)
        'OptionSeries{K:51, px_spec:PriceSpec{}, ref:Stock{S0:50, q:0, vol:0.14}}'

        The following is a bit more cumbersome way to print object's structure
        with a custom precision.

        >>> print(OptionSeries(ref=Stock(S0=50, vol=1/7), K=51).full_spec(print_as_line=0))
        OptionSeries
        K: 51
        px_spec: PriceSpec{}
        ref: Stock
          S0: 50
          q: 0
          vol: 0.142857143


        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """
        self.update(ref=ref, right=right, K=K, T=T, clone=clone, desc=desc)
        SpecPrinter.print_precision = print_precision

    def update(self, **kwargs):
        """ Updates current objects' parameters

        Use this method to add/update any specification for the current option.

        Parameters
        ----------
        kwargs :
            parameters (key=value,  key=value, ...) that needs to be updated


        Examples
        --------
        >>> o = OptionSeries(ref=Stock(S0=50, vol=.3), right='put', K=52, T=2).update(K=53) # sets new strike
        >>> o      # print out object's variables.
        OptionSeries
        K: 53
        T: 2
        _right: put
        _signCP: -1
        px_spec: PriceSpec{}
        ref: Stock
          S0: 50
          q: 0
          vol: 0.3

        >>> OptionSeries(clone=o, K=54).update(right='call')  # copy parameters from o
        OptionSeries
        K: 54
        T: 2
        _right: call
        _signCP: 1
        px_spec: PriceSpec{}
        ref: Stock
          S0: 50
          q: 0
          vol: 0.3


        :Authors:
            Oleg Melnikov <xisreal@gmail.com>
        """
        self.reset()   # delete old calculations, before updating parameters

        # First, clone an object, then update remaining parameters
        if 'clone' in kwargs:
            if kwargs['clone'] is not None: self.clone = kwargs['clone']
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
            'call', 'put', or 'other' indicating the right of this option object

        """
        if getattr(self, '_right') is None:
            warnings.warn('Hmmm... I will use "call" right, since you did not provide any', UserWarning)
            self._right = 'call'

        return self._right

    def set_right(self, right='call'):
        """ Sets option's right to a new string.

        This is a setter method that hides direct access to the right attribute.

        Parameters
        ----------
        right : str
            Right of the option: 'call', 'put', or other valid options.

        Returns
        -------
        self : object
            Returns this object handle

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
        There is no setter property for ``signCP``, instead it must be set via ``right`` property.

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
        str, ''
            Option style for objects inheriting OptionValuation

        Examples
        --------

        >>> from qfrm import *; American().style
        'American'

        >>> from qfrm import *; European().style
        'European'

        >>> OptionSeries().style
        ''
        """

        return type(self).__name__.replace('OptionValuation', '').replace('OptionSeries', '')
        # if any('European' == i.__name__ for i in self.__class__.__bases__):
        #     return type(self).__name__
        # else:
        #     return None

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
        except: tkr = ''

        K = str(getattr(self, 'K', '')) + ' '
        T = '' if getattr(self, 'T', None) is None else str(self.T) + 'yr '
        style = self.style + ' '
        right = str(getattr(self, 'right', '')) + ' '


        # K = '' if getattr(self, 'K', None) is None else str(self.K) + ' '
        # T = '' if getattr(self, 'T', None) is None else str(self.T) + 'yr '
        # style = '' if self.style is None else self.style + ' '
        # right = '' if getattr(self, 'right', None) is None else str(self.right) + ' '
        # s = re.sub(r'(\s){2,}', repl=' ', string=s)  # replace multiple blanks with one


        return re.sub(r'(\s){2,}', ' ', (tkr + K + T + style + str(right)).strip()) # remove extra spaces

    @property
    def specs(self):
        """ Compile option series, rfr, foreign rfr, volatility, dividend yield

        Returns
        -------
        str
            Option pricing specifications, including interest rates, volatility, ...

        Examples
        --------
        >>> from qfrm import *
        >>> s = Stock(S0=50, vol=0.3, tkr='IBM')
        >>> OptionSeries(ref=s, K=51, right='call').specs
        'IBM 51 call, Stock{S0:50, q:0, tkr:IBM, vol:0.3}'

        >>> American(ref=Stock(S0=50, vol=0.3), K=51, right='call').specs
        '51 American call, Stock{S0:50, q:0, vol:0.3} rf_r=None frf_r=0'

        """
        try: ref = self.ref.full_spec(print_as_line=True)
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

        >>> o = OptionSeries(right='call')
        >>> OptionSeries(clone=o).right  # create new option similar to o
        'call'

        >>> from qfrm import *
        >>> American(clone=European(frf_r=.05))  # create American similar to European
        American
        frf_r: 0.05
        px_spec: PriceSpec{}

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
    def __init__(self, rf_r=None, frf_r=0, *args, **kwargs):
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
        # seed0 : int, None, optional
        #     None or positive integer to seed random number generator (rng).
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
        self.rf_r, self.frf_r = rf_r, frf_r
        super().__init__(*args, **kwargs)  # pass remaining arguments to base (parent) class
        self.reset()

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

        >>> o.update(rf_r=0.04)  # doctest: +ELLIPSIS
        OptionValuation...frf_r: 0...rf_r: 0.04...

        >>> o.update(ref=Stock(q=0.01))  # doctest: +ELLIPSIS
        OptionValuation...frf_r: 0...q: 0.01...rf_r: 0.04...

        >>> o.net_r   # doctest: +ELLIPSIS
        0.03

        """
        try: q = 0 if self.ref.q is None else self.ref.q
        except: q = 0

        frf_r = 0 if self.frf_r is None else self.frf_r
        rf_r = 0 if self.rf_r is None else self.rf_r

        return rf_r - q - frf_r   # calculate RFR net of yield and foreign RFR





