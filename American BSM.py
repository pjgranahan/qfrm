__author__ = 'Andrew Weatherly'



class Util():
    """ A collection of utility functions, most of which are static methods, i.e. can be called as Util.isiterable().

    FYI: Decorator @staticmethod allows use of functions without initializing an object
    Ex. we can use Util.demote(x) instead of Util().demote(x). It's faster.

    .. sectionauthor:: Oleg Melnikov
    """
    @staticmethod
    def is_iterable(x):
        """
        Checks if x is iterable.
        :param x: any object
        :type x: object
        :return: True if x is iterable, False otherwise
        :rtype: bool
        :Exmaple:

        >>> Util.is_iterable(1)
        False

        >>> Util.is_iterable((1,2,3))
        True

        >>> Util.is_iterable([1,'blah',3])
        True
        """
        try:
            (a for a in x)
            return True
        except TypeError:
            return False

    @staticmethod
    def is_number(x):
        """
        Checks if x is numeric (float, int, complex, ...)
        :param x: any object
        :type x: object
        :return:  True, if x is numeric; False otherwise.
        :rtype: bool
        ..seealso:: Stackoverflow: how-can-i-check-if-my-python-object-is-a-number
        """
        from numbers import Number
        return isinstance(x, Number)

    @staticmethod
    def are_numbers(x):
        """ Checks if x is an iterable of numbers.
        :param x: any object
        :type x: object
        :return: True if x is iterable, False otherwise
        :rtype: bool

        :Example:

        >>> Util.arenumbers(5)
        False

        >>> Util.arenumbers([1,'blah',3.])
        False

        >>> Util.arenumbers([1,'2',3.])
        False

        >>> Util.arenumbers((1, 2., 3. + 4j, 5.4321))
        True

        >>> Util.arenumbers({1, 2., 3. + 4j, 5.4321})
        True

        """
        try:
            return all(Util.is_number(n) for n in x)
        except TypeError:
            return False

    @staticmethod
    def are_bins(x):
        return Util.are_non_negative(x) and Util.is_monotonic(x)

    # @staticmethod
    # def to_tuple(x):
    #     """ Converts an iterable (of numbers) or a number to a tuple of floats.
    #
    #     :param x: any iterable object of numbers or a number
    #     :type x:  numeric|iterable
    #     :return:  tuple of floats
    #     :rtype: tuple
    #     """
    #     assert Util.is_number(x) or Util.is_iterable(x), 'to_tuple() failed: input must be iterable or a number.'
    #     return (float(x),) if Util.is_number(x) else tuple((float(y)) for y in x)

    @staticmethod
    def round_tuple(t, ndigits=5):
        """ Rounds tuple of numbers to ndigits.
        returns a tuple of rounded floats. Used for printing output.
        :param t: tuple
        :type t:
        :param ndigits: number of decimal (incl. period) to keep
        :type ndigits: int
        :return: tuple of rounded numbers
        :rtype: Tuple[float,...,float]
        """
        assert False, 'round_tuple() method is absolete. Use round()'
        # return tuple(round(float(x), ndigits) for x in t)

    @staticmethod
    def cpn2cf(cpn=6, freq=2, ttm=2.1):
        """ Converts regular coupon payment specification to a series of cash flows indexed by time to cash flow (ttcf).

        :param cpn:     annual coupon payment in $
        :type cpn:      float|int
        :param freq:    payment frequency, per anum
        :type freq:     float|int
        :param ttm:     time to maturity of a bond, in years
        :type ttm:      float|int
        :return:        dictionary of cash flows (tuple) and their respective times to cf (tuple)
        :rtype:         dict('ttcf'=tuple, 'cf'=tuple)
        .. seealso:: stackoverflow.com/questions/114214/class-method-differences-in-python-bound-unbound-and-static
        :Example:

        >>> # convert $6 semiannula (SA) coupon bond payments to indexed cash flows
        >>> Util.cpn2cf(6,2,2.1)  # returns {'cf': (3.0, 3.0, 3.0, 3.0, 103.0),  'ttcf': (0.1, 0.6, 1.1, 1.6, 2.1)}
        """
        from numpy import arange

        if cpn == 0: freq = 1  # set frequency to annual for zero coupon instruments
        period = 1./freq            # time (in year units) period between coupon payments
        end = ttm + period / 2.          # small offset (anything less than a period) to assure expiry is included
        start = period if (ttm % period) == 0 else ttm % period  # time length from now till next cpn, yrs
        c = float(cpn)/freq   # coupon payment per period, $

        ttcf = tuple((float(x) for x in arange(start, end, period)))        # times to cash flows (tuple of floats)
        cf = tuple(map(lambda i: c if i < (len(ttcf) - 1) else c + 100, range(len(ttcf)))) # cash flows (tuple of floats)
        return {'ttcf': ttcf, 'cf': cf}

    @staticmethod
    def demote(x):
        """ Attempts to simplify x to a tuple (if x is a more complex data type) or just singleton.
        Basically, demotes to a simpler object, if possible.

        :param x:   any object
        :type x:    any
        :return:    original object or tuple or value of a singleton
        """

        if Util.is_iterable(x):
            x = tuple(e for e in x)
            if len(x) == 1: x = x[0]
        return x

    @staticmethod
    def is_monotonic(x, direction=1, strict=True):
        # http://stackoverflow.com/questions/4983258/python-how-to-check-list-monotonicity
        assert direction in (1,-1), 'Direction must be 1 for up, -1 for down'

        x = Util.to_tuple(x)[::direction]
        y = (x + (max(x) + 1,))
        return all(a < b if strict else a <= b for a, b in zip(y, y[1:]))

    @staticmethod
    def are_same_sign(x, sign=1, ignore_zero=True):
        assert sign in (1,-1), 'sign must be 1 (for positive) or -1 (for negatives)'
        return all(a*sign >= 0 if ignore_zero else a*sign >0 for a in Util.to_tuple(x))

    @staticmethod
    def are_positive(x):
        return Util.are_same_sign(x, 1, False)

    @staticmethod
    def are_non_negative(x):
        return Util.are_same_sign(x, 1, True)

    @staticmethod
    def round(x, prec=5, to_tuple=False):  #, to_float=False):
        """ Recirsively rounds an iterable to the desired precision.
        :param x: tuple
        :type x: iterable
        :param prec: number of decimal (incl. period) to keep
        :type prec: int
        :param to_tuple: indicates whether to keep original data type or convert output to tuple
        :type to_tuple: bool
        :return: tuple of rounded numbers
        :rtype: Tuple[float,...,float]
        :Example:

        >>> x = (1, 1/3, 1/7,[1/11, 1/13, {1/19, 1/29}]);  from numpy import array; a = array(x)
        >>> Util.round(x)
        >>> Util.round(x, to_tuple=True)
        >>> Util.round(array(x))
        >>> Util.round(array, to_tuple=True)

        .. seealso::
            http://stackoverflow.com/questions/24642669/python-quickest-way-to-round-every-float-in-nested-list-of-tuples
        """
        if to_tuple: x = Util.to_tuple(x)
        try:
            return round(x, prec) #if to_float else round(x, prec)
        except TypeError:
            return type(x)(Util.round(y, prec) for y in x)

    @staticmethod
    def to_tuple(a):
        """
        Recursively converts a iterable (and arrays) to tuple.
        :param a: variable to be converted to tuple
        :type a:  iterable|array
        :return:  tuple
        :rtype:  tuple
        :Example:

        >>> from  numpy import array; x = (1, 1/3, 1/7,[1/11, 1/13, {1/19, 1/29}]); a = array(x)
        >>> Util.to_tuple(x)
        >>> Util.to_tuple(a)

        .. seealso::
            http://stackoverflow.com/questions/10016352/convert-numpy-array-to-tuple
        """
        try:
            return tuple((Util.to_tuple(i)) for i in a)
        except TypeError:
            return a


class Stock:
    """ Class representing an underlying instrument.
    .. sectionauthor:: Oleg Melnikov
    Sets parameters of an equity stock share: S0, vol, ticker, dividend yield, curr, tkr ...
    """
    def __init__(self, S0=50, vol=.3, q=0, curr=None, tkr=None, desc=None):
        """ Class object constructor.
        :param S0: stock price today ( or at the time of evaluation), positive number. Used in pricing options.
        :type S0:  float
        :param vol: volatility of this stock as a rate, positive number. Used in pricing options.
            Ex. if volatility is 30%, enter vol=.3
        :type vol:  float
        :param q:   dividend yield rate, usually used with equity indices. Optional
        :type q:    float
        :param curr: currency name/symbol of this stock... Optional
        :type curr:  str
        :param tkr:  stock ticker. Optional.
        :type tkr:   str
        :param desc: any additional information related to the stock.
        :type desc:  dict
        :return:     __init__() method always implicitly returns self, i.e. a reference to this object
        :rtype:      __main__.Stock
        """
        self.S0, self.vol, self.q, self.curr, self.tkr, self.desc = S0, vol, q, curr, tkr, desc


class OptionSeries:
    """ Class representing an option series.

    This class describes the option specs outside of valuation. So, it doesn't contain interest rates needed for pricing.
    This class can be used for plotting and evaluating option packages (strategies like bull spread, straddle, ...).
    It can also be inherited by classes that require an important extension - option valuation.

    Sets option series specifications: ref, K, T, .... This is a ligth object with only a few methods.
    .. sectionauthor:: Oleg Melnikov

    .. seealso::
        http://stackoverflow.com/questions/6535832/python-inherit-the-superclass-init
        http://stackoverflow.com/questions/285061/how-do-you-programmatically-set-an-attribute-in-python
    """
    def __init__(self, ref=Stock(S0=50, vol=.3), right='put', K=52, T=2, clone=None, desc={}):
        """ Constructor.

        If clone object is supplied, its specs are used.

        :param ref: any suitable object of an underlying instrument (must have S0 & vol variables). Required, if clone = None.
        :type ref:  object
        :param right: 'call', 'put', and 'other' for more exotic instruments. Required, if clone = None.
        :type right:  str
        :param K:   strike price, positive number. Required, if clone = None.
        :type K:    float
        :param T:   time to maturity, in years, positive number. Required, if clone = None.
        :type T:    float
        :param clone:   Another option object from which this object will inherit specifications. Optional.
            This is useful if you want to price European option as (for example) American.
            Then European option's specs will be used to create a new American option. Just makes things simple.
        :type clone:  object inherited from OptionValuation
        :param desc:  any number of describing variables. Optional.
        :type desc:   dict
        :return:   __init__() method always implicitly returns self, i.e. a reference to this object
        :rtype:    __main__.OptionSeries
        """
        if clone is None:
            self.ref, self.K, self.T, self.desc = ref, K, T, desc
            self.signCP = 1 if right.lower() == 'call' else -1 if right.lower() == 'put' else 0  # 0 for other rights
        else:                 # copy specs from another class
            [setattr(self, v, getattr(clone, v)) for v in vars(clone)]

    @property   # Allows setting a read-only property accessible as OptionSeries().right
    def right(self):
        """ Returns option's right as a string.
        :return: 'call', 'put', or 'other'
        :rtype: str
        """
        return 'call' if self.signCP == 1 else 'put' if self.signCP == -1 else 'other'

    @property
    def style(self):
        """ Returns option style (European, American, Bermudan, Asian, Binary,...) as a string.
        It first checks whether this object inherited class 'OptionValuation'.
        Option style can be drawn from the class name. See example.
        :return: option style for objects inheriting OptionValuation
        :rtype: str | None

        :Example:

        >>> American().style
        'American'
        >>> European().style
        'European'
        >>> OptionSeries().style  # returns None

        """
        if any('OptionValuation' == i.__name__ for i in self.__class__.__bases__):
            return type(self).__name__
        else:
            return None

    def spec2str(self, complexity=0):
        """ Returns a formatted string containing option specifications
        :param complexity: indicate level of detail and formatting to include in returned string
            0: return series name, including option style (European, American, ...)
            1: return option series, RFR r, foreign RFR rf, volatility, dividend yield q
            2: return all internal option variables (recursively applied to ref object)
            3: same as 2, but indented accordingly and formatted vertically

        For complexity 2 & 3, yaml.dump does most of the work. 0 is an easy concatenation. 1 draws from ref object.
        :type complexity: int
        :return: formatted string with option specifications
        :rtype:  str

        :Example:

        >>> o = American(ref=Stock(S0=50, vol=.3), right='put', K=52, T=2, r=.05, desc={'note':'$7.42840, Hull p.288'})
        >>> o.spec2str(0)
        '52 2yr American put'
        >>> o.spec2str(1)
        '52 2yr American put,S0=50,vol=0.3,r=0.05,q=0,rf=0'
        >>> o.spec2str(2)
        'American,K:52,T:2,desc:, note:$7.42840, Hull p.288,net_r:0.05,r:0.05,ref:Stock, S0:50, curr:null, desc:null, q:0, tkr:null, vol:0.3,rf:0,signCP:-1,'
        >>> print(o.spec2str(3))
        American
        K: 52
        T: 2
        desc:
          note: $7.42840, Hull p.288
        net_r: 0.05
        r: 0.05
        ref: Stock
          S0: 50
          curr: null
          desc: null
          q: 0
          tkr: null
          vol: 0.3
        rf: 0
        signCP: -1

        .. seealso::
            docs.python.org/3.4/library/pprint.html
            stackoverflow.com/questions/3229419/pretty-printing-nested-dictionaries-in-python
            dpinte.wordpress.com/2008/10/31/pyaml-dump-option
            pprint:   alternative formatting module
        """
        o = self
        s = str(o.K) + ' ' + str(o.T) + 'yr ' + type(o).__name__ + ' ' + str(o.right)  # option series name

        if complexity == 1:
            r = rf = q = vol = ''
            if hasattr(o, 'ref'):  # check if attribute (variable) exists in this object
                if hasattr(o.ref, 'S0'): S0 = (',S0=' + str(o.ref.S0))
                if hasattr(o.ref, 'q'): q = (',q=' + str(o.ref.q))
                if hasattr(o.ref, 'vol'): vol = (',vol=' + str(o.ref.vol))
                vol = (',vol=' + str(o.ref.vol)) if getattr(o.ref, 'vol', 0)!=0 else ''
            if hasattr(o, 'rf'): rf = (',rf=' + str(o.rf))
            if hasattr(o, 'r'): r = (',r=' + str(o.r))
            s += S0 + vol + r + q + rf

        if complexity in (2, 3,):
            from yaml import dump
            s = dump(o, default_flow_style=False).replace('!!python/object:__main__.','')
            if complexity == 2:
                s = s.replace(',',', ').replace('\n', ',').replace(': ', ':').replace('  ',' ')
        return s


class OptionValuation(OptionSeries):
    """ Adds interest rates and some methods shared by subclasses.

    The class inherits from a simpler class that describes an option.

    """
    def __init__(self, r=.05, rf=0, *args, **kwargs):
        """ Constructor simply saves all identified arguments and passes others to the base (parent) class, OptionSeries.

        It also calculates net_r, the rate used in computing growth factor a (p.452) for options with dividends and foreign risk free rates.

        :param r:  risk free rate. Required, unless clone object supplies it (see OptionSeries constructor). number in (0,1) interval
        :type r:   float
        :param rf: foreign risk free rate. Similar to r.
        :type rf: float
        :param args: arguments to be passed to base class constructor.
        :type args: see base class for types of its arguments
        :param kwargs: keyword arguments to be passed to base class constructor.
        :type kwargs: see base class for types of its arguments
        :return:   __init__() method always implicitly returns self, i.e. a reference to this object
        :rtype:    __main__.OptionValuation
        """
        self.r, self.rf = r, rf
        super().__init__(*args, **kwargs)  # pass remaining arguments to base (parent) class
        self.net_r = r - self.ref.q - rf  # calculate RFR net of yield and foreign RFR

    def LT_params(self, nsteps=2):
        """ Calculates a collection of specs/parameters needed for lattice tree pricing.

        Parameters returned:
            dt: time interval between consequtive two time steps
            u: stock price up move factor
            d: stock price down move factor
            a: growth factor, p.452
            p: probability of up move over one time interval dt
            df_T: discount factor over full time interval dt, i.e. per life of an option
            df_dt: discount factor over one time interval dt, i.e. per step

        :param nsteps: number of steps in a tree, positive number. Required.
        :type nsteps:  int
        :return:       LT specs
        :rtype:         dict

        :Example:

        >>> OptionValuation(ref=Stock(S0=42, vol=.2), right='call', K=40, T=.5, r=.1).LT_params(2)
        {'a': 1.0253151205244289,
         'd': 0.9048374180359595,
         'df_T': 0.951229424500714,
         'df_dt': 0.9753099120283326,
         'dt': 0.25,
         'p': 0.60138570166548,
         'u': 1.1051709180756477}
         >>> American(ref=Stock(S0=50, vol=.3), right='put', K=52, T=2, r=.05, desc={'note':'$7.42840, Hull p.288'}).LT_params(3)
        {'a': 1.033895113513574,
         'd': 0.7827444773247475,
         'df_T': 0.9048374180359595,
         'df_dt': 0.9672161004820059,
         'dt': 0.6666666666666666,
         'p': 0.5075681589595774,
         'u': 1.2775561233185384}
        """
        assert isinstance(nsteps, int), 'nsteps must be an integer, >2'
        from math import exp, sqrt
        par = {'dt': self.T / nsteps}
        par['u'] = exp(self.ref.vol * sqrt(par['dt']))
        par['d'] = 1 / par['u']
        par['a'] = exp(self.net_r * par['dt'])   # growth factor, p.452
        par['p'] = (par['a'] - par['d']) / (par['u'] - par['d'])
        par['df_T'] = exp(-self.r * self.T) #exp(-self.r * self.T)
        par['df_dt'] = exp(-self.r * par['dt'])
        # par['nsteps'] = (nsteps,)
        return par

    @property
    def BS_params(self):
        """ Calculates a collection of specs/parameters needed for BSM pricing.

        :return: collection of components of BSM formula, i.e. d1, d2, N(d1), N(d2)
        :rtype:  dict

        :Example:

        >>> OptionValuation(ref=Stock(S0=42, vol=.2), right='call', K=40, T=.5, r=.1).BS_params
        {'Nd1': 0.77913129094266897,
         'Nd2': 0.73494603684590853,
         'd1': 0.7692626281060315,
         'd2': 0.627841271868722}
         >>> American(ref=Stock(S0=50, vol=.3), right='put', K=52, T=2, r=.05, desc={'note':'$7.42840, Hull p.288'}).BS_params
         {'Nd1': 0.63885135045054053,
         'Nd2': 0.47254500437809299,
         'd1': 0.3553901873059548,
         'd2': -0.06887388140597372}
        """
        from math import log, sqrt
        from scipy.stats import norm
        d1 = ((log(self.ref.S0 / self.K) + (self.r + 0.5 * self.ref.vol ** 2) * self.T) / (self.ref.vol * sqrt(self.T)))
        d2 = d1 - self.ref.vol * sqrt(self.T)
        return {'d1': d1, 'd2': d2, 'Nd1': norm.cdf(d1), 'Nd2': norm.cdf(d2)}

    def pxLT(self, nsteps=2, return_tree=False):
        """ Calls _pxLT() method (defined differently by each class) to price this option.

        This method does not do the valuation.
        It's purpose is to vectorize over nsteps argument only, so that this common feature is not redundantly computed by each child class.
        If nsteps is a tuple, then return_tree is disabled for performance.

        :param nsteps:  number of steps for a binomial tree computation. Or, a tuple of numbers of steps. Required.
        :type nsteps:  int|tuple
        :param return_tree: indicates whether lattice tree (tuple of tuples ordered by time steps) must be returned. Required.
        :type return_tree: bool
        :return:  either a single LT price or a tuple of them (if nsteps is a tuple).
        :rtype:   float|tuple of floats|tuple of tuples

        :Example:

        >>> a = American(ref=Stock(S0=50, vol=.3), right='put', K=52, T=2, r=.05, desc={'note':'$7.42840, Hull p.288'})
        >>> a.pxLT(2)  # price American option with a 2-step binomial(lattice) tree
        7.42840190270483
        >>> a.pxLT((2,20,200))  # price American option with LT model using 2, 20, and 200 steps (vectorized I/O)
        (7.42840190270483, 7.5113077715410839, 7.4772083289361388)
        >>> a.pxLT(2, return_tree=True)
        (((27.44058, 50.0, 91.10594), (24.55942, 2.0, 0.0)),    # stock and option values for step 2
        ((37.04091, 67.49294), (14.95909, 0.9327)),             # stock and option values for step 1
        ((50.0,), (7.4284,)))                                   # stock and option values for step 0 (now)
        """
        if Util.is_iterable(nsteps):
            return Util.demote((self._pxLT(nsteps=n, return_tree=False) for n in nsteps))
        else:
            return self._pxLT(nsteps=nsteps, return_tree=return_tree)

    def plot_px_convergence(self, nsteps_max=200, ax=None, vs=None):
        """ Plots convergence of an option price for different nsteps values.

        If vs object is provided, its plot is added, i.e. call vs.plot_px_convergence(...) to add a plot of the benchmark option.
        This is helpful to compare the convergence of LT price for European vs American options.
        BSM price (a constant line) is also plotted.
        If ax is not provided, create a new ax, then continue.

        :param nsteps_max: sets the range of nsteps, so that the LT price can be computed for each time step.
            I.e. this is the maximum range of the x-axis on the resulting plot. pxLT is called with range(1, nsteps_max).
            Required. Positive integer.
        :type nsteps_max: int
        :param ax:  Optional plot object on which to plot the data.
        :type ax:   matplotlib.axes._subplots.AxesSubplot
        :param vs:  any child of OptionValuation, i.e. European, American,...
        :type vs:   object
        :return:    plot the price convergence.
        :rtype:     None

        :Example:

        >>> a = American(ref=Stock(S0=50, vol=.3), right='put', K=52, T=2, r=.05, desc={'note':'$7.42840, Hull p.288'})
        >>> e = European(clone=a)
        >>> a.plot_px_convergence(nsteps_max=200, vs=e)
        """
        # http://stackoverflow.com/questions/510972/getting-the-class-name-of-an-instance-in-python
        import matplotlib.pyplot as plt
        from pandas import DataFrame, Series

        if ax is None: fig, ax = plt.subplots()  # (nrows=1, ncols=1)
        DataFrame({'LT price for ' + self.spec2str(1): (self.pxLT(range(1, nsteps_max))),
                   'BS price for ' + self.spec2str(1): self.pxBS})\
            .plot(ax=ax, grid=True, title='Option price convergence with number of steps')

        if vs is not None: vs.plot_px_convergence(nsteps_max=nsteps_max, ax=ax)
        plt.tight_layout()
        plt.show()


class European(OptionValuation):
    """ European option class.
    Inherits all methods and properties of OptionValuation class.
    """
    @property
    def pxBS(self):
        """ Option valuation via BSM.

        Use BS_params method to draw computed parameters.
        They are also used by other exotic options.
        It's basically a one-liner.

        :return: price of a put or call European option
        :rtype: float
        """
        from math import exp
        _ = self.BS_params
        c = (self.ref.S0 * _['Nd1'] - self.K * exp(-self.r * self.T) * _['Nd2'])
        return c if self.right == 'call' else c - self.ref.S0 + exp(-self.r * self.T) * self.K

    def _pxLT(self, nsteps=3, return_tree=False):
        """ Option valuation via binomial (lattice) tree

        This method is not called directly. Instead, OptionValuation calls it via (vectorized) method pxLT()
        See Ch. 13 for numerous examples and theory.

        :param nsteps: number of time steps in the tree
        :type nsteps: int
        :param return_tree: indicates whether a full tree needs to be returned
        :type return_tree: bool
        :return: option price or a chronological tree of stock and option prices
        :rtype:  float|tuple of tuples

        :Example:

        >>> a = American(ref=Stock(S0=50, vol=.3), right='put', K=52, T=2, r=.05, desc={'note':'$7.42840, Hull p.288'})
        >>> a.pxLT(2)
        7.42840190270483
        >>> a.pxLT((2,20,200))
        (7.42840190270483, 7.5113077715410839, 7.4772083289361388)
        >>> a.pxLT(2, return_tree=True)
        (((27.44058, 50.0, 91.10594), (24.55942, 2.0, 0.0)),    # stock and option values for step 2
        ((37.04091, 67.49294), (14.95909, 0.9327)),             # stock and option values for step 1
        ((50.0,), (7.4284,)))                                   # stock and option values for step 0 (now)
        """
        # http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1341181
        # def pxLT_(nsteps):
        from numpy import cumsum, log, arange, insert, exp, sqrt, sum, maximum, vectorize

        _ = self.LT_params(nsteps)
        S = self.ref.S0 * _['d'] ** arange(nsteps, -1, -1) * _['u'] ** arange(0, nsteps + 1)
        O = maximum(self.signCP * (S - self.K), 0)          # terminal option payouts
        tree = ((S, O),)

        if return_tree:
            for i in range(nsteps, 0, -1):
                O = _['df_dt'] * ((1 - _['p']) * O[:i] + (_['p']) * O[1:])  #prior option prices (@time step=i-1)
                S = _['d'] * S[1:i + 1]                   # prior stock prices (@time step=i-1)
                tree = tree + ((S, O),)
            out = Util.round(tree, to_tuple=True)
        else:
            csl = insert(cumsum(log(arange(nsteps) + 1)), 0, 0)         # logs avoid overflow & truncation
            tmp = csl[nsteps] - csl - csl[::-1] + log(_['p']) * arange(nsteps + 1) + log(1 - _['p']) * arange(nsteps+1)[::-1]
            out = (_['df_T'] * sum(exp(tmp) * tuple(O)))
        return out


class American(OptionValuation):
    def _pxLT(self, nsteps, return_tree=False):
        """  Computes option price via binomial (lattice) tree.

        This method is not called directly. Instead, OptionValuation calls it via (vectorized) method pxLT()

        :param nsteps: number of time steps for which to build a tree
        :type nsteps:  int
        :param return_tree: indicates whether to return the full tree with stock and option prices.
        :type return_tree: bool
        :return:  option price, if return_tree is False, or a full tree, if return_tree is True.
        :rtype:  float | tuple of tuples
        """
        from numpy import arange, maximum

        _ = self.LT_params(nsteps)
        S = self.ref.S0 * _['d'] ** arange(nsteps, -1, -1) * _['u'] ** arange(0, nsteps + 1) # terminal stock prices
        O = maximum(self.signCP * (S - self.K), 0)          # terminal option payouts
        tree = ((S, O),)

        for i in range(nsteps, 0, -1):
            O = _['df_dt'] * ((1 - _['p']) * O[:i] + (_['p']) * O[1:])  #prior option prices (@time step=i-1)
            S = _['d'] * S[1:i + 1]                   # prior stock prices (@time step=i-1)
            Payout = maximum(self.signCP * (S - self.K), 0)   # payout at time step i-1 (moving backward in time)
            O = maximum(O, Payout)
            tree = tree + ((S, O),)

        return Util.round(tree, to_tuple=True) if return_tree else Util.demote(O)

    @property
    def pxBS(self):
        """
        :return: price for an American option estimated with BSM and other parameters.
        :rtype: None
        """
        from math import exp
        from numpy import linspace
        if self.right == 'call' and self.ref.q != 0:
            #Black's approximations outlined on pg. 346
            #Dividend paying stocks assume semi-annual payments
            if self.T > .5:
                dividend_val1 = sum([self.ref.q * self.ref.S0 * exp(-self.r * i) for i in linspace(.5, self.T - .5,
                                    self.T * 2 - .5)])
                dividend_val2 = sum([self.ref.q * self.ref.S0 * exp(-self.r * i) for i in linspace(.5, self.T - 1,
                                    self.T * 2 - 1)])
            else:
                dividend_val1 = 0
                dividend_val2 = 0
            first_val = European(ref=Stock(S0=self.ref.S0 - dividend_val1, vol=self.ref.vol, q=self.ref.q), right=self.right,
                                 K=self.K, r=self.r, T=self.T).pxBS
            second_val = European(ref=Stock(S0=self.ref.S0 - dividend_val2, vol=self.ref.vol, q=self.ref.q),
                                  right=self.right, K=self.K, r=self.r, T=self.T - .5).pxBS
            return max([first_val, second_val])
        elif self.right == 'call':
            #American call is worth the same as European call if there are no dividends
            return European(ref=Stock(S0=self.ref.S0, vol=self.ref.vol), right=self.right, K=self.K, r=self.r, T=self.T).pxBS
        elif self.ref.q != 0:
            # I wasn't able to find a good approximation for American Put BSM w/ dividends so I'm using 200 and 201
            # time step LT and taking the average. This is effectively the Antithetic Variable technique found on pg. 476 due
            # to the oscillating nature of binomial tree
            f_a = (American(ref=Stock(S0=self.ref.S0, vol=self.ref.vol, q=self.ref.q), right=self.right,
                            K=self.K, r=self.r, T=self.T).pxLT(200) + American(ref=Stock(S0=self.ref.S0, vol=self.ref.vol,
                                                                                         q=self.ref.q), right=self.right,
                                                                               K=self.K, r=self.r, T=self.T).pxLT(201)) / 2
            return f_a
        else:
            #Control Variate technique outlined on pg.463
            f_a = American(ref=Stock(S0=self.ref.S0, vol=self.ref.vol), right=self.right,
                           K=self.K, r=self.r, T=self.T).pxLT(100)
            f_bsm = European(ref=Stock(S0=self.ref.S0, vol=self.ref.vol), right=self.right,
                             K=self.K, r=self.r, T=self.T).pxBS
            f_e = European(ref=Stock(S0=self.ref.S0, vol=self.ref.vol), right=self.right,
                           K=self.K, r=self.r, T=self.T).pxLT(100)
            return f_a + (f_bsm - f_e)

#European(ref=Stock(S0=810, vol=.2, q=.02), right='call', K=800, T=.5, r=.05).pxLT(2)   # 53.39, p.291
#American(clone=European(ref=Stock(S0=42, vol=.2), right='call', K=40, T=.5, r=.05)).pxLT()  # 4.1571182276538945
#American(ref=Stock(S0=42, vol=.2), right='call', K=40, T=.5, r=.05).pxBS
#American(ref=Stock(S0=50, vol=.3), right='put', K=52, T=2, r=.05).pxLT(50)  # 7.6708887347472539
#American(ref=Stock(S0=50, vol=.3), right='put', K=52, T=2, r=.05).pxBS
#a = American(ref=Stock(S0=50, vol=.3, q=.02), right='put', K=52, T=2, r=.05)
#a.plot_px_convergence(nsteps_max=100)
print(European(ref=Stock(S0=40, vol=.28), right='call', T=.5, r=.05, K=50).pxBS)

