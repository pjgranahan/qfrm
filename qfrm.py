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
    # def __init__(self, S0=50, vol=.3, q=0, curr=None, tkr=None, desc=None):
    def __init__(self, S0=None, vol=None, q=0, curr=None, tkr=None, desc=None):
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
    # def __init__(self, ref=Stock(S0=50, vol=.3), right='put', K=52, T=2, clone=None, desc={}):
    def __init__(self, ref=None, right=None, K=None, T=None, clone=None, desc=None):
        """ Constructor.

        If clone object is supplied, its specs are used.

        :param ref: any suitable object of an underlying instrument (must have S0 & vol variables).
                Required, if clone = None.
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
        self.update(ref=ref, right=right, K=K, T=T, clone=clone, desc=desc)
        # if clone is None:
        #     self.ref, self.K, self.T, self.desc, self.right = ref, K, T, desc, right
        # else:
        #     self.clone = clone  # copy over all properties of option being cloned

    def update(self, **kwargs):
        """

        :param kwargs:
        :return:

        :Example:

        >>> o = OptionSeries(ref=Stock(S0=50, vol=.3), right='put', K=52, T=2).update(K=53)
        >>> vars(o)
        >>> o = OptionSeries(clone=o, K=54).update(right='call')
        >>> vars(o)

        """
        if 'clone' in kwargs:
            self.clone = kwargs['clone']
            del kwargs['clone']

        # self.__dict__.update(kwargs)
        # if kwargs is not None: [setattr(self, a, kwargs[a]) for a in kwargs]
        for k, v in kwargs.items():
            if v is not None: setattr(self, k, v)

        # for kwarg in kwargs:
        #     if kwarg != 'clone':
        #         self[kwarg] = kwargs[kwarg]

        # self.clone = clone  # copy over all properties of option being cloned
        # if ref is not None: self.ref = ref
        # if K is not None: self.K = K
        # self.ref, self.K, self.T, self.desc, self.right = ref, K, T, desc, right
        return self

    def get_right(self):
        """ Returns option's right as a string.
        :return: 'call', 'put', or 'other'
        :rtype: str
        """
        # return 'call' if self._signCP == 1 else 'put' if self._signCP == -1 else 'other'
        return self._right

    def set_right(self, right='put'):
        if right is not None:
            self._right = right.lower()
            self._signCP = 1 if self._right == 'call' else -1 if self._right == 'put' else 0  # 0 for other rights
        return self

    right = property(get_right, set_right, None, 'option\'s right (str): call or put')

    @property
    def signCP(self): return self._signCP

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
        s = str(o.K) + ' ' + str(o.T) + 'yr ' + self.style + ' ' + str(o.right)  # option series name

        if complexity == 1:
            rf_r = frf_r = q = vol = ''
            if hasattr(o, 'ref'):  # cif reference object is specified, read its parameters
                if hasattr(o.ref, 'S0'): S0 = (',S0=' + str(o.ref.S0))
                if hasattr(o.ref, 'q'): q = (',q=' + str(o.ref.q))
                if hasattr(o.ref, 'vol'): vol = (',vol=' + str(o.ref.vol))
                vol = (',vol=' + str(o.ref.vol)) if getattr(o.ref, 'vol', 0)!=0 else ''
            if hasattr(o, 'frf_r'): frf_r = (',frf_r=' + str(o.frf_r))
            if hasattr(o, 'rf_r'): rf_r = (',rf_r=' + str(o.rf_r))
            s += S0 + vol + rf_r + q + frf_r

        if complexity in (2, 3,):
            from yaml import dump
            s = dump(o, default_flow_style=False).replace('!!python/','').replace('object:__main__.','')
            if complexity == 2:
                s = s.replace(',',', ').replace('\n', ',').replace(': ', ':').replace('  ',' ')
        return s

    def __repr__(self):
        """ Called by the repr() built-in function to compute the “official” string representation of an object.

        :return: full list of object properties
        :rtype: str

        .. seealso::
            http://stackoverflow.com/questions/1436703/difference-between-str-and-repr-in-python
            https://docs.python.org/2/reference/datamodel.html#object.__repr__
            http://stackoverflow.com/questions/1984162/purpose-of-pythons-repr

        """
        return self.spec2str(complexity=2)

    def __str__(self):
        """ Called by str(object) and the built-in functions format() and print()
        to compute the “informal” or nicely printable string representation of an object.

        :return: full list of object properties
        :rtype: str
        """
        return self.spec2str(complexity=3)

    @property
    def style(self):
        """
        :return: option style
        :rtype: str
        """
        return type(self).__name__

    @property
    def clone(self):  return self

    @clone.setter
    def clone(self, clone=None):
        """

        :param clone:
        :return:

        :Example:

        >>> o = OptionSeries(); o.right='call'
        >>> OptionSeries(clone=o).right
        >>> OptionSeries(clone=OptionSeries().set_right('call')).right

        """
        # copy specs from supplied object
        if clone is not None:
            [setattr(self, v, getattr(clone, v)) for v in vars(clone)]
            # self.__dict__.update(vars(clone))   # don't use. fails to call set_right()



class OptionValuation(OptionSeries):
    """ Adds interest rates and some methods shared by subclasses.

    The class inherits from a simpler class that describes an option.


    """
    def __init__(self, rf_r=None, frf_r=0, seed0=None, *args, **kwargs):
        """ Constructor simply saves all identified arguments and passes others to the base (parent) class, OptionSeries.

        It also calculates net_r, the rate used in computing growth factor a (p.452) for options with dividends and foreign risk free rates.

        :param rf_r:  risk free rate. Required, unless clone object supplies it (see OptionSeries constructor). number in (0,1) interval
        :type rf_r:   float
        :param frf_r: foreign risk free rate.
        :type frf_r: float
        :param seed0: None or positive integer to seed random number generator (RNG).
        :type seed0: int, None
        :param args: arguments to be passed to base class constructor.
        :type args: see base class for types of its arguments
        :param kwargs: keyword arguments to be passed to base class constructor.
        :type kwargs: see base class for types of its arguments
        :return:   __init__() method always implicitly returns self, i.e. a reference to this object
        :rtype:    __main__.OptionValuation

        :Example:

        >>> o = OptionValuation(frf_r=.01); o.net_r
        >>> o.frf_r = .02; o.net_r

        """
        self.rf_r, self.frf_r, self.seed0 = rf_r, frf_r, seed0
        super().__init__(*args, **kwargs)  # pass remaining arguments to base (parent) class

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
        par['df_T'] = exp(-self.rf_r * self.T)
        par['df_dt'] = exp(-self.rf_r * par['dt'])
        return par

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
                   'BS price for ' + self.spec2str(1): (self.pxBS)})\
            .plot(ax=ax, grid=True, title='Option price convergence with number of steps')

        if vs is not None: vs.plot_px_convergence(nsteps_max=nsteps_max, ax=ax)
        plt.tight_layout()
        plt.show()

    @property
    def net_r(self):
        """
        :return: net value of interest rate used to price this option
        :rtype: float

        :Example:

        >>> o = OptionValuation(rf_r=0.05); vars(o)
        >>> vars(o.update(rf_r=0.04))
        >>> vars(o.update(ref=Stock(q=0.01)))
        >>> o.net_r

        """
        try: q = 0 if self.ref.q is None else self.ref.q
        except: q = 0

        frf_r = 0 if self.frf_r is None else self.frf_r
        rf_r = 0 if self.rf_r is None else self.rf_r

        return rf_r - q - frf_r   # calculate RFR net of yield and foreign RFR
