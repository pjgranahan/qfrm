import warnings

from pricespec import PriceSpec

from qfrm.specprinter import SpecPrinter


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
        self : option

        """
        self.px_spec = PriceSpec(px=None)
        return self