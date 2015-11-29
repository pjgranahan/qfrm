import yaml
import numbers
import numpy as np
import re
import functools


class Util():
    """ A collection of utility functions, most of which are static methods,
    i.e. can be called as ``Util.is_iterable()``.

    FYI: Decorator ``@staticmethod`` allows use of functions without initializing an object
    Ex. we can use ``Util.demote(x)`` instead of ``Util().demote(x)``. It's faster.

    :Authors:
        Oleg Melnikov <xisreal@gmail.com>
    """
    @staticmethod
    def is_iterable(x):
        """ Checks if ``x`` is iterable.

        Parameters
        ----------
        x : object
            any object

        Returns
        -------
        bool
            ``True`` if ``x`` is iterable, ``False`` otherwise

        Exmaples
        --------

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
        """ Checks if ``x`` is numeric (``float``, ``int``, ``complex``, ...)

        Parameters
        ----------
        x : object
            any object

        Returns
        -------
        bool
            ``True``, if ``x`` is numeric; ``False`` otherwise.

        """
        return isinstance(x, numbers.Number)

    @staticmethod
    def are_numbers(x):
        """ Checks if x is an iterable of numbers.

        Parameters
        ----------
        x : array_like
            any object (value, iterable,...) that need to be verified as being numeric or not

        Returns
        -------
        bool
            True if x is iterable, False otherwise

        Examples
        --------

        >>> Util.are_numbers(5)
        False

        >>> Util.are_numbers([1,'blah',3.])
        False

        >>> Util.are_numbers([1,'2',3.])
        False

        >>> Util.are_numbers((1, 2., 3. + 4j, 5.4321))
        True

        >>> Util.are_numbers({1, 2., 3. + 4j, 5.4321})
        True

        """
        try:
            return all(Util.is_number(n) for n in x)
        except TypeError:
            return False

    @staticmethod
    def are_bins(x):
        return Util.are_non_negative(x) and Util.is_monotonic(x)

    @staticmethod
    def cpn2cf(cpn=6, freq=2, ttm=2.1):
        """ Converts regular coupon payment specification to a series of cash flows indexed by time to cash flow (ttcf).

        Parameters
        ----------
        cpn : float, int
            annual coupon payment in $
        freq : float, int
            payment frequency, per anum
        ttm : float, int
            time to maturity of a bond, in years

        Returns
        -------
        dict('ttcf'=tuple, 'cf'=tuple)
            dictionary of cash flows (tuple) and their respective times to cf (tuple)


        Examples
        --------

        >>> # convert $6 semiannula (SA) coupon bond payments to indexed cash flows
        >>> print(Util.cpn2cf(6,2,2.1))  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        {...}

        """


        if cpn == 0: freq = 1  # set frequency to annual for zero coupon instruments
        period = 1./freq            # time (in year units) period between coupon payments
        end = ttm + period / 2.          # small offset (anything less than a period) to assure expiry is included
        start = period if (ttm % period) == 0 else ttm % period  # time length from now till next cpn, yrs
        c = float(cpn)/freq   # coupon payment per period, $

        ttcf = tuple((float(x) for x in np.arange(start, end, period)))        # times to cash flows (tuple of floats)
        cf = tuple(map(lambda i: c if i < (len(ttcf) - 1) else c + 100, range(len(ttcf)))) # cash flows(tuple of floats)
        return {'ttcf': ttcf, 'cf': cf}

    @staticmethod
    def demote(x):
        """ Attempts to simplify ``x`` to a ``tuple`` (if x is a more complex data type) or just singleton.
        Basically, demotes to a simpler object, if possible.

        Parameters
        ----------
        x : object
            any object

        Returns
        -------
        object
            original object or tuple or value of a singleton
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
    def round(x, prec=5, to_tuple=False):
        """ Recursively rounds an iterable to the desired precision.

        Parameters
        ----------
        x : iterable
            iterable of numbers
        prec : int
            number of decimal (incl. period) to keep
        to_tuple: bool
            indicates whether to keep original data type or convert output to tuple

        Returns
        -------
        Tuple[float,...,float]
            tuple of rounded numbers

        Examples
        --------

        >>> x = (1, 1/3, 1/7,[1/11, 1/13, {1/19, 1/29}]);  import numpy as np; a = np.array(x)
        >>> Util.round(x)
        (1, 0.33333, 0.14286, [0.09091, 0.07692, {0.03448, 0.05263}])
        >>> Util.round(x, to_tuple=True)
        (1, 0.33333, 0.14286, (0.09091, 0.07692, (0.03448, 0.05263)))

        """
        if to_tuple: x = Util.to_tuple(x)
        try:
            return round(x, prec)
        except TypeError:
            return type(x)(Util.round(y, prec) for y in x)

    @staticmethod
    def to_tuple(a, leaf_as_float=False):
        """ Recursively converts a iterable (and arrays) to ``tuple``.

        Parameters
        ----------
        a : array, iterable
            variable to be converted to tuple

        Returns
        -------
        tuple

        Examples
        --------
        >>> import numpy as np; x = (1, 1/3, 1/7,[1/11, 1/13, {1/19, 1/29}]); a = np.array(x)
        >>> Util.to_tuple(x)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        (1, 0.333..., 0.142857142..., (0.0909...,  0.076923076...,  (0.034482758..., 0.052631578...)))

        >>> Util.to_tuple(a)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        (1, 0.333..., 0.142857142..., (0.0909...,  0.076923076...,  (0.034482758..., 0.052631578...)))

        Notes
        --------
        http://stackoverflow.com/questions/10016352/convert-numpy-array-to-tuple
        """
        try:  return tuple((Util.to_tuple(i)) for i in a)
        except TypeError: return float(a) if leaf_as_float else a


class SpecPrinter:
    r""" Helper class for printing class's internal variables.

    This is a base class that is inherited by any child class needs to display its specifications (class variables).

    Examples
    --------
    >>> class A(SpecPrinter):
    ...     def __init__(self, **kwargs):
    ...        self.a=[1/17, 1/19, 1/23]; self.b=None; self.c = {'a':1/7,'b':1/13,'c':'bla'}
    ...        super().__init__(**kwargs) #(print_precision=print_precision)
    >>> A()  # print structure of A(); same as print(str(A())), print(A()), print(repr(A()))
    A
    a:
    - 0.058823529
    - 0.052631579
    - 0.043478261
    c:
      a: 0.142857143
      b: 0.076923077
      c: bla

    >>> A(print_precision=3).full_spec(print_as_line=True)
    'A{a:[0.06, 0.05, 0.04], c:{a:0.14, b:0.08, c:bla}}'

    >>> str(A())  # doctest: +ELLIPSIS
    'A\na:\n- 0.058823529\n- 0.052631579\n- 0.043478261\nc:\n  a: 0.142857143\n  b: 0.076923077\n  c: bla'


    :Authors:
        Oleg Melnikov <xisreal@gmail.com>
    """
    print_precision = 9

    def __init__(self, print_precision=9):
        """ Constructor

        Sets rounding precision for display of floating numbers

        Parameters
        ----------
        print_precision : int, optional
            Sets number of decimal digits to which printed output is rounded;
            used with whole object print out and with print out of some calculated values (``px``, ...)
            Default 9 digits. If set to ``None``, machine precision is used.
        """
        # if 'print_precision' in kwargs:
        #     if kwargs['print_precision'] != 9:
        #         self._print_precision = kwargs['print_precision']
        #     else:
        #         try: del self.print_precision  # do not store default value
        #         except: pass
        SpecPrinter.print_precision = print_precision

    # @property
    # def print_precision(self):
    #     """ Returns user-saved printing precision or default (9 digits)
    #
    #     Returns
    #     -------
    #     int :
    #         printing precision
    #     """
    #     try: return self._print_precision
    #     except: return 9

    def full_spec(self, print_as_line=True):
        r""" Returns a formatted string containing all variables of this class (recursively)

        Parameters
        ----------
        print_as_line : bool
            If ``True``, print key:value pairs are separated by ``,``
            If ``False``, --- by ``\n``
        print_precision : {None, int}, optional
            Specifies desired floating number precision for screen-printed values (prices, etc).
            Assists with doctesting due to rounding errors near digits in 10^-12 placements
            If value is None, then precision is ignored and default machine precision is used.
            See `round() <https://docs.python.org/3.5/library/functions.html#round>`_
        Returns
        -------
        str
            Formatted string with option specifications


        Notes
        -----
        - `PyYAML documenation <http://pyyaml.org/wiki/PyYAMLDocumentation>`_
        - `YAML dump options <https://dpinte.wordpress.com/2008/10/31/pyaml-dump-option/>`_
        - `Overloading examples <http://pyyaml.org/browser/pyyaml/trunk/lib/yaml/representer.py#L187>`_
        - `RegEx demo <https://regex101.com/r/dZ9iI8/1>`_

        """

        def float_representer(dumper, value):
            text = str(value if SpecPrinter.print_precision is None else round(value, SpecPrinter.print_precision))
            return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)
        yaml.add_representer(float, float_representer)

        # each yaml dump has trailing '\n', which we identify and remove
        s = yaml.dump(self, default_flow_style=print_as_line, width=1000)  # , explicit_end=True

        s = re.sub(r'\w+: null', '', s)  # RegEx removes null keys. Demo: https://regex101.com/r/dZ9iI8/1
        # s = re.sub(r'\b\w+:\s+null(|$)', '', s).strip() # RegEx removes null keys.
        # s = s.replace('\n...\n','')   # trim trailing new line (explicit end)
        s = re.sub(u'(?imu)^\s*\n', u'', s)  # removes lines of spaces

        s = s.replace('!!python/object:','').replace('!!python/tuple','')
        s = s.replace('__main__.','').replace(type(self).__name__ + '.','').replace('SpecPrinter.', '')
        s = s.replace('OptionValuation.','').replace('OptionSeries.','').replace('Util.', '')

        s = s.replace(' {', '{')
        s = re.sub(re.compile(r'(,\s){2,}'), ', ', s)  # ", , , , , ... "   |->  ", "
        # s = s.replace('{,  ','{').replace('{, ','{')

        if print_as_line:
            s = s.replace(',', ', ').replace(': ', ':')  #.replace('  ', ' ')
            s = re.sub(r'(\s){2,}', ' ', s)    # replace successive spaces with one instance

        # s = re.sub(r'(,\s){2,}', ', ', s)  # replace successive instances of ', ' with one instance
        # s = re.sub(r'(\n\s){2,}', '\n ', s)    # replace successive spaces with one instance
        # s = re.sub(r'(\n\s\s){2,}', '\n\s\s', s)    # replace successive spaces with one instance
        # s = functools.reduce( (lambda x, y: x + '\n ' + y if y else x), s.split('\n '))
        # s = functools.reduce( (lambda x, y: x + '\n  ' + y if y else x), s.split('\n  '))

        # s = yaml.dump(self, default_flow_style=not new_line).replace('!!python/object:','').replace('!!python/tuple','')
        # s = s.replace('__main__.','').replace(type(self).__name__ + '.','').replace('null','-')
        # s = s.replace('__main__.','').replace('OptionValuation.','').replace('OptionSeries.','').replace('null','-')
        # s = s.replace('Util.', '').replace(', ,',', ').replace('{,  ','{').replace('{, ','{')
        # if not new_line:
            # s = s.replace(',', ', ').replace('\n', ',').replace(': ', ':').replace('  ', ' ')

        return s.strip()

    def __repr__(self):
        return self.full_spec(print_as_line=False)

    def __str__(self):
        return self.full_spec(print_as_line=False)

    def print_value(self, v):
        if Util.is_number(v):
            return v if SpecPrinter.print_precision is None else round(v, SpecPrinter.print_precision)

# from qfrm import *
# print(OptionSeries(ref=Stock(S0=50, vol=1/3), K=51, right='call').full_spec(print_as_line=False))
# OptionValuation(ref=Stock(S0=50, vol=1/7), K=51, right='call',print_precision=4).full_spec(print_as_line=True)


# import yaml
# class A:
#     def __init__(self):
#         self.abc = 1
#         self.hidden = 100
#         self.xyz = [1,2,3]
#
#     def __repr__(self):
#         return yaml.dump( vars(self).copy().pop('hidden'))
#
# A()