import re
import yaml
import numbers
import functools
import math
import warnings
import numpy as np
import operator as op


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
    def promote(x, length=1):
        """ Promotes a number or singleton to tuple of desired length.

        If ``x`` is a tuple of length > 1, it's not replicated and retains its size

        Parameters
        ----------
        x : number, tuple

        size : int
            desired length of replicated number

        Returns
        -------
        Tuple
            tuple made of ``x`` of desired size

        Examples
        --------
        >>> Util.promote(1)
        (1,)
        >>> Util.promote(1, length=5)
        (1, 1, 1, 1, 1)
        >>> Util.promote((1, 2, 3,))
        (1, 2, 3)
        >>> Util.promote([1, 2, 3])
        (1, 2, 3)
        >>> Util.promote({1, 2, 3})
        (1, 2, 3)
        >>> from numpy import array; Util.promote(array([1,2,3]))
        (1, 2, 3)
        """
        if Util.is_number(x): x = [x]
        if not isinstance(x, list): x = [i for i in x]
        if len(x) == 1 and length > 1: x = (x * length)
        return tuple(x)

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

    @staticmethod
    def norm_cdf(x, mu=0, sigma=1):
        """

        Parameters
        ----------
        x :
        mu : float
            distribution's mean
        sigma : float
            distribution's standard deviation

        Returns
        -------
        float
            pdf or cdf value, depending on input flag ``f``

        Notes
        -----
        http://stackoverflow.com/questions/809362/how-to-calculate-cumulative-normal-distribution-in-python

        Examples
        --------
        Compares total absolute error for 100 values

        >>> from scipy.stats import norm
        >>> sum( [abs(Util.norm_cdf(x) - norm.cdf(x)) for x in range(100)])
        """

        y = 0.5 * (1 - math.erf(-(x - mu)/(sigma * math.sqrt(2.0))))
        if y > 1: y = 1
        return y

    @staticmethod
    def norm_pdf(x, mu=0, sigma=1):
        u = (x - mu)/abs(sigma)
        y = (1/(math.sqrt(2 * math.pi) * abs(sigma))) * math.exp(-u*u/2)
        return y

    # @staticmethod

    @staticmethod
    def maximum(x, y):
        """ Similar to ``numpy.maximum``.

        The only difference is that maximum does not handle comparison with NaN values.

        Parameters
        ----------
        x, y : number, iterable
            values to compare. They can be numbers or iterables.
            If both of length > 1, then lengths must match.

        Examples
        --------
        >>> import numpy; x = numpy.random.random(10)
        >>> import random; y = [random.random() for i in range(len(x))]  # Python standard library

        Compare max from both functions

        >>> Util.maximum(x, y) - numpy.maximum(x, y)
        >>> Util.maximum(x, y[0]) - numpy.maximum(x, y[0])
        >>> Util.maximum(x[0], y) - numpy.maximum(x[0], y)
        >>> Util.maximum(x, float(y[0])) - numpy.maximum(x, float(y[0]))

        Compare timing with tuples

        >>> import timeit; from numpy import maximum, random
        >>> x = random.random(100); y = random.random(len(x))
        >>> (timeit.timeit('Util.maximum(x, y)', 'from __main__ import Util, x, y', number=100),
        ... timeit.timeit('maximum(x, y)', 'from __main__ import maximum, x, y', number=100))

        Compare timing with arrays

        >>> import timeit; from random import random; from numpy import maximum
        >>> x = [random() for i in range(100)]; y = [random() for i in range(len(x))]
        >>> (timeit.timeit('Util.maximum(x, y)', 'from __main__ import Util, x, y', number=100),
        ... timeit.timeit('maximum(x, y)', 'from __main__ import maximum, x, y', number=100))

        """
        x = x if Util.is_iterable(x) else [x]
        y = y if Util.is_iterable(y) else [y]
        if len(y) == 1: y = y * len(x)
        if len(x) == 1: x = x * len(y)

        assert len(x) == len(y), 'Assert: input lengths are equal or one of them is of length 1 (or scalar)'
        return Util.demote((max(i) for i in tuple(zip(x,y))))

    @staticmethod
    def minimum(x, y):
        """ Similar to ``numpy.maximum``.

        The only difference is that maximum does not handle comparison with NaN values.

        Parameters
        ----------
        x, y : number, iterable
            values to compare. They can be numbers or iterables.
            If both of length > 1, then lengths must match.

        Examples
        --------
        >>> import random  # Python standard library
        >>> import numpy
        >>> x = numpy.random.random(10)   # generate random numbers with np
        >>> y = [random.random() for i in range(len(x))]

        Compare max from both functions

        >>> Util.minimum(x, y) - numpy.minimum(x, y)
        >>> Util.minimum(x, y[0]) - numpy.minimum(x, y[0])
        >>> Util.minimum(x[0], y) - numpy.minimum(x[0], y)
        >>> Util.minimum(x, float(y[0])) - numpy.minimum(x, float(y[0]))
        """
        x = x if Util.is_iterable(x) else [x]
        y = y if Util.is_iterable(y) else [y]
        if len(y) == 1: y = y * len(x)
        if len(x) == 1: x = x * len(y)

        assert len(x) == len(y), 'Assert: input lengths are equal or one of them is of length 1 (or scalar)'
        return Util.demote((min(i) for i in tuple(zip(x,y))))

    @staticmethod
    def arange(start=None, stop=None, step=1, incl_start=True, incl_stop=False):
        """ Similar to ``numpy.arange()``.

        In contrast to ``numpy.arange()`` this function does not allow ``dtype`` specification.
        See `numpy.arange docs <http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.arange.html>`_

        Parameters
        ----------
        start : number, optional
            starting value of the interval. default is 0
        stop : number
            end of interval
        step : number, optional
            step size. default is 1. If step is specified, start must be given.
            Must be negative, if ``start > stop``
        incl_stop : bool
            If ``False``, starting number is excluded.
        incl_stop : bool
            If ``False`` stopping number is excluded (like ``numpy.arange`` does).
            If ``True`` stopping number is included in sequence.

        Returns
        -------
        tuple[float,...]
            tuple of floats (or int) values from ``start`` to ``stop`` incremented by ``step``.

        Examples
        --------
        >>> from numpy import arange
        >>> (Util.arange(1, 5, 1), arange(1, 5, 1))
        ((1, 2, 3, 4), array([1, 2, 3, 4]))
        >>> (Util.arange(1, 5), arange(1, 5))
        ((1, 2, 3, 4), array([1, 2, 3, 4]))
        >>> (Util.arange(5), arange(5))
        ((0, 1, 2, 3, 4), array([0, 1, 2, 3, 4]))
        >>> (Util.arange(5, 1, -1), arange(5, 1, -1))
        ((5, 4, 3, 2), array([5, 4, 3, 2]))
        >>> (Util.arange(5, 1, 1), arange(5, 1, 1))
        ((), array([], dtype=int32))

        >>> from numpy import arange
        >>> start = 2; stop=5.3; step=.8
        >>> seq = Util.arange(start, stop, step);
        >>> npseq = arange(start, stop, step);
        >>> [i - j for i,j in zip(seq, seq)]   # differences between sequence elements
        [0, 0.0, 0.0, 0.0, 0.0]

        The difference below shows an increasing rounding error,
        (but this can also be raising from `numpy`)

        >>> start = 21.1; stop=10.3; step=-5/3
        >>> [i - j for i,j in zip(arange(start, stop, step), tuple(Util.arange(start, stop, step)))]
        [0.0, 0.0, 0.0, 0.0, -1.7763568394002505e-15, -3.5527136788005009e-15, -5.3290705182007514e-15]
        """

        if step is None: step = 1
        if stop is None: stop = start; start = 0
        if start is None: start = 0

        if (start < stop and step < 0) or (start > stop and step > 0):
            seq = ()
        else:
            seq, next = (start,), start + step
            while (next <= stop and start < stop) or (next >= stop and start > stop):
                seq += (next,); next = seq[len(seq) - 1] + step

            if not incl_stop and (seq[len(seq)-1] == stop):
                seq = seq[0:(len(seq)-1)]

            if not incl_start and (len(seq) > 0):
                seq = seq[1:(len(seq))]
        return seq

    @staticmethod
    def log(x, as_tuple=True):
        """ Imitates ``numpy.log``

        Examples
        ----------
        >>> Util.log(Util.arange(4, incl_start=False))
        (0.0, 0.6931471805599453, 1.0986122886681098)
        """
        return Util.map(math.log, x, as_tuple=as_tuple)
        # if Util.is_iterable(x): y = (math.log(i) for i in x)
        # else: y = math.log(x)
        # return tuple(y) if as_tuple else y

    @staticmethod
    def exp(x, as_tuple=True):
        """ Exp function that works on a number or on iterable.
        Examples
        --------
        >>> Util.exp(Util.arange(3))
        (1.0, 2.718281828459045, 7.38905609893065)

        """
        return Util.map(math.exp, x, as_tuple=as_tuple)
        # if Util.is_iterable(x): y = (math.exp(i) for i in x)
        # else: y = math.exp(x)
        # return tuple(y) if as_tuple else y

    @staticmethod
    def cumsum(x, as_tuple=True):
        """ Imitates ``numpy.cumsum``

        Parameters
        ----------
        x : iterable
            numbers to cumulatively summate
        as_tuple : bool
            indicates whether return type is tuple or generator

        Examples
        --------
        >>> Util.cumsum([1,2,3,4,5])
        (1, 3, 6, 10, 15)
        >>> from numpy import cumsum
        >>> cumsum([1,2,3,4,5])
        array([ 1,  3,  6, 10, 15], dtype=int32)

        """
        if Util.is_iterable(x):
            def cumsum_(it):
                total = 0
                for a in it:
                    total += a
                    yield total
            y = cumsum_(x)
            return tuple(y) if as_tuple else y
        else:
            return x

    @staticmethod
    def pow(x, y=1, as_tuple=True):
        """
        >>> Util.pow(2, 4)
        (16,)
        >>> Util.pow((2,), 4)
        (16,)
        >>> Util.pow([1,2,3,], 4)
        (1, 16, 81)
        """
        out = (i**y for i in Util.promote(x))
        return tuple(out) if as_tuple else out

    @staticmethod
    def sqrt(x, as_tuple=True):
        """
        >>> Util.sqrt(3)
        (1.7320508075688772,)
        >>> Util.sqrt((1,2,3))
        (1.0, 1.4142135623730951, 1.7320508075688772)
        >>> Util.sqrt({1,2,3})
        (1.0, 1.4142135623730951, 1.7320508075688772)
        """
        return Util.pow(x, y=0.5, as_tuple=as_tuple)

    @staticmethod
    def map(fun, x, as_tuple=True):
        """
        >>> Util.map(math.log10, [1,2,3])
        (0.0, 0.3010299956639812, 0.47712125471966244)
        """
        out = map(fun, Util.promote(x))
        return tuple(out) if as_tuple else out

    @staticmethod
    def add(x, y, as_tuple=True):
        """ Adds iterables and/or scalars
        >>> Util.add(1, 2)
        (3,)
        >>> Util.add((1,2,3), 1)
        (2, 3, 4)
        >>> Util.add(1, [1,2,3])
        (2, 3, 4)
        >>> Util.add([1, 2, 3], (4, 5 ))
        (5, 7, 9)
        """
        if Util.is_number(y): y = (y,)
        x = Util.promote(x, len(y));
        y = Util.promote(y, length=len(x))
        assert len(x) == len(y), 'Assert that inputs are of the same length'
        out = (i + j for i, j in zip(x, y))
        return tuple(out) if as_tuple else out

    def sub(x, y, as_tuple=True):
        """ Subtracts y from x. They can be iterables or scalars.
        >>> Util.sub(1, 2)
        (3,)
        >>> Util.sub((1,2,3), 1)
        (2, 3, 4)
        >>> Util.sub(1, [1,2,3])
        (2, 3, 4)
        >>> Util.sub([1, 2, 3], (4, 5 ))
        (5, 7, 9)
        """
        if Util.is_number(y): y = (y,)
        x = Util.promote(x, len(y));
        y = Util.promote(y, length=len(x))
        assert len(x) == len(y), 'Assert that inputs are of the same length'
        out = (i - j for i, j in zip(x, y))
        return tuple(out) if as_tuple else out

    def mult(x, y, as_tuple=True):
        """Multiplies iterables and/or scalars
        """
        if Util.is_number(y): y = (y,)
        x = Util.promote(x, len(y));
        y = Util.promote(y, length=len(x))
        assert len(x) == len(y), 'Assert that inputs are of the same length'
        out = (i * j for i, j in zip(x, y))
        return tuple(out) if as_tuple else out


class SpecPrinter:
    r""" Helper class for printing class's internal variables.

    This is a base class that is inherited by any child class needs to display its specifications (class variables).

    Examples
    --------
    >>> class A(SpecPrinter):
    ...     def __init__(self, **kwargs):
    ...        self.a=[1/17, 1/19, 1/23]; self.b=None; self.c = {'a':1/7,'b':1/13,'c':'bla'}
    ...        super().__init__(**kwargs)
    >>> A()  # dumps variables of A(); same as print(str(A())), print(A()), print(repr(A()))
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
    'A{a:[0.059, 0.053, 0.043], c:{a:0.143, b:0.077, c:bla}}'

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

        def numpy_representer_str(dumper, data):
            astr = ', '.join(['%s']*data.shape[0])%tuple(data)
            return dumper.represent_scalar('!ndarray:', astr)

        def numpy_representer_seq(dumper, data):
            return dumper.represent_sequence('!ndarray:', data.tolist())

        yaml.add_representer(float, float_representer)
        yaml.add_representer(np.ndarray, numpy_representer_str)
        yaml.add_representer(np.ndarray, numpy_representer_seq)

        # each yaml dump has trailing '\n', which we identify and remove
        s = yaml.dump(self, default_flow_style=print_as_line, width=1000)  # , explicit_end=True

        s = re.sub(r'\w+: null', '', s)  # RegEx removes null keys. Demo: https://regex101.com/r/dZ9iI8/1
        # s = re.sub(r'\b\w+:\s+null(|$)', '', s).strip() # RegEx removes null keys.
        # s = s.replace('\n...\n','')   # trim trailing new line (explicit end)
        s = re.sub(u'(?imu)^\s*\n', u'', s)  # removes lines of spaces

        s = s.replace('!!python/object:', '').replace('!!python/tuple', '')
        s = s.replace('__main__.', '').replace(type(self).__name__ + '.', '').replace('SpecPrinter.', '')
        s = s.replace('OptionValuation.', '').replace('OptionSeries.', '')
        s = s.replace('qfrm.', '').replace('Util.', '').replace('!ndarray: ', '')

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


class Vec(tuple):
    """ It's a vectorized ``tuple`` (or ``list``).

    Basic algebraic operators are overloaded to act element-wise.
    Outputs of operations are lists. Performance yields ``numpy`` slightly,
    but conversion to ``array`` takes longer.
    See Python's `operator module docs <https://docs.python.org/3.5/library/operator.html>`_.

    **Vectorized (element-wise) behavior**, where a, b, c, x, y are numbers.
    ``(a,b)`` can be any iterable wrapped as ``vec`` type.

    - ``(a,b)+y -> [a+y, b+y];   [a,b]+[y] -> [a+y, b+y];    [a,b]+[x,y] -> [a+x, b+y]``
    - ``(a,b)**y -> [a**y, b**y];   [a,b]**[y] -> [a**y, b**y];    [a,b]**[x,y] -> [a**x, b**y]``
    - ``(a,b)<y -> [a<y, b<y];   [a,b]**[y] -> [a<y, b<y];    [a,b]**[x,y] -> [a<x, b<y]``
    - ``-(a,b) -> [-a, -b]``
    - ``(a,b).exp() -> [exp(a),exp(b)]``

    Examples
    --------
    >>> from numpy import array; from timeit import repeat; a = Vec((1, 2)); na = array([1, 2, 3]);

    >>> [a + 1, a + 1., a + (1,), a + a]  # right addition only! These will fail: -10 + a, (-10,) + a, a + array(1)
    [(2, 3), (2.0, 3.0), (2, 3), (2, 4)
    >>> [a - 1, a - 1., a - (1,), a - a, op.sub(a,1)]  # right subtraction only!
    [(0, 1), (0.0, 1.0), (0, 1), (0, 0), (0, 0)]
    >>> [a * 2, a * 2., a * (2,), a * a]  # right multiplication only!
    [(2, 4), (2.0, 4.0), (2, 4), (1, 4)]
    >>> [a / 2, a / 2., a / (2,), a / a]  # right division only!
    [(0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (1.0, 1.0)]
    >>> [a > 1, a > 1., a > (1,), a > a]
    [(False, True), (False, True), (False, True), (False, False)]
    >>> [a >= 2, a >= 2., a >= (2,), a >= a]
    [(False, True), (False, True), (False, True), (True, True)]
    >>> [a == 1, a == 1., a == (1,), a / a]
    [(True, False), (True, False), (True, False), (1.0, 1.0)]
    >>> [a != 1, a != 1., a != (1,), a != a]
    [(False, True), (False, True), (False, True), (False, False)]
    >>> [a < 2, a < 2., a < (2,), a < a]
    [(True, False), (True, False), (True, False), (False, False)]
    >>> [a <= 1, a <= 1., a <= (1,), a <= a]
    [(True, False), (True, False), (True, False), (True, True)]
    >>> (-a, op.neg(a), a.__neg__())
    ((-1, -2), (-1, -2), (-1, -2))
    >>> [op.abs(-a), (-a).__abs__()]
    [(1, 2), (1, 2)]
    >>> (op.concat(a, 1), op.concat(a, (1,)), op.concat(a, a))
    >>> a.exp()
    (2.718281828459045, 7.38905609893065)
    >>> a.log()
    (0.0, 0.6931471805599453, 1.0986122886681098)
    >>> [a.max(1.5), a.max((1.5,)), a.max(a + .5), Vec((-2,-1,0,1,2)).max(0)]
    [(1.5, 2), (1.5, 2), (1.5, 2.5), (0, 0, 0, 1, 2)]
    >>> [a.min(1.5), a.min((1.5,)), a.min(a + .5), Vec((-2,-1,0,1,2)).min(0)]
    [(1, 1.5), (1, 1.5), (1, 2), (-2, -1, 0, 0, 0)]
    >>> a.cumsum()
    (1, 3, 6)
    >>> a.pow(2)
    (1, 4, 9)
    >>> a.map(math.log10)
    (0.0, 0.3010299956639812, 0.47712125471966244)

    Performance comparison to ``numpy``:

    >>> repeat('a + 5', 'from __main__ import a', number=100000)     # 3x slower  # doctest: +ELLIPSIS
    >>> repeat('na + 5', 'from __main__ import na', number=100000)   # doctest: +ELLIPSIS

    >>> repeat('Vec((1, 2)) + 5', 'from __main__ import Vec', number=100000)      # 1.5x slower # doctest: +ELLIPSIS
    >>> repeat('array((1, 2)) + 5', 'from __main__ import array', number=100000)  # doctest: +ELLIPSIS
    """
    def __add__(self, y): return self.op(y, op.add)
    def __sub__(self, y): return self.op(y, op.sub)
    def __mul__(self, y): return self.op(y, op.mul)
    def __truediv__(self, y): return self.op(y, op.truediv)
    def __pow__(self, y): return self.op(y, op.pow) # [a,b]**[c,d] -> [a**c,b**d]; [a,b]**[c,d] -> [a**y,b**y]
    def __lt__(self, y): return self.op(y, op.lt)
    def __le__(self, y): return self.op(y, op.le)
    def __eq__(self, y): return self.op(y, op.eq)
    def __ne__(self, y): return self.op(y, op.ne)
    def __ge__(self, y): return self.op(y, op.ge)
    def __gt__(self, y): return self.op(y, op.gt)
    def __neg__(self): return Vec(map(op.neg, self))
    def __abs__(self): return Vec(map(op.abs, self))
    def __concat__(self, y): return self.op(y, op.concat)  # concatination of vectors
    def exp(self): return Vec(map(math.exp, self))
    def log(self): return Vec(map(math.log, self))
    def sqrt(self): return Vec(map(math.sqrt, self))
    def map(self, fun): return Vec(map(fun, self))
    def max(self, y): return self.op(y, max)
    def min(self, y): return self.op(y, min)
    def cumsum(self): pass
    def op(self, y, op):
        if isinstance(y, numbers.Number): out = [op(i, y) for i in self]
        else:
            if len(y) == 1: out = [op(i, y[0]) for i in self]
            elif len(y) == len(self): out =[op(i, j) for i, j in zip(self, y)]
            else: print('Opeartion failed. Assure y is a number, singleton or iterable of matching length')
        return Vec(out)


#
# class tupleX(tuple):
#     """
#     >>> a = tupleX((1, 2, 3));     b = tupleX([3, 4, 5])
#     >>> a.add(10)
#     (11, 12, 13)
#     >>> a.sub(1)
#     (11, 12, 13)
#     >>> a.add(b)
#     (4, 6, 8)
#     >>> a.mult(10)
#     (10, 20, 30)
#     >>> a.mult(b)
#     (3, 8, 15)
#     >>> a.exp()
#     (2.718281828459045, 7.38905609893065, 20.085536923187668)
#     >>> a.log()
#     (0.0, 0.6931471805599453, 1.0986122886681098)
#     >>> a.cumsum()
#     (1, 3, 6)
#     >>> a.pow(2)
#     (1, 4, 9)
#     >>> a.map(math.log10)
#     (0.0, 0.3010299956639812, 0.47712125471966244)
#     """
#     def add(self, y):  return tupleX(Util.add(self, y))
#     def sub(self, y): return tupleX(Util.sub(self, y))
#     def mult(self, y): return tupleX(Util.mult(self, y))
#     def exp(self): return tupleX(Util.exp(self))
#     def log(self): return tupleX(Util.log(self))
#     def cumsum(self): return tupleX(Util.cumsum(self))
#     def pow(self, y): return tupleX(Util.pow(self, y))
#     def sqrt(self): return tupleX(Util.sqrt(self))
#     def map(self, fun): return tupleX(Util.map(fun, self))
#     # def __pow__(self, power): return self.pow(self, power)
