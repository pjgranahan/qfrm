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
    def to_tuple(a, leaf_as_float=False):
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
        try:  return tuple((Util.to_tuple(i)) for i in a)
        except TypeError: return float(a) if leaf_as_float else a
