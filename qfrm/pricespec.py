from qfrm.specprinter import SpecPrinter


class PriceSpec(SpecPrinter):
    """ Object for storing calculated price and related intermediate parameters.

    Use this object to store the price, sub/method and any intermediate results in your option object.

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

        >>> PriceSpec(price=1/7)
        PriceSpec
        price: 0.142857143

        >>> PriceSpec(price=1/7, print_precision=4)
        PriceSpec
        price: 0.1429
        """

        # super().__init__(kwargs)
        # if 'print_precision' in kwargs:
        #     super().__init__(print_precision=kwargs['print_precision'])
        #     del kwargs['print_precision']
        # super().__init__(print_precision=print_precision)
        SpecPrinter.print_precision = print_precision
        self.add(**kwargs)

    # @property
    # def px(self):
    #     """Getter method for price variable.
    #
    #     Getter and setter allow access to price and use of user-defined rounding of price value.
    #
    #     Returns
    #     -------
    #     None or float
    #         Properly rounded price value (if exists)
    #
    #     """
    #     try: return self.print_value(self._px)
    #     except: return None
    #
    # @px.setter
    # def px(self, px): self._px = px

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