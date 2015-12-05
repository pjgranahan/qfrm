from qfrm.specprinter import SpecPrinter


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