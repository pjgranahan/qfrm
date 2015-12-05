import re

import numpy as np
import yaml

from qfrm.util import Util


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