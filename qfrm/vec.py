import math
import numbers
import operator as op


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
    >>> from numpy import array; from timeit import repeat; v = Vec((1, 2)); nv = array([1, 2, 3]);
    >>> [type(Vec(4)), isinstance(Vec(4), tuple)]
    [<class 'Util.Vec'>, True]
    >>> [Vec(1) + v, v + 1, v + 1., v + (1,), v + v]  # right addition only! These will fail: -10 + a, (-10,) + a, a + array(1)
    [(2, 3), (2, 3), (2.0, 3.0), (2, 3), (2, 4)]
    >>> [v - 1, v - 1., v - (1,), v - v, op.sub(v,1)]  # right subtraction only!
    [(0, 1), (0.0, 1.0), (0, 1), (0, 0), (0, 1)]
    >>> [v * 2, v * 2., v * (2,), v * v]  # right multiplication only!
    [(2, 4), (2.0, 4.0), (2, 4), (1, 4)]
    >>> [v / 2, v / 2., v / (2,), v / v]  # right division only!
    [(0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (1.0, 1.0)]
    >>> [v ** 2, v ** 2., v ** (2,), v ** v]  # right multiplication only!
    [(1, 4), (1.0, 4.0), (1, 4), (1, 4)]
    >>> [v > 1, v > 1., v > (1,), v > v]
    [(False, True), (False, True), (False, True), (False, False)]
    >>> [v >= 2, v >= 2., v >= (2,), v >= v]
    [(False, True), (False, True), (False, True), (True, True)]
    >>> [v == 1, v == 1., v == (1,), v / v]
    [(True, False), (True, False), (True, False), (1.0, 1.0)]
    >>> [v != 1, v != 1., v != (1,), v != v]
    [(False, True), (False, True), (False, True), (False, False)]
    >>> [v < 2, v < 2., v < (2,), v < v]
    [(True, False), (True, False), (True, False), (False, False)]
    >>> [v <= 1, v <= 1., v <= (1,), v <= v]
    [(True, False), (True, False), (True, False), (True, True)]
    >>> (-v, op.neg(v), v.__neg__())
    ((-1, -2), (-1, -2), (-1, -2))
    >>> [op.abs(-v), (-v).__abs__()]
    [(1, 2), (1, 2)]
    >>> [v.max(1.5), v.max((1.5,)), v.max(v + .5), Vec((-2,-1,0,1,2)).max(0)]
    [(1.5, 2), (1.5, 2), (1.5, 2.5), (0, 0, 0, 1, 2)]
    >>> [v.min(1.5), v.min((1.5,)), v.min(v + .5), Vec((-2,-1,0,1,2)).min(0)]
    [(1, 1.5), (1, 1.5), (1, 2), (-2, -1, 0, 0, 0)]
    >>> v.exp
    (2.718281828459045, 7.38905609893065)
    >>> v.log
    (0.0, 0.6931471805599453)
    >>> v.sqrt
    (1.0, 1.4142135623730951)
    >>> Vec([1,2,3,4,5]).cumsum
    (1, 3, 6, 10, 15)
    >>> v.map(math.log10)
    (0.0, 0.3010299956639812)

    Compare performance to ``numpy``:

    >>> repeat('v + 5', 'from __main__ import v', number=10000)     # 3x slower  # doctest: +ELLIPSIS
    >>> repeat('nv + 5', 'from __main__ import nv', number=10000)   # doctest: +ELLIPSIS

    >>> repeat('Vec((1, 2)) + 5', 'from __main__ import Vec', number=10000)      # 1.5x slower # doctest: +ELLIPSIS
    >>> repeat('array((1, 2)) + 5', 'from __main__ import array', number=10000)  # doctest: +ELLIPSIS
    """
    def __new__(self, x):  # Vec(4)
        if isinstance(x, numbers.Number): x = Vec((x,))
        return super(Vec, self).__new__(self, x)
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
    @property
    def exp(self): return Vec(map(math.exp, self))
    @property
    def log(self): return Vec(map(math.log, self))
    @property
    def sqrt(self): return Vec(map(math.sqrt, self))
    def max(self, y): return self.op(y, max)
    def min(self, y): return self.op(y, min)
    def map(self, fun): return Vec(map(fun, self))
    @property
    def cumsum(self):
        def cumsum_(it):
            total = 0
            for a in it:  total += a; yield total
        return Vec(cumsum_(self))
    def op(self, y, op):
        if isinstance(y, numbers.Number): out = [op(i, y) for i in self]
        else:
            if len(y) == 1: out = [op(i, y[0]) for i in self]
            elif len(self) == 1: out = [op(self[0], j) for j in y]
            elif len(y) == len(self): out =[op(i, j) for i, j in zip(self, y)]
            else: print('Opeartion failed. Assure y is a number, singleton or iterable of matching length')
        return Vec(out)