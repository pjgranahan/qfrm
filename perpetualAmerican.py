__author__ = 'Tianyi Yao'

def pxBS(right='call', S=50., K=50., vol=0.3, r=0.08, q=0.01):
    """ pricing perpetual American option using Black-Scholes model
    :param right: call or put
    :type right: string
    :param S: current price of a stock underlying
    :type S: float
    :param K: strike price of the option
    :type K: float
    :param vol: volatility of the underlying
    :type vol: float
    :param r: risk free interest rate, cc, annualized
    :type r: float
    :param q: dividend
    :type q: float
    :return: option price
    :rtype: float



    :Example:
    >>> pxBS() #the default value
    37.190676833752335
    >>> pxBS('put',S=30.,K=30.,vol=0.25,r=0.05,q=0.01)
    5.837758667830096
    """

    #verify the inputs meet the requirement
    try:
        S, K, vol, r, q=float(S), float(K), float(vol), float(r), float(q)

    except:
        print('Input value error. S, K, vol, r, q should all be floats')

    assert right in ['call','put'], 'right should be either put or call'
    assert vol>=0, 'vol should be >=0'
    assert K>=0, 'K should be >=0'
    assert r>=0, 'r should be >=0'
    assert q>0, 'q should be >0, q=0 will give an alpha1 of infinity'


    #import external libraries
    from math import sqrt

    #compute parameters required in the pricing
    w=r-q-((vol**2)/2.)
    alpha1=(-w+sqrt((w**2)+2*(vol**2)*r))/(vol**2)
    H1=K*(alpha1/(alpha1-1))
    alpha2=(w+sqrt((w**2)+2*(vol**2)*r))/(vol**2)
    H2=K*(alpha2/(alpha2+1))

    #price the perpetual American option
    if right=='call':
        if S<H1:
            return (K/(alpha1-1))*((((alpha1-1)/alpha1)*(S/K))**alpha1)
        elif S>H1:
            return S-K
        else:
            print('The option cannot be priced')
    else:
        if S>H2:
            return (K/(alpha2+1))*((((alpha2+1)/alpha2)*(S/K))**(-alpha2))
        elif S<H2:
            return K-S
        else:
            print('The option cannot be priced ')

