def Barrier_BS(S0,K,r,q,sigma,T,H,Right,knock,dir):

    from scipy.stats import norm
    from numpy import exp, log, sqrt


    assert H<=K, "For 'down and in' option, H should be less than K."

    d1 = (log(S0/K) + (r-q+(sigma**2)/2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    c = S0*exp(-q*T)*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
    p = K*exp(-r*T)*norm.cdf(-d2) - S0*exp(-q*T)*norm.cdf(-d1)

    l = (r-q+sigma**2)/(sigma**2)
    y = log((H**2)/(S0*K))/(sigma*sqrt(T)) + l*sigma*sqrt(T)
    x1 = log(S0/H)/(sigma*sqrt(T)) + l*sigma*sqrt(T)
    y1 = log(H/S0)/(sigma*sqrt(T)) + l*sigma*sqrt(T)

    if (Right == 'call'):
        if (knock == 'down'):
            cdi = S0*exp(-q*T)*((H/S0)**(2*l))*norm.cdf(y) - K*exp(-r*T)*((H/S0)**(2*l-2))*norm.cdf(y-sigma*sqrt(T))
            cdo = S0*norm.cdf(x1)*exp(-q*T) - K*exp(-r*T)*norm.cdf(x1-sigma*sqrt(T)) \
                  - S0*exp(-q*T)*((H/S0)**(2*l))*norm.cdf(y1) + K*exp(-r*T)*((H/S0)**(2*l-2))*norm.cdf(y1-sigma*sqrt(T))
            cdo = c - cdi
            optionin = cdi
            optiono = cdo
        else:
            cui = S0*norm.cdf(x1)*exp(-q*T) - K*exp(-r*T)*norm.cdf(x1-sigma*sqrt(T)) - \
                  S0*exp(-q*T)*((H/S0)**(2*l))*(norm.cdf(-y)-norm.cdf(-y1)) + \
                  K*exp(-r*T)*((H/S0)**(2*l-2))*(norm.cdf(-y+sigma*sqrt(T))-norm.cdf(-y1+sigma*sqrt(T)))
            cuo = c - cui
            optionin = cui
            optiono = cuo
    else:
        if (knock == 'down'):
            pui = -S0*exp(-q*T)*((H/S0)**(2*l))*norm.cdf(-y) + K*exp(-r*T)*((H/S0)*(2*l-2))*norm.cdf(-y+sigma*sqrt(T))
            puo = -S0*norm.cdf(-x1)*exp(-q*T) + K*exp(-r*T)*norm.cdf(-x1+sigma*sqrt(T)) + \
                  S0*exp(-q*T)*((H/S0)**(2*l))*norm.cdf(-y1) - K*exp(-r*T)*((H/S0)**(2*l-2))*norm.cdf(-y1+sigma*sqrt(T))
            puo = p - pui
            optionin = pui
            optiono = puo
        else:
            pdi = -S0*norm.cdf(-x1)*exp(-q*T) + K*exp(-r*T)*norm.cdf(-x1+sigma*sqrt(T)) + \
                  S0*exp(-q*T)*((H/S0)**(2*l))*(norm.cdf(y)-norm.cdf(y1)) - \
                  K*exp(-r*T)*((H/S0)**(2*l-2))*(norm.cdf(y-sigma*sqrt(T)) - norm.cdf(y1-sigma*sqrt(T)))
            pdo = p - pdi
            optionin = pdi
            optiono = pdo

    if (dir=='in'):
        return(optionin)
    else:
        return(optiono)

Barrier_BS(50,40,0.01,0.03,0.15,1,30,'call','down','in')





