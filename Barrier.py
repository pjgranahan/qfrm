def Barrier_BS(S0,K,r,q,sigma,T,H,Right,knock,dir):

    from scipy.stats import norm
    from numpy import exp, log, sqrt

    # Compute Parameters
    d1 = (log(S0/K) + (r-q+(sigma**2)/2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    c = S0*exp(-q*T)*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
    p = K*exp(-r*T)*norm.cdf(-d2) - S0*exp(-q*T)*norm.cdf(-d1)

    l = (r-q+sigma**2)/(sigma**2)
    y = log((H**2)/(S0*K))/(sigma*sqrt(T)) + l*sigma*sqrt(T)
    x1 = log(S0/H)/(sigma*sqrt(T)) + l*sigma*sqrt(T)
    y1 = log(H/S0)/(sigma*sqrt(T)) + l*sigma*sqrt(T)

    # Consider Call Option
    # Two Situations: H<=K vs H>K
    if (Right == 'call'):
        if (H<=K):
            cdi = S0*exp(-q*T)*((H/S0)**(2*l))*norm.cdf(y) - K*exp(-r*T)*((H/S0)**(2*l-2))*norm.cdf(y-sigma*sqrt(T))
            cdo = S0*norm.cdf(x1)*exp(-q*T) - K*exp(-r*T)*norm.cdf(x1-sigma*sqrt(T)) \
                  - S0*exp(-q*T)*((H/S0)**(2*l))*norm.cdf(y1) + K*exp(-r*T)*((H/S0)**(2*l-2))*norm.cdf(y1-sigma*sqrt(T))
            cdo = c - cdi
            cuo = 0
            cui = c
        else:
            cdo = S0*norm.cdf(x1)*exp(-q*T) - K*exp(-r*T)*norm.cdf(x1-sigma*sqrt(T)) \
                  - S0*exp(-q*T)*((H/S0)**(2*l))*norm.cdf(y1) + K*exp(-r*T)*((H/S0)**(2*l-2))*norm.cdf(y1-sigma*sqrt(T))
            cdi = c - cdo
            cui = S0*norm.cdf(x1)*exp(-q*T) - K*exp(-r*T)*norm.cdf(x1-sigma*sqrt(T)) - \
                  S0*exp(-q*T)*((H/S0)**(2*l))*(norm.cdf(-y)-norm.cdf(-y1)) + \
                  K*exp(-r*T)*((H/S0)**(2*l-2))*(norm.cdf(-y+sigma*sqrt(T))-norm.cdf(-y1+sigma*sqrt(T)))
            cuo = c - cui
    # Consider Put Option
    # Two Situations: H<=K vs H>K
    else:
        if (H>K):
            pui = -S0*exp(-q*T)*((H/S0)**(2*l))*norm.cdf(-y) + K*exp(-r*T)*((H/S0)*(2*l-2))*norm.cdf(-y+sigma*sqrt(T))
            puo = p - pui
            pdo = 0
            pdi = p
        else:
            puo = -S0*norm.cdf(-x1)*exp(-q*T) + K*exp(-r*T)*norm.cdf(-x1+sigma*sqrt(T)) + \
                  S0*exp(-q*T)*((H/S0)**(2*l))*norm.cdf(-y1) - K*exp(-r*T)*((H/S0)**(2*l-2))*norm.cdf(-y1+sigma*sqrt(T))
            pui = p - puo
            pdi = -S0*norm.cdf(-x1)*exp(-q*T) + K*exp(-r*T)*norm.cdf(-x1+sigma*sqrt(T)) + \
                  S0*exp(-q*T)*((H/S0)**(2*l))*(norm.cdf(y)-norm.cdf(y1)) - \
                  K*exp(-r*T)*((H/S0)**(2*l-2))*(norm.cdf(y-sigma*sqrt(T)) - norm.cdf(y1-sigma*sqrt(T)))
            pdo = p - pdi

    if (Right == 'call'):
        if (knock == 'down'):
            if (dir == 'in'):
                return(cdi)
            else:
                return(cdo)
        else:
            if (dir == 'in'):
                return(cui)
            else:
                return(cdi)
    else:
        if (knock == 'down'):
            if (dir == 'in'):
                return(pdi)
            else:
                return(pdi)
        else:
            if (dir == 'in'):
                return(pui)
            else:
                return(pdi)


Barrier_BS(50,40,0.01,0.03,0.15,1,60,'call','down','in')
Barrier_BS(50,40,0.01,0.03,0.15,0.5,60,'call','up','in')
Barrier_BS(60,40,0.01,0.03,0.15,0.75,30,'put','up','in')
Barrier_BS(50,30,0.01,0.03,0.15,10,60,'put','down','out')
Barrier_BS(90,100,0.05,0.45,0.15,3,72,'call','up','out')
Barrier_BS(100,80,0.1,0.15,0.2,5,60,'put','down','out')





