"""

This module provides functions for calculating option prices using the Black-Scholes model.

Functions:
- nb_days_year(): Returns the number of trading days in a year.
- normal_cdf(d): Calculates the cumulative distribution function of a standard normal distribution.
- normal_pdf(d): Calculates the probability density function of a standard normal distribution.
- d1(s, k, t_in_days, v): Calculates the d1 parameter used in the Black-Scholes formula.
- d2(s, k, t_in_days, sigma): Calculates the d2 parameter used in the Black-Scholes formula.
- bs_call(s, k, t_in_days, v): Calculates the price of a call option using the Black-Scholes formula.
- bs_put(s, k, t_in_days, v): Calculates the price of a put option using the Black-Scholes formula.
- bs_call_s(s, k, t_in_days, v): Calculates the sensitivity of a call option price to the underlying asset price.
- bs_call_k(s, k, t_in_days, v): Calculates the sensitivity of a call option price to the strike price.
- bs_call_t(s, k, t_in_days, v): Calculates the sensitivity of a call option price to the time to expiration.
- bs_call_v(s, k, t_in_days, v): Calculates the sensitivity of a call option price to the volatility.
- bs_call_ss(s, k, t_in_days, v): Calculates the second-order sensitivity of a call option price to the underlying asset price.
- bs_call_st(s, k, t_in_days, v): Calculates the sensitivity of a call option price to the time to expiration and the underlying asset price.
- bs_put_s(s, k, t_in_days, v): Calculates the sensitivity of a put option price to the underlying asset price.
- bs_oom(s, k, t_in_days, v): Calculates the price of an option using the Black-Scholes formula, taking into account whether it is a call or put option.
- bs_put_iv(s, k): Calculates the implied volatility of a put option.
- bs_call_iv(s, k): Calculates the implied volatility of a call option.
- bs_put_imp_vol(s, k, t_in_days, premium): Calculates the implied volatility of a put option given the option premium.
- bs_call_imp_vol(s, k, t_in_days, premium): Calculates the implied volatility of a call option given the option premium.
"""
import numpy as np
import scipy.stats as si
import time
from colorama import Fore, Back, Style, init
# corolram init style auto-reset
init(autoreset=True)

def nb_days_year():
    return (252)


def normal_cdf(d):
    return si.norm._cdf(d)

def normal_pdf( d):
    return si.norm._pdf(d)

def d1(s, k, t_in_days, v):
    """
    Calculates the d1 value for the Black-Scholes formula.

    Parameters:
    - s: float, the current price of the underlying asset
    - k: float, the strike price of the option
    - t_in_days: float, the time to expiration in days
    - v: float, the volatility of the underlying asset

    Returns:
    - float, the d1 value

    """
    t_in_years = t_in_days / nb_days_year()
    return (np.log(s / k) + (0.5 * v ** 2) * t_in_years) / (v * np.sqrt(t_in_years))

def d2(s, k, t_in_days, v):
    """
        Calculates the d2 value for the Black-Scholes formula.
     Parameters:
     - s: float, the current price of the underlying asset
     - k: float, the strike price of the option
     - t_in_days: float, the time to expiration in days
     - v: float, the volatility of the underlying asset
     Returns:    def d2(s, k, t_in_days, v):
     - float, the d2 value
    """
    t_in_years = t_in_days / nb_days_year()
    return (np.log(s / k) + (- 0.5 * v ** 2) * t_in_years) / (v * np.sqrt(t_in_years))


def bs_call(s, k, t_in_days, v):
    return (s * normal_cdf(d1(s, k, t_in_days, v)) - k * normal_cdf(d2(s, k, t_in_days, v)))


def bs_put(s, k, t_in_days, v):
    return (-s * normal_cdf(-d1(s, k, t_in_days, v)) + k * normal_cdf(-d2(s, k, t_in_days, v)))

def bs_call_s(s, k, t_in_days, v):
    return (normal_cdf(d1(s, k, t_in_days, v)))
def bs_call_k(s, k, t_in_days, v):
    return (normal_cdf(d1(s, k, t_in_days, v))-1)
def bs_call_t(s, k, t_in_days, v):
    return (normal_pdf(d1(s, k, t_in_days, v))*s*v/(2*np.sqrt(t_in_days/nb_days_year())))
def bs_call_v(s, k, t_in_days, v):
    return (s * normal_pdf(d1(s, k, t_in_days, v)) * np.sqrt(t_in_days/nb_days_year()))
def bs_call_ss(s, k, t_in_days, v):
    return (normal_pdf(d1(s, k, t_in_days, v)) / (s * v * np.sqrt(t_in_days/nb_days_year())))
def bs_call_st(s, k, t_in_days, v):
    return (bs_call_s(s, k, t_in_days+1, v)- bs_call_s(s, k, t_in_days-1, v))*(252.0/2)

def bs_put_s(s, k, t_in_days, v):
    return (normal_cdf(d1(s, k, t_in_days, v))-1)
def bs_oom(s, k, t_in_days, v):
    if s > k:
        return (bs_put(s, k, t_in_days, v))
    return (bs_call(s, k, t_in_days, v))


def bs_put_iv(s, k):
    result = 0.0
    if s < k:
        result = k - s
    return (result)


def bs_call_iv(s, k):
    result = 0.0
    if s > k:
        result = s - k
    return (result)


def bs_put_imp_vol(s, k, t_in_days, premium):
    vu = 250 / 100
    vd = 0 / 100
    vm = (vu + vd) / 2
    for i in range(1, 50):
        vm = (vu + vd) / 2
        p = bs_put(s, k, t_in_days, vm)
        if p >= premium:
            vu = vm
        if p <= premium:
            vd = vm
    return (vm)


def bs_call_imp_vol(s, k, t_in_days, premium):
    vu = 250 / 100
    vd = 0 / 100
    vm = (vu + vd) / 2
    for i in range(1, 50):
        vm = (vu + vd) / 2
        p = bs_call(s, k, t_in_days, vm)
        if p >= premium:
            vu = vm
        if p <= premium:
            vd = vm
    return (vm)

def Price(s, k, t_in_days, v):
    c = bs_call(s, k, t_in_days, v)
    delta=bs_call_s(s, k, t_in_days, v)
    gamma=bs_call_ss(s, k, t_in_days, v)
    theta=bs_call_t(s, k, t_in_days, v)
    theta_1d=bs_call_t(s, k, t_in_days, v)/252
    vega=bs_call_v(s, k, t_in_days, v)
    print(f'\n >>')
    print(Fore.YELLOW+f'    [{s=}, {k=}, {t=}, {v=}]')
    print(Fore.LIGHTCYAN_EX+f'  Call Price: {c:5.4f}, {delta=:5.4f},{gamma=:5.4f}, {theta=:5.4f},{theta_1d=:5.4f} {vega=:5.2f}')
    print(f'\n >>')
    return {'s':s,'t':t_in_days,'v':v,'Price':c, 'Delta':delta, 'Gamma':gamma, 'Theta':theta, 'Theta_1d':theta_1d, 'Vega':vega}

def PnL_Explain(k,s_1,t_1,v_1,s_2,t_2,v_2):
    start=Price(s_1, k, t_1, v_1)
    end=Price(s_2, k, t_2, v_2)
    total_pnl=end['Price']-start['Price']
    print(f"PnL={total_pnl=:5.4f}")
    print(f'\n')
    delta_pnl=start['Delta']*(end['s']-start['s'])
    print(f"    Delta:   {delta_pnl=:+7.4f}")
    gamma_pnl=0.5*start['Gamma']*(end['s']-start['s'])**2
    print(f"    Gamma   :{gamma_pnl=:+7.4f}")
    theta_pnl=start['Theta_1d']*(end['t']-start['t'])  
    print(f"    Theta   :{theta_pnl=:+7.4f}")
    vega_pnl=start['Vega']*(end['v']-start['v'])
    print(f"    Vega    :{vega_pnl =:+7.4f}")
    print(f'\n  AllGreeks:{delta_pnl+gamma_pnl+theta_pnl+vega_pnl=:+7.4f}')
    print(f'\n  DeltaNeut:{gamma_pnl+theta_pnl+vega_pnl=:+7.4f}')



if __name__ == '__main__':

    """
    Test the Black-Scholes functions.
    """
    assert abs(bs_call(100, 100, nb_days_year(), 0.2) - 7.965567455405804) < 0.000000000000001
    assert abs(bs_put(100, 100,nb_days_year(), 0.2) - 7.965567455405804) < 0.000000000000001
   
    """"
    Test the Black-Scholes Greeks
    """
    s=100
    k=100
    t=30
    v=0.2
    Price(s, k, t, v)

    s=100
    k=100
    t=29
    v=0.2
    Price(s, k, t, v)

    
    s_1=100 ; s_2=101.26
    k=100
    t_1=30 ; t_2=29
    v_1=0.2 ; v_2=0.20    
    
    PnL_Explain(k,s_1,t_1,v_1,s_2,t_2,v_2)




