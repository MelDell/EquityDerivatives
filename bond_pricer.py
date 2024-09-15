import numpy as np
def bond_pricer(coupon, rate, n):
    """
    Calculate the price of a bond given the coupon rate, interest rate, and number of periods.

    Parameters:
    coupon (float): The coupon payment of the bond.
    rate (float): The interest rate.
    n (int): The number of periods.

    Returns:
    float: The price of the bond.
    """
    price = 0
    for i in range(1, n + 1):
        price = price + coupon / (1 + rate) ** i
    price = price + 100 / (1 + rate) ** n
    return price

def bond_yield(coupon, price, n):
    """
    Calculate the yield of a bond given the coupon rate, bond price, and number of periods.

    Parameters:
    coupon (float): The coupon payment of the bond.
    price (float): The current price of the bond.
    n (int): The number of periods.

    Returns:
    float: The yield of the bond.
    """
    yield_max = 10
    yield_min = 0
    for i in range(100):
        yield_mid = (yield_max + yield_min) / 2
        bond_price = bond_pricer(coupon, yield_mid, n)
        if bond_price > price:
            yield_min = yield_mid
        else:
            yield_max = yield_mid
    return (yield_max + yield_min) / 2

def bond_yield_2y(coupon, price):

    return (coupon + np.sqrt(coupon**2 + 4*price*(coupon+100)))/(2*price)-1