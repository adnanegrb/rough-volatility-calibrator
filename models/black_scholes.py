"""
Black-Scholes Option Pricing Model:

Reference: Black & Scholes (1973), Journal of Political Economy, 81:635-654

This is the baseline model for the rough-volatility-calibrator project.
We start here because every advanced model (Heston, rBergomi) is benchmarked against BS.

Author: GARAB Adnane
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """
    Closed-form BS price for a European option.
    option_type: 'call' or 'put'
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """
    Delta — how much the option price moves when the spot moves by 1.
    This is the main hedging ratio used in practice.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1


def implied_vol(market_price: float, S: float, K: float, T: float, r: float, option_type: str = "call") -> float:
    """
    Recover the implied vol from a market price by inverting BS numerically.
    We use Brent's method — more robust than Newton when the function is flat.
    Returns NaN if no solution is found (deep ITM/OTM edge cases).
    """
    objective = lambda sigma: bs_price(S, K, T, r, sigma, option_type) - market_price

    try:
        return brentq(objective, 1e-6, 10.0, xtol=1e-8)
    except ValueError:
        return np.nan


def plot_vol_smile(strikes: list, market_prices: list, S: float, T: float, r: float):
    """
    Plot the implied vol smile across strikes.
    This is the key visual — BS would give a flat line, real markets don't.
    """
    import matplotlib.pyplot as plt

    ivols = [implied_vol(p, S, K, T, r) * 100 for p, K in zip(market_prices, strikes)]

    plt.figure(figsize=(10, 6))
    plt.plot(strikes, ivols, 'o-', color='steelblue', linewidth=2)
    plt.axvline(x=S, color='red', linestyle='--', alpha=0.5, label='ATM spot')
    plt.xlabel("Strike K")
    plt.ylabel("Implied Volatility (%)")
    plt.title(f"Volatility Smile  |  T = {T:.2f}y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/vol_smile.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    call = bs_price(S, K, T, r, sigma, "call")
    put  = bs_price(S, K, T, r, sigma, "put")

    # Call-put parity check: C - P should equal S - K*e^{-rT}
    parity = S - K * np.exp(-r * T)
    print(f"Call={call:.4f} | Put={put:.4f} | Parity check: {call - put:.4f} vs {parity:.4f}")

    iv = implied_vol(call, S, K, T, r)
    print(f"Implied vol: {iv:.4f} (expected {sigma})")
