import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bs_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price, S, K, T, r):
    lb = max(S - K * np.exp(-r * T), 0) + 1e-10
    if price <= lb:
        return np.nan
    try:
        return brentq(lambda s: bs_price(S, K, T, r, s) - price, 1e-4, 5.0)
    except:
        return np.nan


def simulate_rough_bergomi(T, r, H, eta, rho, xi0, n_paths=100000, n_steps=200):
    """
    Simulate Rough Bergomi paths with S0=1 (normalized).
    Uses antithetic variates + martingale correction.
    """
    assert n_paths % 2 == 0
    half  = n_paths // 2
    dt    = T / n_steps
    grid  = np.arange(1, n_steps + 1) * dt
    alpha = H - 0.5

    Z1_h = np.random.standard_normal((half, n_steps))
    Z2_h = rho * Z1_h + np.sqrt(1 - rho**2) * np.random.standard_normal((half, n_steps))
    Z1   = np.concatenate([Z1_h, -Z1_h], axis=0)
    Z2   = np.concatenate([Z2_h, -Z2_h], axis=0)

    dW2 = Z2 * np.sqrt(dt)

    V_H = np.zeros((n_paths, n_steps))
    for i in range(n_steps):
        j      = np.arange(i + 1)
        kernel = (grid[i] - grid[j] + dt) ** alpha
        V_H[:, i] = (dW2[:, :i+1] * kernel[np.newaxis, :]).sum(axis=1)

    var_VH = grid ** (2 * H) / (2 * H)
    v = xi0 * np.exp(eta * V_H - 0.5 * eta**2 * var_VH[np.newaxis, :])

    log_S = np.zeros(n_paths)
    for i in range(n_steps):
        vi     = np.maximum(v[:, i], 0)
        log_S += (r - 0.5 * vi) * dt + np.sqrt(vi * dt) * Z1[:, i]

    S_T  = np.exp(log_S)
    S_T *= np.exp(r * T) / S_T.mean()

    return S_T, v


def rbergomi_smile(strikes_norm, T, r, H, eta, rho, xi0):
    """
    Implied vol smile across normalized strikes (K/S).
    """
    S_T, _ = simulate_rough_bergomi(T, r, H, eta, rho, xi0)

    ivols = []
    for K in strikes_norm:
        price = np.exp(-r * T) * np.mean(np.maximum(S_T - K, 0))
        iv    = implied_vol(price, 1.0, K, T, r)
        ivols.append(iv if not np.isnan(iv) else 0.0)

    return np.array(ivols)
