"""
Rough Bergomi Model
====================================
Reference: Bayer, C., Friz, P. & Gatheral, J. (2016). Pricing under Rough Volatility.
           Quantitative Finance, 16(6), 887-904.

Author: [Ton nom]
"""

import numpy as np
from models.black_scholes import implied_vol


def fractional_kernel(t: np.ndarray, H: float) -> np.ndarray:
    return np.where(t > 0, t ** (H - 0.5), 0.0)


def simulate_rough_bergomi(S: float, T: float, r: float,
                            H: float, eta: float, rho: float,
                            xi0: float = None,
                            n_paths: int = 30000,
                            n_steps: int = 100) -> tuple:
    if xi0 is None:
        xi0 = 0.04

    dt = T / n_steps
    t  = np.linspace(0, T, n_steps + 1)

    dW1 = np.random.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.standard_normal((n_paths, n_steps)) * np.sqrt(dt)

    W_H = np.zeros((n_paths, n_steps + 1))
    for i in range(1, n_steps + 1):
        s = t[1:i+1]
        k = fractional_kernel(t[i] - s + dt, H)
        W_H[:, i] = np.sum(k[np.newaxis, :] * dW2[:, :i], axis=1)

    v = xi0 * np.exp(eta * W_H - 0.5 * eta**2 * t[np.newaxis, :] ** (2 * H))

    S_paths = np.zeros((n_paths, n_steps + 1))
    S_paths[:, 0] = S

    for i in range(n_steps):
        S_paths[:, i+1] = S_paths[:, i] * np.exp(
            (r - 0.5 * v[:, i]) * dt + np.sqrt(v[:, i] * dt) * dW1[:, i]
        )

    return S_paths, v


def rbergomi_smile(S: float, strikes: np.ndarray, T: float, r: float,
                   H: float, eta: float, rho: float,
                   xi0: float = None) -> np.ndarray:
    """
    Simulate once, reprice across all strikes.
    Strikes should be normalized (divide by spot before calling).
    """
    S_paths, _ = simulate_rough_bergomi(S, T, r, H, eta, rho, xi0=xi0,
                                         n_paths=50000, n_steps=200)
    S_T = S_paths[:, -1]

    ivols = []
    for K in strikes:
        price = np.exp(-r * T) * np.mean(np.maximum(S_T - K, 0))
        iv    = implied_vol(price, S, K, T, r)
        ivols.append(iv if not np.isnan(iv) else 0.0)

    return np.array(ivols)
