import numpy as np
from scipy.stats import norm
from models.black_scholes import implied_vol


def heston_mc(S: float, K: float, T: float, r: float,
              v0: float, kappa: float, theta: float,
              xi: float, rho: float,
              n_paths: int = 50000, n_steps: int = 200) -> float:
    """
    Monte Carlo price of a European call under Heston dynamics.

    dS = r*S*dt + sqrt(v)*S*dW1
    dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW2
    corr(dW1, dW2) = rho

    Params:
        v0    : initial variance
        kappa : mean reversion speed
        theta : long-run variance
        xi    : vol of vol
        rho   : correlation between spot and vol
    """
    dt = T / n_steps
    S_t = np.full(n_paths, S)
    v_t = np.full(n_paths, v0)

    for _ in range(n_steps):
        Z1 = np.random.standard_normal(n_paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_paths)

        # Variance process — full truncation to avoid negative variance
        v_t = np.maximum(v_t, 0)
        v_t = v_t + kappa * (theta - v_t) * dt + xi * np.sqrt(v_t * dt) * Z2

        # Spot process
        S_t = S_t * np.exp((r - 0.5 * v_t) * dt + np.sqrt(v_t * dt) * Z1)

    payoff = np.maximum(S_t - K, 0)
    return np.exp(-r * T) * np.mean(payoff)


def heston_smile(S: float, strikes: np.ndarray, T: float, r: float,
                 v0: float, kappa: float, theta: float,
                 xi: float, rho: float) -> np.ndarray:
    """
    Compute the implied vol smile produced by Heston across a range of strikes.
    """
    ivols = []
    for K in strikes:
        price = heston_mc(S, K, T, r, v0, kappa, theta, xi, rho)
        iv = implied_vol(price, S, K, T, r)
        ivols.append(iv)
    return np.array(ivols)


if __name__ == "__main__":
    # Typical SPX-calibrated Heston parameters
    S, T, r    = 100, 0.25, 0.05
    v0         = 0.04    # initial variance (vol = 20%)
    kappa      = 2.0     # mean reversion speed
    theta      = 0.04    # long-run variance
    xi         = 0.3     # vol of vol
    rho        = -0.7    # negative correlation (leverage effect)

    strikes = np.linspace(85, 115, 10)
    ivols   = heston_smile(S, strikes, T, r, v0, kappa, theta, xi, rho)

    for K, iv in zip(strikes, ivols):
        print(f"K={K:.0f}  IV={iv*100:.2f}%")
