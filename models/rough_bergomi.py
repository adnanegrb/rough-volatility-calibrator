import numpy as np
from models.black_scholes import implied_vol


def fractional_kernel(t: np.ndarray, H: float) -> np.ndarray:
    """
    Volterra kernel K(t) = t^(H - 0.5).
    This is what makes the BM fractional — memory effect.
    """
    return np.where(t > 0, t ** (H - 0.5), 0.0)


def simulate_rough_bergomi(S: float, T: float, r: float,
                            H: float, eta: float, rho: float,
                            xi0: float = None,
                            n_paths: int = 20000, n_steps: int = 200) -> tuple:
    """
    Simulate paths under Rough Bergomi dynamics.
    dS = r*S*dt + sqrt(v)*S*dW
    v_t = xi0 * exp(eta * W^H_t - 0.5 * eta^2 * t^(2H))
    where W^H is a Volterra process approximating fBm with Hurst H.
    Returns spot paths and variance paths.
    """
    if xi0 is None:
        xi0 = 0.15**2

    dt  = T / n_steps
    t   = np.linspace(0, T, n_steps + 1)

    dW1 = np.random.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.standard_normal((n_paths, n_steps)) * np.sqrt(dt)

    W_H = np.zeros((n_paths, n_steps + 1))
    for i in range(1, n_steps + 1):
        s      = t[1:i+1]
        k      = fractional_kernel(t[i] - s + dt, H)
        W_H[:, i] = np.sum(k[np.newaxis, :] * dW2[:, :i], axis=1)

    v = xi0 * np.exp(eta * W_H - 0.5 * eta**2 * t[np.newaxis, :] ** (2 * H))

    S_paths = np.zeros((n_paths, n_steps + 1))
    S_paths[:, 0] = S

    for i in range(n_steps):
        S_paths[:, i+1] = S_paths[:, i] * np.exp(
            (r - 0.5 * v[:, i]) * dt + np.sqrt(v[:, i * dt) * dW1[:, i]
        )

    return S_paths, v


def rbergomi_price(S: float, K: float, T: float, r: float,
                   H: float, eta: float, rho: float,
                   n_paths: int = 20000, n_steps: int = 200) -> float:
    """
    Monte Carlo price of a European call under Rough Bergomi.
    """
    S_paths, _ = simulate_rough_bergomi(S, T, r, H, eta, rho, n_paths, n_steps)
    payoff = np.maximum(S_paths[:, -1] - K, 0)
    return np.exp(-r * T) * np.mean(payoff)


def rbergomi_smile(S: float, strikes: np.ndarray, T: float, r: float,
                   H: float, eta: float, rho: float,
                   xi0: float = None) -> np.ndarray:
    """
    Implied vol smile produced by Rough Bergomi across strikes.
    We simulate once and reprice across all strikes for efficiency.
    """
    S_paths, _ = simulate_rough_bergomi(S, T, r, H, eta, rho, xi0=xi0)
    S_T = S_paths[:, -1]

    ivols = []
    for K in strikes:
        price = np.exp(-r * T) * np.mean(np.maximum(S_T - K, 0))
        iv    = implied_vol(price, S, K, T, r)
        ivols.append(iv)

    return np.array(ivols)


if __name__ == "__main__":
    S, T, r = 100, 0.25, 0.05
    H       = 0.1    # roughness — empirical value on SPX
    eta     = 1.9    # vol of vol
    rho     = -0.9   # leverage effect

    strikes = np.linspace(85, 115, 10)
    ivols   = rbergomi_smile(S, strikes, T, r, H, eta, rho)

    for K, iv in zip(strikes, ivols):
        print(f"K={K:.0f}  IV={iv*100:.2f}%")
