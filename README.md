# rough-volatility-calibrator

Volatility models calibrated on real SPX options data — from Black-Scholes to Rough Bergomi.

The core idea: BS assumes constant vol. Markets don't. This project builds up from the BS
baseline, through Heston, to the Rough Bergomi model (Bayer, Friz, Gatheral 2016) which
captures the empirical roughness of volatility (H ≈ 0.1) and reproduces the short-term
implied vol smile that classical models fail to fit.


## Why Rough Bergomi

The volatility smile — the fact that implied vol varies across strikes — is a well-known
failure of Black-Scholes. Heston partially fixes this but struggles at short maturities.

Gatheral, Jaisson & Rosenbaum (2018) showed that realized volatility behaves like a
fractional Brownian motion with Hurst exponent H ≈ 0.1, far rougher than standard BM.
The Rough Bergomi model encodes this directly and fits the SPX smile remarkably well
with very few parameters.


## Stack

- Python 3.10+
- numpy, scipy, matplotlib
- yfinance for market data
```bash
pip install -r requirements.txt
```


## References

- Black, F. & Scholes (1973). *The Pricing of Options and Corporate Liabilities*
- Heston, S. (1993). *A Closed-Form Solution for Options with Stochastic Volatility*
- Bayer, C., Friz, P. & Gatheral, J. (2016). *Pricing under Rough Volatility*
- Gatheral, J., Jaisson, T. & Rosenbaum, M. (2018). *Volatility is Rough*
