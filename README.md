# Dissertation
# Stochastic Volatility Model Calibration: A Horse Race

A Python implementation comparing three stochastic volatility models —
Heston, Scott, and Hull-White — against a synthetic options market.
Built as part of an MSc Financial Engineering dissertation at the
University of York.

## Overview

Black-Scholes assumes constant volatility. Real markets don't.
This project uses the Heston model as a "true" synthetic market,
generates an implied volatility smile from it, and then calibrates
the Scott and Hull-White models against that smile to see which
fits better — and where each model breaks down.

The result is a visual "horse race": a volatility smile chart showing
how closely each competing model can replicate the Heston-generated market.

## Models Implemented

| Model | Volatility Process | Pricing Method |
|---|---|---|
| **Heston (1993)** | Mean-reverting CIR process | Fourier inversion (semi-analytical) |
| **Scott (1987)** | Mean-reverting log-normal (OU) | Monte Carlo (Euler-Maruyama) |
| **Hull-White (1987)** | Log-normal with drift | Monte Carlo (Euler-Maruyama) |

## What the Code Does

1. **Generates a synthetic market** using the Heston model with known parameters
2. **Computes implied volatilities** across a range of strikes using Black-Scholes inversion
3. **Calibrates the Scott model** by minimising the sum of squared IV errors
4. **Calibrates the Hull-White model** using the same objective function
5. **Plots the resulting volatility smiles** for visual comparison

## Requirements

```bash
pip install numpy scipy matplotlib
```

Python 3.8+ recommended.

## Usage

```bash
python pricing.py
```

The script will:
- Print the synthetic market's implied volatility surface to the console
- Run calibration for both competing models (this takes 2–5 minutes)
- Display a volatility smile comparison chart

## Key Results

The Heston model generates a pronounced volatility skew (higher IV
for low strikes) driven by its negative spot-vol correlation parameter
`rho = -0.6`. The Scott model, sharing a similar mean-reverting
log-volatility structure, fits this skew more closely. The Hull-White
model, lacking mean reversion, struggles to match the curvature at
the wings.

## Parameters (True Market)

| Parameter | Value | Description |
|---|---|---|
| `S0` | 100 | Initial spot price |
| `r` | 0.02 | Risk-free rate |
| `T` | 1.0 | Time to maturity (years) |
| `v0` | 0.04 | Initial variance |
| `kappa` | 2.0 | Mean reversion speed |
| `theta` | 0.04 | Long-run variance |
| `sigma_v` | 0.3 | Volatility of volatility |
| `rho` | -0.6 | Spot-vol correlation |

## Project Structure
├── pricing.py # Main script
└── README.md
