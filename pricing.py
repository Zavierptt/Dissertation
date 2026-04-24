import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

# Suppress integration warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# --- Part 1: Heston Model as the "True" Market ---

def heston_char_func(u, T, r, kappa, theta, sigma_v, rho, v0, S0):
    """
    Heston characteristic function for option pricing.
    This is used to generate the 'true' market prices.
    """
    xi = kappa - rho * sigma_v * 1j * u
    d = np.sqrt((rho * sigma_v * 1j * u - xi)**2 - sigma_v**2 * (-u * 1j - u**2))
    g = (xi - d) / (xi + d)
    
    C = r * 1j * u * T + (kappa * theta) / sigma_v**2 * (
        (xi - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
    )
    D = (xi - d) / sigma_v**2 * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
    
    return np.exp(C + D * v0 + 1j * u * np.log(S0))

def heston_price_fourier(S0, K, T, r, kappa, theta, sigma_v, rho, v0):
    """
    Price a European call option using the Heston model via Fourier inversion.
    """
    # Integrand for P1
    integrand1 = lambda u: np.real(
        np.exp(-1j * u * np.log(K)) * heston_char_func(u - 1j, T, r, kappa, theta, sigma_v, rho, v0, S0) / 
        (1j * u * heston_char_func(-1j, T, r, kappa, theta, sigma_v, rho, v0, S0))
    )
    # Perform the integration for P1
    P1, _ = quad(integrand1, 1e-15, 100)
    P1 = 0.5 + (1 / np.pi) * P1

    # Integrand for P2
    integrand2 = lambda u: np.real(
        np.exp(-1j * u * np.log(K)) * heston_char_func(u, T, r, kappa, theta, sigma_v, rho, v0, S0) / (1j * u)
    )
    # Perform the integration for P2
    P2, _ = quad(integrand2, 1e-15, 100)
    P2 = 0.5 + (1 / np.pi) * P2
    
    return S0 * P1 - K * np.exp(-r * T) * P2

# --- Part 2: Implied Volatility Calculation ---

def black_scholes_call(S, K, T, r, sigma):
    """Standard Black-Scholes formula."""
    if sigma <= 1e-6:
        return np.maximum(0, S - K * np.exp(-r * T))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(market_price, S, K, T, r, initial_guess=0.2):
    """Calculate implied volatility using a simple solver."""
    def error_function(sigma):
        # Ensure sigma is a single float
        sigma_val = sigma[0] if isinstance(sigma, (np.ndarray, list)) else sigma
        return (black_scholes_call(S, K, T, r, sigma_val) - market_price)**2
    
    res = minimize(error_function, initial_guess, bounds=[(1e-5, 3.0)], method='L-BFGS-B')
    return res.x[0]

# --- Part 3: Competing Models for Calibration ---

def simulate_paths_euler(S0, r, T, params, model_type, num_paths, num_steps):
    """
    Simulates asset paths for Scott or Hull-White models using Euler-Maruyama.
    """
    dt = T / num_steps
    S = np.zeros((num_steps + 1, num_paths))
    S[0, :] = S0
    
    if model_type == 'scott':
        kappa, theta_y, sigma_y, rho = params
        Y = np.zeros((num_steps + 1, num_paths))
        Y[0, :] = theta_y # Start at long-run mean
        
        z1 = np.random.normal(size=(num_steps, num_paths))
        z2 = np.random.normal(size=(num_steps, num_paths))
        w1 = z1
        w2 = rho * z1 + np.sqrt(1 - rho**2) * z2
        
        for t in range(1, num_steps + 1):
            dY = kappa * (theta_y - Y[t-1, :]) * dt + sigma_y * np.sqrt(dt) * w2[t-1, :]
            Y[t, :] = Y[t-1, :] + dY
            vol = np.exp(Y[t-1, :])
            dS = r * S[t-1, :] * dt + vol * S[t-1, :] * np.sqrt(dt) * w1[t-1, :]
            S[t, :] = S[t-1, :] + dS
            
    elif model_type == 'hull_white':
        mu_v, sigma_v = params
        v = np.zeros((num_steps + 1, num_paths))
        v[0, :] = 0.04 # Start at a reasonable variance
        
        w1 = np.random.normal(size=(num_steps, num_paths))
        w2 = np.random.normal(size=(num_steps, num_paths))
        
        for t in range(1, num_steps + 1):
            dv = mu_v * v[t-1, :] * dt + sigma_v * v[t-1, :] * np.sqrt(dt) * w2[t-1, :]
            v[t, :] = np.maximum(0, v[t-1, :] + dv)
            vol = np.sqrt(v[t-1, :])
            dS = r * S[t-1, :] * dt + vol * S[t-1, :] * np.sqrt(dt) * w1[t-1, :]
            S[t, :] = S[t-1, :] + dS
            
    return S

def price_option_mc(S0, K, T, r, params, model_type, num_paths=10000, num_steps=100):
    """Price a European call option using Monte Carlo."""
    paths = simulate_paths_euler(S0, r, T, params, model_type, num_paths, num_steps)
    payoffs = np.maximum(paths[-1, :] - K, 0)
    price = np.mean(payoffs) * np.exp(-r * T)
    return price

# --- Part 4: The Simulation-Based "Horse Race" ---

if __name__ == '__main__':
    # --- Step 1: Define the "True" Market (Heston Model) ---
    S0 = 100.0
    r = 0.02
    T = 1.0
    
    v0_true = 0.04      
    kappa_true = 2.0    
    theta_true = 0.04   
    sigma_v_true = 0.3  
    rho_true = -0.6     
    
    print("--- Step 1: 'True' Market Parameters (Heston) ---")
    print(f"Initial Price: {S0}, Risk-Free Rate: {r*100:.2f}%")
    print(f"v0={v0_true}, kappa={kappa_true}, theta={theta_true}, sigma_v={sigma_v_true}, rho={rho_true}\n")

    # --- Step 2: Generate the Synthetic Volatility Surface ---
    strikes = np.arange(80, 121, 2)
    heston_prices = [heston_price_fourier(S0, k, T, r, kappa_true, theta_true, sigma_v_true, rho_true, v0_true) for k in strikes]
    market_ivs = [implied_volatility(price, S0, k, T, r) for price, k in zip(heston_prices, strikes)]
    
    print("--- Step 2: Generated Synthetic Market Smile (from Heston) ---")
    for k, iv in zip(strikes, market_ivs):
        print(f"Strike: {k}, Implied Vol: {iv*100:.2f}%")
    print("\n")

    # --- Step 3: Calibrate Competing Models ---
    
    def scott_calibration_objective(params):
        model_prices = [price_option_mc(S0, k, T, r, params, 'scott') for k in strikes]
        model_ivs = [implied_volatility(price, S0, k, T, r) for price, k in zip(model_prices, strikes)]
        return np.sum((np.array(model_ivs) - np.array(market_ivs))**2)

    def hull_white_calibration_objective(params):
        model_prices = [price_option_mc(S0, k, T, r, params, 'hull_white') for k in strikes]
        model_ivs = [implied_volatility(price, S0, k, T, r) for price, k in zip(model_prices, strikes)]
        return np.sum((np.array(model_ivs) - np.array(market_ivs))**2)

    print("--- Step 3: Calibrating Competing Models (This may take a few minutes) ---")
    
    initial_guess_scott = [1.5, np.log(0.2), 0.25, -0.5]
    bounds_scott = [(0.1, 5), (-3, -1), (0.01, 1), (-0.99, 0.99)]
    
    print("Calibrating Scott model...")
    scott_result = minimize(scott_calibration_objective, initial_guess_scott, bounds=bounds_scott, method='L-BFGS-B')
    calibrated_params_scott = scott_result.x
    print("Scott Calibration Complete.")
    print(f"Calibrated Scott Params: kappa={calibrated_params_scott[0]:.2f}, theta_y={calibrated_params_scott[1]:.2f}, sigma_y={calibrated_params_scott[2]:.2f}, rho={calibrated_params_scott[3]:.2f}\n")

    initial_guess_hw = [0.1, 0.3]
    bounds_hw = [(-0.5, 0.5), (0.01, 1.0)]
    
    print("Calibrating Hull-White model...")
    hw_result = minimize(hull_white_calibration_objective, initial_guess_hw, bounds=bounds_hw, method='L-BFGS-B')
    calibrated_params_hw = hw_result.x
    print("Hull-White Calibration Complete.")
    print(f"Calibrated Hull-White Params: mu_v={calibrated_params_hw[0]:.2f}, sigma_v={calibrated_params_hw[1]:.2f}\n")

    # --- Step 4: Analyze and Plot Results ---
    
    scott_prices_calibrated = [price_option_mc(S0, k, T, r, calibrated_params_scott, 'scott') for k in strikes]
    scott_ivs_calibrated = [implied_volatility(price, S0, k, T, r) for price, k in zip(scott_prices_calibrated, strikes)]
    
    hw_prices_calibrated = [price_option_mc(S0, k, T, r, calibrated_params_hw, 'hull_white') for k in strikes]
    hw_ivs_calibrated = [implied_volatility(price, S0, k, T, r) for price, k in zip(hw_prices_calibrated, strikes)]

    plt.figure(figsize=(12, 7))
    plt.plot(strikes, np.array(market_ivs) * 100, 'o-', label='"True" Market Smile (Heston)', linewidth=2, markersize=8)
    plt.plot(strikes, np.array(scott_ivs_calibrated) * 100, 's--', label='Best-Fit Scott Model Smile', linewidth=2)
    plt.plot(strikes, np.array(hw_ivs_calibrated) * 100, '^--', label='Best-Fit Hull-White Model Smile', linewidth=2)
    
    plt.title('Model Calibration Horse Race vs. Synthetic Heston Market', fontsize=16)
    plt.xlabel('Strike Price (K)', fontsize=12)
    plt.ylabel('Implied Volatility (%)', fontsize=12)
    plt.axvline(S0, color='gray', linestyle=':', label=f'At-the-Money (S0={S0})')
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
