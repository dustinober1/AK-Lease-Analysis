#!/usr/bin/env python3
"""
Financial Risk Modeling with Monte Carlo Simulation
Alaska OCS Oil & Gas Leases
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("📈 FINANCIAL RISK MODELING")
print("=" * 30)

# Set random seed for reproducibility
np.random.seed(42)

# Load processed data
df = pd.read_csv('data/AK_Leases.csv')
df['high_bid_dlrs'] = pd.to_numeric(df['BID_AMOUNT'], errors='coerce')
df['lease_size_acres'] = pd.to_numeric(df['CURRENT_AREA'], errors='coerce')
df['royalty_rate'] = pd.to_numeric(df['ROYALTY_RATE'], errors='coerce')

# Monte Carlo parameters
n_simulations = 10000
oil_price_scenarios = np.array([40, 60, 80, 100, 120])  # $/barrel scenarios
gas_price_scenarios = np.array([2, 3, 4, 5, 6])        # $/MMBtu scenarios

# Historical volatility assumptions (annualized)
oil_volatility = 0.35
gas_volatility = 0.45
correlation = 0.6

# Lease characteristics for modeling
avg_lease_size = df['lease_size_acres'].mean()
avg_bid_amount = df['high_bid_dlrs'].mean()
avg_royalty_rate = df['royalty_rate'].mean() / 100

print(f"🎲 Monte Carlo Simulation Parameters")
print(f"Number of Simulations: {n_simulations:,}")
print(f"Oil Price Volatility: {oil_volatility:.1%}")
print(f"Gas Price Volatility: {gas_volatility:.1%}")
print(f"Price Correlation: {correlation:.2f}")
print(f"Average Lease Size: {avg_lease_size:,.0f} acres")
print(f"Average Bid Amount: ${avg_bid_amount:,.0f}")
print(f"Average Royalty Rate: {avg_royalty_rate:.2%}")

def monte_carlo_lease_value(n_sims, base_oil_price=70, base_gas_price=3.5, lease_life_years=5):
    """Monte Carlo simulation for lease value under different price scenarios"""
    
    # Generate correlated price paths
    # Create correlated random variables
    random_matrix = np.random.multivariate_normal(
        [0, 0], 
        [[1, correlation], [correlation, 1]], 
        n_sims
    )
    
    # Generate price scenarios using geometric Brownian motion
    dt = 1/12  # Monthly steps
    n_steps = lease_life_years * 12
    
    oil_prices = np.zeros((n_sims, n_steps))
    gas_prices = np.zeros((n_sims, n_steps))
    
    oil_prices[:, 0] = base_oil_price
    gas_prices[:, 0] = base_gas_price
    
    for t in range(1, n_steps):
        # Oil price evolution
        oil_drift = -0.5 * oil_volatility**2 * dt
        oil_shock = oil_volatility * np.sqrt(dt) * random_matrix[:, 0]
        oil_prices[:, t] = oil_prices[:, t-1] * np.exp(oil_drift + oil_shock)
        
        # Gas price evolution
        gas_drift = -0.5 * gas_volatility**2 * dt
        gas_shock = gas_volatility * np.sqrt(dt) * random_matrix[:, 1]
        gas_prices[:, t] = gas_prices[:, t-1] * np.exp(gas_drift + gas_shock)
    
    # Calculate production estimates (simplified)
    # Assume production proportional to lease size and price levels
    production_factor = avg_lease_size / 5760  # Normalize by typical lease size
    
    oil_production = production_factor * 1000  # barrels per month
    gas_production = production_factor * 50    # MMBtu per month
    
    # Calculate revenue streams
    monthly_oil_revenue = oil_prices * oil_production
    monthly_gas_revenue = gas_prices * gas_production
    monthly_total_revenue = monthly_oil_revenue + monthly_gas_revenue
    
    # Calculate royalty payments
    monthly_royalties = monthly_total_revenue * avg_royalty_rate
    
    # Calculate net present value
    discount_rate = 0.10  # 10% annual discount rate
    monthly_discount_rate = discount_rate / 12
    
    # Discount factors
    time_periods = np.arange(1, n_steps + 1)
    discount_factors = 1 / (1 + monthly_discount_rate) ** time_periods
    
    # Calculate NPV for each simulation
    npv_royalties = np.sum(monthly_royalties * discount_factors, axis=1)
    total_lease_value = npv_royalties - avg_bid_amount  # Subtract initial investment
    
    return total_lease_value, oil_prices, gas_prices, monthly_royalties

# Run Monte Carlo simulation
print(f"\n🔄 Running Monte Carlo simulation...")
lease_values, oil_price_paths, gas_price_paths, royalty_streams = monte_carlo_lease_value(n_simulations)
print(f"✅ Simulation completed successfully!")

# Value at Risk (VaR) Analysis
print("\n⚠️ VALUE AT RISK (VaR) ANALYSIS")
print("=" * 35)

# Calculate VaR at different confidence levels
confidence_levels = [0.95, 0.99, 0.999]
var_results = {}

for conf_level in confidence_levels:
    alpha = 1 - conf_level
    var_value = np.percentile(lease_values, alpha * 100)
    var_results[conf_level] = var_value

# Expected Shortfall (Conditional VaR)
def expected_shortfall(values, confidence_level):
    alpha = 1 - confidence_level
    var = np.percentile(values, alpha * 100)
    return np.mean(values[values <= var])

es_results = {conf_level: expected_shortfall(lease_values, conf_level) 
              for conf_level in confidence_levels}

# Statistical summary
mean_value = np.mean(lease_values)
median_value = np.median(lease_values)
std_value = np.std(lease_values)
min_value = np.min(lease_values)
max_value = np.max(lease_values)

print(f"📊 Lease Value Distribution")
print(f"Mean Lease Value: ${mean_value:,.0f}")
print(f"Median Lease Value: ${median_value:,.0f}")
print(f"Standard Deviation: ${std_value:,.0f}")
print(f"Range: ${min_value:,.0f} to ${max_value:,.0f}")
print()
print(f"📉 Value at Risk (VaR) Results")
for conf_level in confidence_levels:
    print(f"VaR at {conf_level:.1%} confidence: ${var_results[conf_level]:,.0f}")
    print(f"Expected Shortfall at {conf_level:.1%}: ${es_results[conf_level]:,.0f}")
    print()

# Probability of profit
prob_profit = np.mean(lease_values > 0)
prob_loss = 1 - prob_profit

print(f"📈 Profitability Analysis")
print(f"Probability of Profit: {prob_profit:.1%}")
print(f"Probability of Loss: {prob_loss:.1%}")
print(f"Expected Return: ${mean_value:,.0f}")
print(f"Return on Investment: {mean_value/avg_bid_amount:.1%}")

# Break-even Analysis
print("\n⚖️ BREAK-EVEN ANALYSIS")
print("=" * 25)

def calculate_breakeven_price(bid_amount, lease_size, royalty_rate, production_years=5):
    """Calculate break-even oil price given lease parameters"""
    
    # Assumptions for break-even calculation
    monthly_production = (lease_size / 5760) * 1000  # barrels per month
    total_production = monthly_production * 12 * production_years
    
    # Operating costs (simplified)
    operating_cost_per_barrel = 25  # $/barrel
    total_operating_costs = total_production * operating_cost_per_barrel
    
    # Break-even calculation
    # Revenue needed = Bid amount + Operating costs
    required_revenue = bid_amount + total_operating_costs
    
    # Revenue = Oil price * Production * Royalty rate
    # Solving for oil price
    breakeven_price = required_revenue / (total_production * royalty_rate)
    
    return breakeven_price, total_production, total_operating_costs

# Calculate break-even for different lease types
lease_scenarios = {
    'Small Lease': {'bid': 50000, 'size': 2304, 'royalty': 0.125},
    'Average Lease': {'bid': avg_bid_amount, 'size': avg_lease_size, 'royalty': avg_royalty_rate},
    'Large Lease': {'bid': 5000000, 'size': 5760, 'royalty': 0.167},
    'Premium Lease': {'bid': 20000000, 'size': 5760, 'royalty': 0.167}
}

breakeven_results = {}
print(f"🎯 Break-even Oil Price Analysis")
print(f"{'Lease Type':<15} {'Bid Amount':<12} {'Lease Size':<12} {'Break-even':<12} {'Production':<12}")
print(f"{'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

for scenario_name, params in lease_scenarios.items():
    breakeven_price, production, operating_costs = calculate_breakeven_price(
        params['bid'], params['size'], params['royalty']
    )
    breakeven_results[scenario_name] = {
        'breakeven_price': breakeven_price,
        'production': production,
        'operating_costs': operating_costs
    }
    
    print(f"{scenario_name:<15} ${params['bid']:<11,.0f} {params['size']:<12,.0f} ${breakeven_price:<11.0f} {production:<12,.0f}")

print(f"\n💡 Break-even Insights")
avg_breakeven = np.mean([result['breakeven_price'] for result in breakeven_results.values()])
print(f"Average Break-even Price: ${avg_breakeven:.0f}/barrel")
print(f"Current Market Context: WTI ~$80/barrel (as of analysis date)")
viable_leases = sum(1 for result in breakeven_results.values() if result['breakeven_price'] < 80)
print(f"Viable Leases at $80/barrel: {viable_leases}/{len(lease_scenarios)}")

print("\n" + "=" * 50)
print("MONTE CARLO RISK ANALYSIS COMPLETE")
print("=" * 50)