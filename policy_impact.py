#!/usr/bin/env python3
"""
Policy Impact Quantification Analysis
Alaska OCS Oil & Gas Leases
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

print("🏛️ POLICY IMPACT QUANTIFICATION")
print("=" * 35)

# Load processed data
df = pd.read_csv('data/AK_Leases.csv')
df['lease_date'] = pd.to_datetime(df['SALE_DATE'], errors='coerce')
df['year'] = df['lease_date'].dt.year
df['high_bid_dlrs'] = pd.to_numeric(df['BID_AMOUNT'], errors='coerce')
df['lease_size_acres'] = pd.to_numeric(df['CURRENT_AREA'], errors='coerce')

# Filter out invalid years
df = df[(df['year'] >= 1970) & (df['year'] <= 2025)]

# Economic multiplier
TOTAL_MULTIPLIER = 2.4  # From previous analysis
JOBS_PER_MILLION = 8.5

# Identify policy periods based on major regulatory changes
policy_periods = {
    'Early Era (1976-1989)': (1976, 1989),
    'OCSLA Amendment (1990-1999)': (1990, 1999),
    'Modern Era (2000-2009)': (2000, 2009),
    'Obama Policies (2010-2016)': (2010, 2016),
    'Recent Era (2017-2024)': (2017, 2024)
}

# Calculate metrics for each policy period
policy_analysis = {}

for period_name, (start_year, end_year) in policy_periods.items():
    period_data = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
    
    if len(period_data) > 0:
        policy_analysis[period_name] = {
            'lease_count': len(period_data),
            'total_bids': period_data['high_bid_dlrs'].sum(),
            'avg_bid': period_data['high_bid_dlrs'].mean(),
            'median_bid': period_data['high_bid_dlrs'].median(),
            'total_acres': period_data['lease_size_acres'].sum(),
            'years_span': end_year - start_year + 1,
            'annual_avg_bids': period_data['high_bid_dlrs'].sum() / (end_year - start_year + 1)
        }

# Convert to DataFrame for easier analysis
policy_df = pd.DataFrame(policy_analysis).T
policy_df = policy_df.round(2)

print(f"📊 Policy Period Analysis")
print(policy_df[['lease_count', 'total_bids', 'avg_bid', 'annual_avg_bids']].to_string())

print(f"\n📈 Policy Impact Metrics")
print(f"Highest Activity Period: {policy_df['annual_avg_bids'].idxmax()}")
print(f"Peak Annual Average: ${policy_df['annual_avg_bids'].max():,.0f}")
print(f"Lowest Activity Period: {policy_df['annual_avg_bids'].idxmin()}")
print(f"Minimum Annual Average: ${policy_df['annual_avg_bids'].min():,.0f}")

# Calculate policy impact ratios
baseline_period = 'Early Era (1976-1989)'
if baseline_period in policy_df.index:
    baseline_activity = policy_df.loc[baseline_period, 'annual_avg_bids']
    print(f"\n🔄 Policy Impact Ratios (vs {baseline_period})")
    for period in policy_df.index:
        if period != baseline_period:
            ratio = policy_df.loc[period, 'annual_avg_bids'] / baseline_activity
            print(f"{period}: {ratio:.1f}x baseline activity")

# Economic Impact of Moratorium Periods
print("\n🚫 MORATORIUM IMPACT ANALYSIS")
print("=" * 32)

# Identify low-activity periods
annual_activity = df.groupby('year').agg({
    'high_bid_dlrs': 'sum',
    'lease_size_acres': 'sum'
}).reset_index()

annual_activity['annual_bids_millions'] = annual_activity['high_bid_dlrs'] / 1e6

# Define moratorium threshold (years with <10% of peak activity)
peak_activity = annual_activity['high_bid_dlrs'].max()
moratorium_threshold = peak_activity * 0.1

low_activity_years = annual_activity[annual_activity['high_bid_dlrs'] < moratorium_threshold]['year'].tolist()
high_activity_years = annual_activity[annual_activity['high_bid_dlrs'] >= moratorium_threshold]['year'].tolist()

# Calculate economic impact of reduced activity
avg_high_activity = annual_activity[annual_activity['high_bid_dlrs'] >= moratorium_threshold]['high_bid_dlrs'].mean()
avg_low_activity = annual_activity[annual_activity['high_bid_dlrs'] < moratorium_threshold]['high_bid_dlrs'].mean()

annual_opportunity_cost = avg_high_activity - avg_low_activity
total_low_activity_years = len(low_activity_years)
total_opportunity_cost = annual_opportunity_cost * total_low_activity_years

print(f"📉 Moratorium Period Analysis")
print(f"Peak Annual Activity: ${peak_activity:,.0f}")
print(f"Moratorium Threshold (<10% of peak): ${moratorium_threshold:,.0f}")
print(f"Low Activity Years: {len(low_activity_years)} (examples: {low_activity_years[:10]})")
print(f"High Activity Years: {len(high_activity_years)}")
print()
print(f"💰 Economic Impact")
print(f"Average High Activity Year: ${avg_high_activity:,.0f}")
print(f"Average Low Activity Year: ${avg_low_activity:,.0f}")
print(f"Annual Opportunity Cost: ${annual_opportunity_cost:,.0f}")
print(f"Total Estimated Opportunity Cost: ${total_opportunity_cost:,.0f}")
print(f"Total Economic Impact Lost: ${total_opportunity_cost * TOTAL_MULTIPLIER:,.0f}")

# Jobs impact
jobs_impact = (total_opportunity_cost / 1_000_000) * JOBS_PER_MILLION
print(f"Estimated Jobs Impact: {jobs_impact:,.0f} jobs")

# Regulatory Cost-Benefit Analysis
print("\n⚖️ REGULATORY COST-BENEFIT ANALYSIS")
print("=" * 38)

avg_bid_amount = df['high_bid_dlrs'].mean()

# Environmental regulation scenarios
regulation_scenarios = {
    'Baseline (Current)': {'environmental_cost': 0, 'delay_months': 0, 'success_rate': 0.15},
    'Enhanced Environmental': {'environmental_cost': 50000, 'delay_months': 6, 'success_rate': 0.12},
    'Streamlined Process': {'environmental_cost': 25000, 'delay_months': -3, 'success_rate': 0.18},
    'Maximum Protection': {'environmental_cost': 150000, 'delay_months': 12, 'success_rate': 0.08}
}

# Calculate net present value impact for each scenario
discount_rate_annual = 0.08

regulation_impact = {}

for scenario, params in regulation_scenarios.items():
    # Calculate impact on lease economics
    additional_costs = params['environmental_cost']
    time_delay_impact = (params['delay_months'] / 12) * discount_rate_annual * avg_bid_amount
    success_rate_multiplier = params['success_rate'] / regulation_scenarios['Baseline (Current)']['success_rate']
    
    # Adjusted lease value
    adjusted_lease_value = (avg_bid_amount - additional_costs - time_delay_impact) * success_rate_multiplier
    
    # Environmental benefits (monetized - simplified)
    environmental_benefit_per_lease = params['environmental_cost'] * 2  # Assume 2:1 benefit ratio
    
    # Net social value
    net_social_value = adjusted_lease_value + environmental_benefit_per_lease - additional_costs
    
    regulation_impact[scenario] = {
        'adjusted_lease_value': adjusted_lease_value,
        'environmental_benefit': environmental_benefit_per_lease,
        'additional_costs': additional_costs,
        'time_delay_cost': time_delay_impact,
        'net_social_value': net_social_value,
        'success_rate': params['success_rate']
    }

# Convert to DataFrame
regulation_df = pd.DataFrame(regulation_impact).T
regulation_df = regulation_df.round(0)

print(f"📊 Regulatory Scenario Analysis (Per Average Lease)")
print(regulation_df[['adjusted_lease_value', 'environmental_benefit', 'additional_costs', 'net_social_value']].to_string())

# Optimal policy recommendation
optimal_scenario = regulation_df['net_social_value'].idxmax()
print(f"\n🎯 Policy Recommendation")
print(f"Optimal Scenario: {optimal_scenario}")
print(f"Net Social Value: ${regulation_df.loc[optimal_scenario, 'net_social_value']:,.0f}")
print(f"Success Rate: {regulation_df.loc[optimal_scenario, 'success_rate']:.1%}")

# Revenue Optimization Model
print("\n📈 REVENUE OPTIMIZATION MODEL")
print("=" * 32)

# Historical oil price approximations (simplified)
oil_price_history = {
    1980: 35, 1985: 28, 1990: 24, 1995: 18, 2000: 30, 
    2005: 56, 2010: 80, 2015: 48, 2020: 40, 2024: 80
}

# Match lease years with approximate oil prices
df_price = df.copy()
df_price['oil_price_estimate'] = df_price['year'].map(
    lambda x: min(oil_price_history.keys(), key=lambda k: abs(k-x)) if not pd.isna(x) else None
).map(oil_price_history)

# Analyze bid sensitivity to oil prices
price_correlation_data = df_price[['high_bid_dlrs', 'oil_price_estimate']].dropna()
if len(price_correlation_data) > 0:
    price_correlation = price_correlation_data.corr().iloc[0,1]
else:
    price_correlation = 0.5

# Optimal timing analysis
price_bins = [0, 30, 50, 70, 100, 150]
price_labels = ['<$30', '$30-50', '$50-70', '$70-100', '>$100']
df_price['price_bin'] = pd.cut(df_price['oil_price_estimate'], bins=price_bins, labels=price_labels, include_lowest=True)

price_analysis = df_price.groupby('price_bin').agg({
    'high_bid_dlrs': ['count', 'sum', 'mean', 'median'],
    'lease_size_acres': 'sum'
}).round(0)

price_analysis.columns = ['Lease_Count', 'Total_Bids', 'Mean_Bid', 'Median_Bid', 'Total_Acres']
price_analysis['Revenue_per_Acre'] = price_analysis['Total_Bids'] / price_analysis['Total_Acres']

print(f"🛢️ Oil Price Impact on Lease Revenue")
print(f"Price-Bid Correlation: {price_correlation:.3f}")
print()
print(price_analysis.to_string())

# Revenue optimization recommendations
if not price_analysis.empty and not price_analysis['Mean_Bid'].isna().all():
    optimal_price_range = price_analysis['Mean_Bid'].idxmax()
    revenue_multiplier = price_analysis.loc[optimal_price_range, 'Mean_Bid'] / price_analysis['Mean_Bid'].min()
    
    print(f"\n🎯 Revenue Optimization Insights")
    print(f"Optimal Price Range for Lease Sales: {optimal_price_range}")
    print(f"Revenue Multiplier vs Worst Timing: {revenue_multiplier:.1f}x")
    print(f"Average Bid in Optimal Range: ${price_analysis.loc[optimal_price_range, 'Mean_Bid']:,.0f}")

# Future scenario modeling
future_scenarios = {'Conservative': 60, 'Moderate': 80, 'Optimistic': 100}
print(f"\n🔮 Future Revenue Projections (Next Lease Sale)")
baseline_bid = df['high_bid_dlrs'].mean()
for scenario, oil_price in future_scenarios.items():
    # Simple linear relationship assumption
    price_factor = oil_price / 70  # Normalize to $70 baseline
    projected_avg_bid = baseline_bid * price_factor
    
    # Assume 100 leases in next sale
    projected_revenue = projected_avg_bid * 100
    projected_economic_impact = projected_revenue * TOTAL_MULTIPLIER
    
    print(f"{scenario} (${oil_price}/barrel): ${projected_revenue:,.0f} revenue, ${projected_economic_impact:,.0f} economic impact")

print("\n" + "=" * 50)
print("POLICY IMPACT ANALYSIS COMPLETE")
print("=" * 50)