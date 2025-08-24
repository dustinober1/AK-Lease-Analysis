#!/usr/bin/env python3
"""
Economic Impact & Market Dynamics Analysis
Alaska OCS Oil & Gas Leases
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import datetime as dt
from collections import Counter

warnings.filterwarnings('ignore')

print("📊 Economic Impact & Market Dynamics Analysis")
print("=" * 50)
print(f"Analysis Date: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Libraries loaded successfully!\n")

# Load the lease data
df = pd.read_csv('data/AK_Leases.csv')

# Data preprocessing for economic analysis
df['lease_date'] = pd.to_datetime(df['SALE_DATE'], errors='coerce')
df['year'] = df['lease_date'].dt.year
df['decade'] = (df['year'] // 10) * 10

# Clean and prepare financial data
financial_cols = ['BID_AMOUNT', 'ROYALTY_RATE', 'CURRENT_AREA']
for col in financial_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Create standardized column names for analysis
df['high_bid_dlrs'] = df['BID_AMOUNT']
df['royalty_rate'] = df['ROYALTY_RATE']
df['lease_size_acres'] = df['CURRENT_AREA']
df['planning_area'] = df['PROT_NAME']
df['lease_status'] = df['LEASE_STATUS_CD']
df['business_associate_name'] = df['BUS_ASC_NAME']

# Create economic impact variables
df['bid_per_acre'] = df['high_bid_dlrs'] / df['lease_size_acres'].replace(0, np.nan)
df['rental_rate'] = 50  # Estimated rental rate per acre per year
df['total_lease_value'] = df['high_bid_dlrs'] + (df['rental_rate'] * df['lease_size_acres'] * 5)

print(f"📈 Dataset Overview")
print(f"Total Records: {len(df):,}")
print(f"Date Range: {df['year'].min():.0f} - {df['year'].max():.0f}")
print(f"Total Bid Amount: ${df['high_bid_dlrs'].sum():,.0f}")
print(f"Active Leases: {df[df['lease_status'] == 'Active'].shape[0]:,} ({df[df['lease_status'] == 'Active'].shape[0]/len(df)*100:.1f}%)\n")

# Economic Impact Calculations
print("💰 ECONOMIC IMPACT ASSESSMENT")
print("=" * 40)

# Total economic metrics
total_bids = df['high_bid_dlrs'].sum()
total_acres = df['lease_size_acres'].sum()
avg_bid_per_acre = df['bid_per_acre'].mean()
median_lease_value = df['total_lease_value'].median()

# Economic multiplier assumptions (based on industry studies)
DIRECT_MULTIPLIER = 1.0
INDIRECT_MULTIPLIER = 0.8  # Indirect economic activity
INDUCED_MULTIPLIER = 0.6   # Induced spending
TOTAL_MULTIPLIER = DIRECT_MULTIPLIER + INDIRECT_MULTIPLIER + INDUCED_MULTIPLIER

# Calculate economic impacts
direct_impact = total_bids
indirect_impact = total_bids * INDIRECT_MULTIPLIER
induced_impact = total_bids * INDUCED_MULTIPLIER
total_economic_impact = direct_impact * TOTAL_MULTIPLIER

# Job creation estimates (jobs per $1M investment)
JOBS_PER_MILLION = 8.5  # Industry average for oil & gas sector
estimated_jobs = (total_bids / 1_000_000) * JOBS_PER_MILLION

print(f"📊 Total Economic Impact Analysis")
print(f"Direct Impact (Lease Payments): ${direct_impact:,.0f}")
print(f"Indirect Impact (Supply Chain): ${indirect_impact:,.0f}")
print(f"Induced Impact (Spending): ${induced_impact:,.0f}")
print(f"TOTAL ECONOMIC IMPACT: ${total_economic_impact:,.0f}")
print()
print(f"👷 Job Creation Estimates")
print(f"Estimated Jobs Created: {estimated_jobs:,.0f}")
print(f"Jobs per Million Invested: {JOBS_PER_MILLION}")
print()
print(f"📏 Per-Acre Economics")
print(f"Average Bid per Acre: ${avg_bid_per_acre:,.2f}")
print(f"Total Acres Under Lease: {total_acres:,.0f}")
print(f"Median Total Lease Value: ${median_lease_value:,.0f}\n")

# Economic Impact by Planning Area
area_impact = df.groupby('planning_area').agg({
    'high_bid_dlrs': ['sum', 'mean', 'count'],
    'lease_size_acres': 'sum',
    'bid_per_acre': 'mean'
}).round(2)

area_impact.columns = ['Total_Bids', 'Avg_Bid', 'Lease_Count', 'Total_Acres', 'Avg_Bid_Per_Acre']
area_impact['Economic_Impact'] = area_impact['Total_Bids'] * TOTAL_MULTIPLIER
area_impact['Est_Jobs'] = (area_impact['Total_Bids'] / 1_000_000) * JOBS_PER_MILLION
area_impact = area_impact.sort_values('Economic_Impact', ascending=False)

print("🗺️ ECONOMIC IMPACT BY PLANNING AREA")
print("=" * 50)
print(area_impact.head(8).to_string())
print()

# Revenue Analysis: Federal vs State
print("💵 REVENUE SHARING ANALYSIS")
print("=" * 35)

# Federal revenue (bonus bids)
federal_bonus_revenue = total_bids

# Estimated production royalties (assuming average production over lease life)
avg_royalty_rate = df['royalty_rate'].mean() / 100  # Convert to decimal
estimated_production_value = total_bids * 5  # Conservative multiplier
estimated_royalty_revenue = estimated_production_value * avg_royalty_rate

# Alaska state share (varies by area, using approximate 50% for federal waters adjacent to state)
STATE_SHARE_BONUS = 0.0  # Federal waters
STATE_SHARE_ROYALTY = 0.27  # Alaska gets 27% of federal royalties in adjacent waters

state_bonus_revenue = federal_bonus_revenue * STATE_SHARE_BONUS
state_royalty_revenue = estimated_royalty_revenue * STATE_SHARE_ROYALTY
federal_royalty_revenue = estimated_royalty_revenue * (1 - STATE_SHARE_ROYALTY)

total_federal_revenue = federal_bonus_revenue + federal_royalty_revenue
total_state_revenue = state_bonus_revenue + state_royalty_revenue

print(f"Federal Government Revenue:")
print(f"  Bonus Payments: ${federal_bonus_revenue:,.0f}")
print(f"  Royalty Share: ${federal_royalty_revenue:,.0f}")
print(f"  TOTAL FEDERAL: ${total_federal_revenue:,.0f}")
print()
print(f"Alaska State Revenue:")
print(f"  Bonus Share: ${state_bonus_revenue:,.0f}")
print(f"  Royalty Share: ${state_royalty_revenue:,.0f}")
print(f"  TOTAL STATE: ${total_state_revenue:,.0f}")
print()
print(f"📈 Revenue Projections")
print(f"Average Royalty Rate: {avg_royalty_rate:.2%}")
print(f"Estimated Production Value: ${estimated_production_value:,.0f}")
print(f"Projected Total Royalties: ${estimated_royalty_revenue:,.0f}\n")

# Market Concentration Analysis
print("🏢 MARKET DYNAMICS & COMPETITION ANALYSIS")
print("=" * 45)

# Company participation analysis
company_analysis = df.groupby('business_associate_name').agg({
    'high_bid_dlrs': ['sum', 'count', 'mean'],
    'lease_size_acres': 'sum'
}).round(2)

company_analysis.columns = ['Total_Bids', 'Lease_Count', 'Avg_Bid', 'Total_Acres']
company_analysis['Market_Share_Bids'] = (company_analysis['Total_Bids'] / total_bids * 100)
company_analysis['Market_Share_Acres'] = (company_analysis['Total_Acres'] / total_acres * 100)
company_analysis = company_analysis.sort_values('Total_Bids', ascending=False)

# Calculate Herfindahl-Hirschman Index (HHI) for market concentration
market_shares = company_analysis['Market_Share_Bids'].values
hhi = np.sum(market_shares ** 2)

# Market concentration interpretation
if hhi < 1500:
    concentration_level = "Unconcentrated (Competitive)"
elif hhi < 2500:
    concentration_level = "Moderately Concentrated"
else:
    concentration_level = "Highly Concentrated"

print(f"📊 Market Concentration Metrics")
print(f"Herfindahl-Hirschman Index (HHI): {hhi:.1f}")
print(f"Market Concentration Level: {concentration_level}")
print(f"Number of Active Companies: {len(company_analysis)}")
print()
print(f"🔝 Top 10 Companies by Total Bids")
print(company_analysis.head(10)[['Total_Bids', 'Lease_Count', 'Market_Share_Bids', 'Avg_Bid']].to_string())
print()

# Competitive vs Non-Competitive Lease Analysis
print("⚔️ COMPETITIVE BIDDING ANALYSIS")
print("=" * 35)

# Simulate competitive intensity based on bid amounts and market activity
df['bid_percentile'] = df['high_bid_dlrs'].rank(pct=True)
df['competitive_intensity'] = np.where(df['bid_percentile'] > 0.75, 'High Competition',
                             np.where(df['bid_percentile'] > 0.5, 'Moderate Competition', 
                                     'Low Competition'))

competition_analysis = df.groupby('competitive_intensity').agg({
    'high_bid_dlrs': ['count', 'sum', 'mean', 'median'],
    'lease_size_acres': 'sum',
    'bid_per_acre': 'mean'
}).round(2)

competition_analysis.columns = ['Lease_Count', 'Total_Bids', 'Mean_Bid', 'Median_Bid', 'Total_Acres', 'Avg_Bid_Per_Acre']

print("📈 Bidding Intensity Analysis")
print(competition_analysis.to_string())

# Price discovery analysis
print(f"\n💰 Price Discovery Metrics")
high_comp_premium = competition_analysis.loc['High Competition', 'Mean_Bid'] / competition_analysis.loc['Low Competition', 'Mean_Bid']
print(f"High Competition Premium: {high_comp_premium:.1f}x")
print(f"Bid Range: ${df['high_bid_dlrs'].min():,.0f} - ${df['high_bid_dlrs'].max():,.0f}")
print(f"Coefficient of Variation: {df['high_bid_dlrs'].std() / df['high_bid_dlrs'].mean():.2f}\n")

print("🎯 STRATEGIC BIDDING PATTERNS")
print("=" * 32)

# Block adjacency analysis (simplified)
clustering_behavior = df.groupby(['planning_area', 'year', 'business_associate_name']).size().reset_index(name='leases_same_area_year')
avg_clustering = clustering_behavior.groupby('business_associate_name')['leases_same_area_year'].mean().sort_values(ascending=False)

print(f"📍 Geographic Clustering Behavior (Top 10)")
print("Average leases per company per area-year:")
for company, avg_leases in avg_clustering.head(10).items():
    print(f"  {company[:30]:<30}: {avg_leases:.1f}")

# Portfolio optimization indicators
print(f"\n📊 Portfolio Diversification Analysis")
top_companies = company_analysis.head(10).index
for company in top_companies[:5]:
    company_data = df[df['business_associate_name'] == company]
    area_diversity = len(company_data['planning_area'].unique())
    year_span = company_data['year'].max() - company_data['year'].min()
    print(f"{company[:25]:<25}: {area_diversity} areas, {year_span} year span")

print("\n" + "=" * 50)
print("ECONOMIC ANALYSIS COMPLETE")
print("=" * 50)