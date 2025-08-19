"""
Visualization utility functions for Alaska OCS Lease Analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style defaults
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_summary_dashboard(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive summary dashboard
    
    Args:
        df: Input DataFrame
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Alaska OCS Lease Analysis - Summary Dashboard', fontsize=20, fontweight='bold')
    
    # 1. Lease status distribution
    status_counts = df['LEASE_IS_ACTIVE'].value_counts()
    axes[0,0].pie(status_counts.values, labels=['Inactive', 'Active'], autopct='%1.1f%%', 
                  colors=['lightcoral', 'lightgreen'])
    axes[0,0].set_title('Active vs Inactive Leases', fontsize=14, fontweight='bold')
    
    # 2. Top companies
    top_companies = df['BUS_ASC_NAME'].value_counts().head(8)
    axes[0,1].bar(range(len(top_companies)), top_companies.values, color='skyblue')
    axes[0,1].set_xticks(range(len(top_companies)))
    axes[0,1].set_xticklabels(top_companies.index, rotation=45, ha='right')
    axes[0,1].set_title('Top 8 Companies by Lease Count', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('Number of Leases')
    
    # 3. Planning areas
    planning_areas = df['MMS_PLAN_AREA_CD'].value_counts()
    axes[0,2].bar(planning_areas.index, planning_areas.values, color='lightsteelblue')
    axes[0,2].set_title('Leases by Planning Area', fontsize=14, fontweight='bold')
    axes[0,2].set_ylabel('Number of Leases')
    
    # 4. Bid amount distribution (log scale)
    bid_data = df['BID_AMOUNT'][df['BID_AMOUNT'] > 0]
    axes[1,0].hist(np.log10(bid_data), bins=30, alpha=0.7, color='gold', edgecolor='black')
    axes[1,0].set_title('Log10(Bid Amount) Distribution', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Log10(Bid Amount)')
    axes[1,0].set_ylabel('Frequency')
    
    # 5. Sales over time
    df['SALE_YEAR'] = pd.to_datetime(df['SALE_DATE'], errors='coerce').dt.year
    yearly_sales = df['SALE_YEAR'].value_counts().sort_index()
    axes[1,1].plot(yearly_sales.index, yearly_sales.values, marker='o', linewidth=2, markersize=4, color='darkgreen')
    axes[1,1].set_title('Lease Sales by Year', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Year')
    axes[1,1].set_ylabel('Number of Leases')
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Area vs Bid relationship
    valid_data = df[(df['CURRENT_AREA'] > 0) & (df['BID_AMOUNT'] > 0)]
    axes[1,2].scatter(valid_data['CURRENT_AREA'], valid_data['BID_AMOUNT'], 
                      alpha=0.6, s=20, color='purple')
    axes[1,2].set_xscale('log')
    axes[1,2].set_yscale('log')
    axes[1,2].set_title('Lease Area vs Bid Amount', fontsize=14, fontweight='bold')
    axes[1,2].set_xlabel('Area (Hectares)')
    axes[1,2].set_ylabel('Bid Amount ($)')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_temporal_analysis_plots(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Create temporal analysis visualizations
    
    Args:
        df: Input DataFrame
        save_path: Optional path to save the figure
    """
    # Prepare temporal data
    df['SALE_YEAR'] = pd.to_datetime(df['SALE_DATE'], errors='coerce').dt.year
    df['DECADE'] = (df['SALE_YEAR'] // 10) * 10
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Temporal Analysis of Alaska OCS Leases', fontsize=18, fontweight='bold')
    
    # 1. Yearly trends
    yearly_stats = df.groupby('SALE_YEAR').agg({
        'LEASE_NUMBER': 'count',
        'BID_AMOUNT': 'sum',
        'CURRENT_AREA': 'sum'
    }).reset_index()
    
    axes[0,0].plot(yearly_stats['SALE_YEAR'], yearly_stats['LEASE_NUMBER'], 
                   marker='o', linewidth=2, color='blue')
    axes[0,0].set_title('Number of Leases by Year', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Year')
    axes[0,0].set_ylabel('Number of Leases')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Total bid value by year
    axes[0,1].plot(yearly_stats['SALE_YEAR'], yearly_stats['BID_AMOUNT'] / 1e6, 
                   marker='s', linewidth=2, color='green')
    axes[0,1].set_title('Total Bid Value by Year', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Year')
    axes[0,1].set_ylabel('Total Bid Amount ($ Millions)')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Decade comparison
    decade_counts = df['DECADE'].value_counts().sort_index()
    axes[1,0].bar(decade_counts.index.astype(str), decade_counts.values, 
                  color='orange', alpha=0.7, edgecolor='black')
    axes[1,0].set_title('Leases by Decade', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Decade')
    axes[1,0].set_ylabel('Number of Leases')
    
    # 4. Average bid over time
    avg_bid_by_year = df.groupby('SALE_YEAR')['BID_AMOUNT'].mean()
    axes[1,1].plot(avg_bid_by_year.index, avg_bid_by_year.values, 
                   marker='D', linewidth=2, color='red')
    axes[1,1].set_title('Average Bid Amount by Year', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Year')
    axes[1,1].set_ylabel('Average Bid Amount ($)')
    axes[1,1].set_yscale('log')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_correlation_heatmap(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Create correlation heatmap for numerical variables
    
    Args:
        df: Input DataFrame
        save_path: Optional path to save the figure
    """
    # Select numerical columns
    numerical_cols = ['BID_AMOUNT', 'CURRENT_AREA', 'ROYALTY_RATE', 'PRIMARY_TERM']
    
    # Add derived features if they exist
    if 'BID_PER_HECTARE' in df.columns:
        numerical_cols.append('BID_PER_HECTARE')
    if 'SALE_YEAR' in df.columns:
        numerical_cols.append('SALE_YEAR')
    if 'IS_ACTIVE' in df.columns:
        numerical_cols.append('IS_ACTIVE')
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Key Variables', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_company_analysis_plot(df: pd.DataFrame, top_n: int = 10, save_path: Optional[str] = None) -> None:
    """
    Create company analysis visualization
    
    Args:
        df: Input DataFrame
        top_n: Number of top companies to show
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Top {top_n} Companies Analysis', fontsize=18, fontweight='bold')
    
    # Get top companies
    top_companies = df['BUS_ASC_NAME'].value_counts().head(top_n)
    
    # 1. Lease count by company
    axes[0,0].barh(range(len(top_companies)), top_companies.values, color='steelblue')
    axes[0,0].set_yticks(range(len(top_companies)))
    axes[0,0].set_yticklabels(top_companies.index)
    axes[0,0].set_title('Number of Leases by Company', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Number of Leases')
    
    # 2. Total bid value by company
    company_bids = df.groupby('BUS_ASC_NAME')['BID_AMOUNT'].sum().sort_values(ascending=False).head(top_n)
    axes[0,1].barh(range(len(company_bids)), company_bids.values / 1e6, color='darkgreen')
    axes[0,1].set_yticks(range(len(company_bids)))
    axes[0,1].set_yticklabels(company_bids.index)
    axes[0,1].set_title('Total Bid Value by Company', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Total Bid Amount ($ Millions)')
    
    # 3. Average bid by company
    company_avg_bids = df.groupby('BUS_ASC_NAME')['BID_AMOUNT'].mean().sort_values(ascending=False).head(top_n)
    axes[1,0].barh(range(len(company_avg_bids)), company_avg_bids.values / 1e3, color='orange')
    axes[1,0].set_yticks(range(len(company_avg_bids)))
    axes[1,0].set_yticklabels(company_avg_bids.index)
    axes[1,0].set_title('Average Bid Amount by Company', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Average Bid Amount ($ Thousands)')
    
    # 4. Active lease percentage by company
    company_activity = df.groupby('BUS_ASC_NAME').agg({
        'LEASE_IS_ACTIVE': lambda x: (x == 'Y').sum() / len(x) * 100
    }).sort_values('LEASE_IS_ACTIVE', ascending=False).head(top_n)
    
    axes[1,1].barh(range(len(company_activity)), company_activity['LEASE_IS_ACTIVE'].values, color='red')
    axes[1,1].set_yticks(range(len(company_activity)))
    axes[1,1].set_yticklabels(company_activity.index)
    axes[1,1].set_title('Active Lease Percentage by Company', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Active Lease Percentage (%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_interactive_plotly_dashboard(df: pd.DataFrame, save_path: Optional[str] = None) -> go.Figure:
    """
    Create interactive Plotly dashboard
    
    Args:
        df: Input DataFrame
        save_path: Optional path to save the HTML file
        
    Returns:
        Plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Lease Status Distribution', 'Bid Amount vs Area', 
                       'Sales Over Time', 'Planning Area Distribution'),
        specs=[[{"type": "pie"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # 1. Pie chart for lease status
    status_counts = df['LEASE_IS_ACTIVE'].value_counts()
    fig.add_trace(go.Pie(labels=['Inactive', 'Active'], values=status_counts.values,
                        name="Lease Status"), row=1, col=1)
    
    # 2. Scatter plot: Bid amount vs Area
    valid_data = df[(df['CURRENT_AREA'] > 0) & (df['BID_AMOUNT'] > 0)]
    fig.add_trace(go.Scatter(x=valid_data['CURRENT_AREA'], y=valid_data['BID_AMOUNT'],
                            mode='markers', name='Leases',
                            hovertemplate='Area: %{x}<br>Bid: $%{y}<extra></extra>'),
                  row=1, col=2)
    
    # 3. Time series of sales
    df['SALE_YEAR'] = pd.to_datetime(df['SALE_DATE'], errors='coerce').dt.year
    yearly_sales = df['SALE_YEAR'].value_counts().sort_index()
    fig.add_trace(go.Scatter(x=yearly_sales.index, y=yearly_sales.values,
                            mode='lines+markers', name='Sales per Year'),
                  row=2, col=1)
    
    # 4. Bar chart for planning areas
    planning_areas = df['MMS_PLAN_AREA_CD'].value_counts()
    fig.add_trace(go.Bar(x=planning_areas.index, y=planning_areas.values,
                        name='Planning Areas'), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title_text="Alaska OCS Lease Analysis - Interactive Dashboard",
        showlegend=False,
        height=800
    )
    
    # Update axes
    fig.update_xaxes(type="log", title_text="Area (Hectares)", row=1, col=2)
    fig.update_yaxes(type="log", title_text="Bid Amount ($)", row=1, col=2)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="Number of Leases", row=2, col=1)
    fig.update_xaxes(title_text="Planning Area", row=2, col=2)
    fig.update_yaxes(title_text="Number of Leases", row=2, col=2)
    
    if save_path:
        fig.write_html(save_path)
    
    return fig

def create_statistical_distribution_plots(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Create statistical distribution analysis plots
    
    Args:
        df: Input DataFrame
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Statistical Distribution Analysis', fontsize=18, fontweight='bold')
    
    # 1. Bid amount distribution
    bid_data = df['BID_AMOUNT'][df['BID_AMOUNT'] > 0]
    axes[0,0].hist(np.log10(bid_data), bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0,0].axvline(np.log10(bid_data.median()), color='red', linestyle='--', 
                      label=f'Median: ${bid_data.median():,.0f}')
    axes[0,0].set_title('Log10(Bid Amount) Distribution', fontsize=14)
    axes[0,0].set_xlabel('Log10(Bid Amount)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].legend()
    
    # 2. Area distribution
    area_data = df['CURRENT_AREA'][df['CURRENT_AREA'] > 0]
    axes[0,1].hist(area_data, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0,1].axvline(area_data.median(), color='red', linestyle='--',
                      label=f'Median: {area_data.median():.0f} ha')
    axes[0,1].set_title('Lease Area Distribution', fontsize=14)
    axes[0,1].set_xlabel('Area (Hectares)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()
    
    # 3. Royalty rate distribution
    axes[0,2].hist(df['ROYALTY_RATE'].dropna(), bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[0,2].set_title('Royalty Rate Distribution', fontsize=14)
    axes[0,2].set_xlabel('Royalty Rate (%)')
    axes[0,2].set_ylabel('Frequency')
    
    # 4. Box plot comparison: Active vs Inactive bids
    active_bids = df[df['LEASE_IS_ACTIVE'] == 'Y']['BID_AMOUNT']
    inactive_bids = df[df['LEASE_IS_ACTIVE'] == 'N']['BID_AMOUNT']
    
    box_data = [active_bids[active_bids > 0], inactive_bids[inactive_bids > 0]]
    axes[1,0].boxplot(box_data, labels=['Active', 'Inactive'])
    axes[1,0].set_yscale('log')
    axes[1,0].set_title('Bid Amount: Active vs Inactive', fontsize=14)
    axes[1,0].set_ylabel('Bid Amount ($)')
    
    # 5. Primary term distribution
    axes[1,1].hist(df['PRIMARY_TERM'].dropna(), bins=15, alpha=0.7, color='purple', edgecolor='black')
    axes[1,1].set_title('Primary Term Distribution', fontsize=14)
    axes[1,1].set_xlabel('Primary Term (Years)')
    axes[1,1].set_ylabel('Frequency')
    
    # 6. Q-Q plot for bid amounts
    from scipy import stats
    bid_sample = np.log10(bid_data.sample(min(1000, len(bid_data)), random_state=42))
    stats.probplot(bid_sample, dist="norm", plot=axes[1,2])
    axes[1,2].set_title('Q-Q Plot: Log10(Bid Amount) vs Normal', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()