import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_distributions(df, cols, log_scale=False):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Distribution Analysis of Key Variables', fontsize=16)
    for i, col in enumerate(cols):
        ax = axes[i//3, i%3]
        data = df[col].dropna()
        if log_scale:
            data = data[data > 0]
            ax.hist(np.log10(data), bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'Log10({col}) Distribution')
            ax.set_xlabel(f'Log10({col})')
        else:
            ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, cols):
    corr_matrix = df[cols].corr()
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Matrix of Key Variables', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_active_inactive_comparison(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Active vs Inactive Lease Comparison', fontsize=16)
    sns.boxplot(x='IS_ACTIVE', y='BID_AMOUNT', data=df, ax=axes[0,0])
    axes[0,0].set_title('Bid Amount Distribution')
    axes[0,0].set_yscale('log')
    sns.boxplot(x='IS_ACTIVE', y='CURRENT_AREA', data=df, ax=axes[0,1])
    axes[0,1].set_title('Lease Area Distribution')
    sns.boxplot(x='IS_ACTIVE', y='BID_PER_HECTARE', data=df, ax=axes[1,0])
    axes[1,0].set_title('Bid per Hectare Distribution')
    axes[1,0].set_yscale('log')
    sns.histplot(x='SALE_YEAR', hue='IS_ACTIVE', data=df, multiple='stack', ax=axes[1,1])
    axes[1,1].set_title('Sale Year Distribution')
    plt.tight_layout()
    plt.show()

def plot_elbow_curve(inertias, k_range):
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_clusters(cluster_df, pca_result, cluster_labels, optimal_k):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Lease Clusters Analysis ({optimal_k} clusters)', fontsize=16)
    sns.scatterplot(x='BID_AMOUNT', y='CURRENT_AREA', hue='Cluster', data=cluster_df, palette='viridis', ax=axes[0,0])
    axes[0,0].set_xscale('log')
    sns.scatterplot(x='SALE_YEAR', y='ROYALTY_RATE', hue='Cluster', data=cluster_df, palette='viridis', ax=axes[0,1])
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=cluster_labels, palette='viridis', ax=axes[1,0])
    axes[1,0].set_xlabel('PC1')
    axes[1,0].set_ylabel('PC2')
    cluster_summary = cluster_df.groupby('Cluster').mean()
    sns.heatmap(cluster_summary.T, annot=True, cmap='viridis', ax=axes[1,1])
    plt.tight_layout()
    plt.show()

def plot_feature_importance(feature_importance):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importance', y='Feature')
    plt.title('Feature Importance for Bid Amount Prediction')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()

def plot_prediction_vs_actual(y_test, y_pred, r2):
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Log10(Bid Amount)')
    plt.ylabel('Predicted Log10(Bid Amount)')
    plt.title(f'Predicted vs Actual Bid Amounts (R² = {r2:.3f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_time_series(yearly_stats):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Temporal Trends in Alaska OCS Leasing', fontsize=16)
    yearly_stats.plot(x='SALE_YEAR', y='LEASE_NUMBER_count', ax=axes[0,0], marker='o')
    axes[0,0].set_title('Number of Lease Sales by Year')
    yearly_stats.plot(x='SALE_YEAR', y='BID_AMOUNT_sum', ax=axes[0,1], marker='o')
    axes[0,1].set_title('Total Bid Amount by Year')
    yearly_stats.plot(x='SALE_YEAR', y='BID_AMOUNT_mean', ax=axes[1,0], marker='o')
    axes[1,0].set_title('Average Bid Amount by Year')
    axes[1,0].set_yscale('log')
    yearly_stats.plot(x='SALE_YEAR', y='ACTIVE_PERCENTAGE', ax=axes[1,1], marker='o')
    axes[1,1].set_title('Percentage of Active Leases by Sale Year')
    plt.tight_layout()
    plt.show()
