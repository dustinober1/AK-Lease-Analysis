# Alaska Oil & Gas Lease Analysis 🛢️📊

A comprehensive data analysis project examining Alaska Outer Continental Shelf (OCS) oil and gas lease patterns, market trends, and geospatial insights.

## 🎯 Project Overview

This project analyzes federal oil and gas leasing data from Alaska's Outer Continental Shelf, providing insights into bidding patterns, geographic distribution, temporal trends, and market dynamics. The analysis combines statistical methods, geospatial visualization, and machine learning to uncover key patterns in offshore energy leasing.

## 📊 Key Findings

- **Dataset**: 2,000+ lease records spanning multiple decades
- **Total Bid Value**: Over $500 million in lease bids analyzed
- **Geographic Coverage**: Multiple planning areas including Beaufort Sea, Cook Inlet, and Gulf of Alaska
- **Temporal Range**: Lease sales from 1976 to present
- **Active Status**: Comprehensive analysis of active vs. inactive lease patterns

## 🗂️ Project Structure

```
AK-Lease-Analysis/
├── data/                          # Raw datasets
│   ├── AK_Leases.csv             # Main lease dataset
│   ├── AK_Leases.geojson         # Geospatial lease boundaries
│   └── AK_Lease_Metadata.json    # Dataset metadata
├── notebooks/                     # Jupyter analysis notebooks
│   ├── 01_data_exploration.ipynb # Initial data exploration
│   ├── 02_geospatial_analysis.ipynb # Geographic analysis & mapping
│   └── 03_statistical_analysis.ipynb # Statistical modeling & clustering
├── visualizations/               # Generated plots and maps
├── reports/                      # Analysis reports and summaries
├── src/                         # Source code utilities
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AK-Lease-Analysis.git
   cd AK-Lease-Analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

4. **Start with the notebooks in order:**
   - `01_data_exploration.ipynb` - Overview and initial analysis
   - `02_geospatial_analysis.ipynb` - Geographic patterns and mapping
   - `03_statistical_analysis.ipynb` - Advanced analytics and modeling

## 📈 Analysis Components

### 1. Data Exploration
- **Descriptive Statistics**: Comprehensive overview of lease characteristics
- **Temporal Analysis**: Trends in leasing activity over time
- **Company Analysis**: Market participation by major oil companies
- **Financial Overview**: Bid amounts, royalty rates, and lease values

### 2. Geospatial Analysis
- **Interactive Mapping**: Folium-based maps of lease locations
- **Spatial Patterns**: Geographic clustering and density analysis
- **Planning Area Comparison**: Regional differences in leasing activity
- **Temporal-Spatial Trends**: How leasing patterns changed geographically over time

### 3. Statistical Analysis
- **Correlation Analysis**: Relationships between key variables
- **Hypothesis Testing**: Statistical comparisons between lease groups
- **Clustering**: Machine learning-based lease categorization
- **Predictive Modeling**: Random Forest model for bid amount prediction

## 🛠️ Technologies Used

- **Data Analysis**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn, plotly
- **Geospatial**: geopandas, folium
- **Machine Learning**: scikit-learn
- **Development**: Jupyter Notebook, Python

## 📊 Key Visualizations

1. **Geographic Distribution Maps**: Interactive maps showing lease locations and status
2. **Temporal Trend Charts**: Time series analysis of leasing activity
3. **Statistical Distributions**: Histograms and box plots of key metrics
4. **Correlation Heatmaps**: Relationships between variables
5. **Cluster Analysis**: Machine learning-based lease categorization
6. **Predictive Model Performance**: Model accuracy and feature importance

## 🔍 Analytical Insights

### Market Dynamics
- Identification of peak leasing periods and market cycles
- Analysis of bid competition and pricing strategies
- Geographic preferences and exploration patterns

### Regulatory Patterns
- Impact of regulatory changes on leasing activity
- Relationship between lease terms and market conditions
- Active vs. inactive lease patterns

### Geographic Insights
- Spatial clustering of high-value leases
- Regional differences in bid amounts and success rates
- Environmental and geological factor correlations

## 📝 Data Sources

**Primary Dataset**: Bureau of Ocean Energy Management (BOEM)
- **Title**: Alaska OCS Oil & Gas Leases
- **Last Updated**: September 17, 2024
- **Coverage**: Active and inactive federal oil and gas leases
- **Format**: CSV, GeoJSON, Metadata JSON

**Data Quality**: Professional government dataset with comprehensive validation and regular updates.

## 🎓 Skills Demonstrated

- **Data Science**: End-to-end analysis pipeline from raw data to insights
- **Statistical Analysis**: Hypothesis testing, correlation analysis, regression modeling
- **Geospatial Analysis**: GIS data handling, spatial statistics, interactive mapping
- **Machine Learning**: Clustering, predictive modeling, feature engineering
- **Data Visualization**: Professional charts, interactive maps, statistical plots
- **Project Management**: Structured workflow, documentation, reproducible analysis

## 🤝 Contributing

This project is part of a data science portfolio. For questions or suggestions:

1. Open an issue for bugs or feature requests
2. Fork the repository for your own analysis
3. Submit pull requests for improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Bureau of Ocean Energy Management (BOEM)** for providing comprehensive lease data
- **Alaska OCS Region** for detailed geographic and regulatory information
- **Open Source Community** for the excellent Python data science ecosystem

## 📧 Contact

**Your Name** - [your.email@example.com]
- LinkedIn: [Your LinkedIn Profile]
- Portfolio: [Your Portfolio Website]
- GitHub: [@yourusername](https://github.com/yourusername)

---

*This project demonstrates advanced data analysis capabilities applied to real-world energy sector data, showcasing skills in statistical analysis, geospatial visualization, and machine learning.*