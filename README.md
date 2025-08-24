# Alaska Oil & Gas Lease Analysis 🛢️📊

A comprehensive data analysis project examining Alaska Outer Continental Shelf (OCS) oil and gas lease patterns, market trends, and geospatial insights using advanced statistical methods and machine learning techniques.

## 🎯 Project Overview

This project analyzes federal oil and gas leasing data from Alaska's Outer Continental Shelf, providing insights into bidding patterns, geographic distribution, temporal trends, and market dynamics. The analysis combines traditional statistical methods, geospatial visualization, machine learning, and advanced statistical techniques including Bayesian regression, time series forecasting, survival analysis, and causal inference to uncover key patterns in offshore energy leasing.

## 📊 Key Findings

- **Dataset**: 2,446 lease records spanning multiple decades (1976-2024)
- **Total Bid Value**: Over $8.1 billion in lease bids analyzed
- **Geographic Coverage**: Multiple planning areas including Beaufort Sea (60%+ of leases), Cook Inlet, and Gulf of Alaska
- **Temporal Range**: Lease sales from 1976 to present with peak activity in 1980s and early 2000s
- **Active Status**: 15% active leases, 85% inactive/expired with clear lifecycle patterns
- **Predictive Model**: Random Forest achieves R² = 0.654 ± 0.023 for bid amount prediction

## 🗂️ Project Structure

```
AK-Lease-Analysis/
├── data/                              # Raw datasets
│   ├── AK_Leases.csv                 # Main lease dataset (2,446 records)
│   ├── AK_Leases.geojson            # Geospatial lease boundaries
│   └── AK_Lease_Metadata.json       # Dataset metadata and documentation
├── notebooks/                         # Jupyter analysis notebooks
│   ├── 01_data_exploration.ipynb     # Initial data exploration and quality assessment
│   ├── 02_geospatial_analysis.ipynb  # Geographic analysis & interactive mapping
│   ├── 03_statistical_analysis.ipynb # Statistical modeling & clustering
│   ├── 04_advanced_statistical_methods.ipynb # Bayesian, time series, survival analysis
│   └── 05_economic_impact_modeling.ipynb # Economic impact & market dynamics analysis
├── visualizations/                    # Generated plots and maps
│   ├── alaska_leases_map.html        # Interactive Folium map
│   └── correlation_matrix.png        # Statistical correlation heatmap
├── reports/                          # Analysis reports and summaries
│   └── executive_summary.md          # Comprehensive executive summary
├── src/                              # Source code utilities
│   ├── data_utils.py                # Data processing utilities
│   └── visualization_utils.py       # Visualization helper functions
├── tests/                            # Unit tests
│   ├── test_data_utils.py           # Data processing tests
│   ├── test_geospatial.py           # Geospatial analysis tests
│   └── test_validation.py           # Data validation tests
├── requirements.txt                  # Python dependencies
├── TECHNICAL_ENHANCEMENTS.md        # Advanced technical features documentation
├── LICENSE                          # MIT License
└── README.md                        # Project documentation
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

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

5. **Start with the notebooks in order:**
   - `01_data_exploration.ipynb` - Data quality assessment and exploratory analysis
   - `02_geospatial_analysis.ipynb` - Geographic patterns and interactive mapping
   - `03_statistical_analysis.ipynb` - Machine learning models and clustering
   - `04_advanced_statistical_methods.ipynb` - Bayesian, time series, and survival analysis
   - `05_economic_impact_modeling.ipynb` - Economic impact & market dynamics analysis

## 📈 Analysis Components

### 1. Data Exploration & Quality Assessment
- **Comprehensive Data Profiling**: 44 variables across 2,446 lease records
- **Data Quality Metrics**: Missing value analysis, outlier detection, temporal consistency
- **Descriptive Statistics**: Statistical overview of lease characteristics with confidence intervals
- **Temporal Analysis**: Trends in leasing activity from 1976-2024
- **Company Analysis**: Market participation by 50+ business associates
- **Financial Overview**: Bid amounts ($1K - $23M+), royalty rates (12.5%-16.67%), lease values

### 2. Geospatial Analysis
- **Interactive Mapping**: Folium-based maps showing lease locations and status
- **Spatial Statistics**: Geographic clustering and density analysis with statistical validation
- **Planning Area Comparison**: Regional differences across Beaufort Sea, Cook Inlet, Gulf of Alaska
- **Temporal-Spatial Trends**: Evolution of geographic leasing patterns over time
- **Boundary Analysis**: Federal OCS boundaries and lease block visualization

### 3. Statistical Analysis & Machine Learning
- **Correlation Analysis**: Relationships between variables with 95% confidence intervals
- **Hypothesis Testing**: Statistical significance testing for key relationships
- **Clustering Analysis**: K-means clustering with optimal parameter selection (4 clusters identified)
- **Predictive Modeling**: Random Forest model with cross-validation (R² = 0.654 ± 0.023)
- **Feature Importance**: Bootstrap confidence intervals for model interpretability
- **Model Diagnostics**: Residual analysis, learning curves, validation curves

### 4. Advanced Statistical Methods
- **Bayesian Regression**: Uncertainty quantification with 95% credible intervals
- **Time Series Forecasting**: ARIMA models for lease activity prediction
- **Survival Analysis**: Kaplan-Meier estimation for lease lifecycle modeling
- **Causal Inference**: Propensity score matching for policy impact assessment

### 5. Economic Impact & Market Dynamics
- **Economic Impact Assessment**: $19.5B total economic impact with multiplier effects
- **Market Competition Analysis**: HHI-based market concentration metrics
- **Financial Risk Modeling**: Monte Carlo simulation with 10,000 scenarios
- **Policy Impact Quantification**: Revenue optimization and regulatory cost-benefit analysis

## 🛠️ Technologies Used

### Core Data Science Stack
- **Data Analysis**: pandas (≥2.0.0), numpy (≥1.24.0), scipy (≥1.10.0)
- **Visualization**: matplotlib (≥3.7.0), seaborn (≥0.12.0), plotly (≥5.14.0)
- **Geospatial**: geopandas (≥0.13.0), folium (≥0.14.0)
- **Machine Learning**: scikit-learn (≥1.3.0)
- **Development**: Jupyter Notebook (≥1.0.0), Python 3.8+

### Advanced Statistical Methods
- **Bayesian Statistics**: pymc (≥5.0.0), arviz (≥0.15.0)
- **Time Series**: statsmodels (≥0.14.0), prophet (≥1.1.0)
- **Survival Analysis**: lifelines (≥0.27.0)
- **Testing**: pytest for unit testing

### Data Quality & Testing
- **Unit Testing**: Comprehensive test suite covering data processing and geospatial analysis
- **Data Validation**: Automated data quality checks and statistical validation
- **Code Quality**: Professional code structure with utility modules and error handling

## 📊 Key Visualizations

1. **Interactive Geographic Maps**: Folium-based lease location mapping with status indicators
2. **Time Series Analysis**: Temporal trends in leasing activity with ARIMA forecasting
3. **Statistical Distributions**: Histograms and box plots with confidence intervals
4. **Correlation Matrices**: Heatmaps with statistical significance testing
5. **Machine Learning Results**: Cluster analysis, feature importance with uncertainty bars
6. **Survival Curves**: Kaplan-Meier lease lifecycle analysis
7. **Bayesian Model Results**: Posterior distributions and uncertainty quantification
8. **Causal Inference**: Propensity score matching visualization and treatment effects
9. **Economic Impact Visualization**: Market dynamics, risk distributions, policy scenarios
10. **Monte Carlo Risk Analysis**: Value-at-Risk and probability distributions

## 🔍 Analytical Insights & Business Value

### Market Intelligence
- **Peak Activity Periods**: 1980s and early 2000s generated 10x average revenue
- **Competitive Bidding**: Clear price discovery mechanisms with rational geological risk assessment
- **Market Cycles**: Boom-bust patterns aligned with oil price cycles enable timing strategies
- **Company Strategies**: Major oil companies (Shell, Exxon, BP) dominate high-value leases

### Geographic Investment Patterns
- **Beaufort Sea Dominance**: 60%+ of leases concentrated in proven exploration areas
- **High-Value Clustering**: Premium locations show statistically significant spatial clustering
- **Regional Performance**: Cook Inlet shows consistent activity, Gulf of Alaska periodic high-value sales
- **Block Performance**: Standard 2,304-hectare blocks predominate with predictable pricing

### Risk Assessment & Predictions
- **Lease Lifecycle**: 85% termination rate within primary term, 15% long-term retention
- **Survival Analysis**: Median lease duration patterns enable portfolio planning
- **Predictive Accuracy**: Random Forest model explains 65.4% ± 2.3% of bid variance
- **Uncertainty Quantification**: Bayesian methods provide confidence intervals for all predictions

### Policy Impact Analysis
- **Regulatory Effects**: Clear impact of policy changes visible in leasing patterns
- **Royalty Rate Analysis**: Causal inference shows statistical relationship with lease performance
- **Environmental Planning**: Geographic clustering enables focused regulatory oversight
- **Revenue Optimization**: Data-driven insights for optimal sale timing and location selection

### Economic Impact & Investment Intelligence
- **Total Economic Impact**: $19.5B with 2.4x multiplier effect across Alaska economy
- **Job Creation**: 69,124 estimated jobs at $8.5 per million invested
- **Market Concentration**: HHI = 1,016 indicating healthy competitive market structure
- **Risk Assessment**: 22.4% probability of profit with high volatility but massive upside potential
- **Optimal Timing**: $30-50/barrel oil price windows provide 23.6x revenue multiplier
- **Policy Optimization**: Streamlined regulatory process maximizes social value

## 📝 Data Sources

**Primary Dataset**: Bureau of Ocean Energy Management (BOEM)
- **Title**: Alaska OCS Oil & Gas Leases
- **Last Updated**: September 17, 2024
- **Coverage**: Active and inactive federal oil and gas leases
- **Format**: CSV, GeoJSON, Metadata JSON

**Data Quality**: Professional government dataset with comprehensive validation and regular updates.

## 🎓 Skills Demonstrated

### Advanced Statistical & Data Science Skills
- **Bayesian Statistics**: Uncertainty quantification with conjugate priors and posterior inference
- **Time Series Analysis**: ARIMA modeling, stationarity testing, forecasting with confidence intervals
- **Survival Analysis**: Kaplan-Meier estimation, hazard modeling, censored data handling
- **Causal Inference**: Propensity score matching, treatment effect estimation, confounding control
- **Machine Learning**: Random Forest with hyperparameter optimization, clustering with parameter selection
- **Model Validation**: K-fold cross-validation, bootstrap methods, residual analysis, learning curves

### Statistical Rigor & Professional Standards
- **Hypothesis Testing**: Comprehensive significance testing with proper multiple comparison control
- **Confidence Intervals**: Bootstrap and analytical confidence intervals for all key statistics
- **Model Diagnostics**: Q-Q plots, homoscedasticity testing, assumption validation
- **Feature Engineering**: Statistical feature selection with importance confidence intervals
- **Uncertainty Quantification**: Bayesian credible intervals and frequentist confidence bounds

### Business Intelligence & Decision Science
- **Market Analysis**: $500M+ lease portfolio analysis with competitive intelligence insights
- **Risk Modeling**: Lease lifecycle analysis with survival probability estimates
- **Investment Strategy**: Geographic and temporal pattern recognition for optimal bidding
- **Policy Evaluation**: Causal impact assessment of regulatory changes on market outcomes
- **Predictive Analytics**: R² = 0.654 ± 0.023 model performance with production-ready validation

### Technical Implementation
- **Production-Ready Code**: Modular design with comprehensive unit testing and error handling
- **Reproducible Research**: All analyses include random seeds and version control
- **Data Engineering**: Robust data processing pipeline with quality validation
- **Interactive Visualization**: Professional dashboards and maps for stakeholder communication
- **Documentation Standards**: Complete technical documentation with assumptions and limitations

> 📋 **See [TECHNICAL_ENHANCEMENTS.md](TECHNICAL_ENHANCEMENTS.md) for detailed documentation of advanced statistical methods and validation techniques.**

## 🤝 Contributing

This project demonstrates advanced data science capabilities for professional portfolio purposes. For questions or collaboration:

1. **Issues**: Report bugs or suggest feature enhancements
2. **Forks**: Use this analysis as a template for your own energy/geospatial projects
3. **Pull Requests**: Contribute improvements to methodology or documentation
4. **Discussion**: Engage with statistical methods or business insights

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. The underlying data is public domain from the U.S. Bureau of Ocean Energy Management.

## 📚 Additional Resources

- **Executive Summary**: [reports/executive_summary.md](reports/executive_summary.md) - Comprehensive business analysis
- **Technical Deep Dive**: [TECHNICAL_ENHANCEMENTS.md](TECHNICAL_ENHANCEMENTS.md) - Advanced statistical methods
- **Interactive Map**: [visualizations/alaska_leases_map.html](visualizations/alaska_leases_map.html) - Geographic visualization
- **Data Source**: [BOEM Alaska Leases](https://www.boem.gov/alaska) - Official lease database

## 🙏 Acknowledgments

- **Bureau of Ocean Energy Management (BOEM)** for maintaining comprehensive, high-quality lease data
- **Alaska OCS Region** for detailed geographic and regulatory context
- **Python Data Science Community** for the excellent ecosystem of statistical and visualization tools
- **Statistical Computing Community** for advancing open-source Bayesian and causal inference methods

---

*This analysis represents a comprehensive examination of Alaska OCS oil and gas leasing patterns using advanced statistical methods. The project demonstrates professional-level data science capabilities suitable for energy sector analysis, regulatory policy evaluation, and investment decision support.*