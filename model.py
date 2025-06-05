import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------------------------------
# 1. Read CSV files
# -------------------------------------------------------------------------------
metadata_country = pd.read_csv('Metadata_Country_API_SP.POP.65UP.TO.ZS_DS2_en_csv_v2_22652.csv')
metadata_indicator = pd.read_csv('Metadata_Indicator_API_SP.POP.65UP.TO.ZS_DS2_en_csv_v2_22652.csv')
aging_data_raw = pd.read_csv('API_SP.POP.65UP.TO.ZS_DS2_en_csv_v2_22652.csv', skiprows=4)
# skiprows=4 because the World Bank data file typically has several description lines at the top

# -------------------------------------------------------------------------------
# 2. Data preprocessing
# -------------------------------------------------------------------------------
# Rename columns for easier usage
aging_data_raw = aging_data_raw.rename(columns={'Country Name': 'country', 'Country Code': 'country_code'})

# Keep only columns for years 2000–2023
years = [str(year) for year in range(2000, 2024)]
available_years = [y for y in years if y in aging_data_raw.columns]

# Create a DataFrame containing only valid countries and percentage data for 2000–2023
aging_data = aging_data_raw[['country', 'country_code'] + available_years].copy()

# Drop rows that have all missing values from 2000–2023
aging_data = aging_data.dropna(subset=available_years, how='all')

# -------------------------------------------------------------------------------
# 3. K-Means clustering (Modeling Step 1: Clustering Analysis)
#    Use the 2023 aging percentage to create three clusters (low, medium, high)
# -------------------------------------------------------------------------------
# 3.1 Keep only countries that have data for 2023
cluster_df = aging_data.dropna(subset=['2023']).copy()

# 3.2 Extract the feature matrix: here we only use the "2023" column for clustering
X_cluster = cluster_df[['2023']].values

# 3.3 Apply KMeans to form three clusters
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_cluster)
cluster_df['cluster'] = cluster_labels

# 3.4 Calculate the Silhouette Score to evaluate clustering performance
sil_score = silhouette_score(X_cluster, cluster_labels)

# 3.5 View a sample of clustering results (sorted by the cluster column)
cluster_summary = cluster_df[['country', 'country_code', '2023', 'cluster']].sort_values('cluster').reset_index(drop=True)

# -------------------------------------------------------------------------------
# 4. Linear regression trend analysis (Modeling Step 2: Time Series or Trend Analysis)
#    Perform a simple linear regression on each country's 2000–2023 aging percentages
# -------------------------------------------------------------------------------
regression_results = []

# 4.1 Construct a year feature matrix (same X_years for all countries)
X_years_full = np.array([int(y) for y in available_years]).reshape(-1, 1)

for idx, row in aging_data.iterrows():
    country_name = row['country']
    country_code = row['country_code']
    # Extract this row's 2000–2023 data
    y_vals = row[available_years].values.astype(float)
    mask = ~np.isnan(y_vals)  # Only select non-NaN values

    # Only perform regression if there are at least 3 valid data points
    if mask.sum() >= 3:
        X_fit = X_years_full[mask]  # Remove rows corresponding to NaN years
        y_fit = y_vals[mask]

        # Fit linear regression
        model = LinearRegression()
        model.fit(X_fit, y_fit)
        y_pred = model.predict(X_fit)

        # Calculate MSE and R^2
        mse = mean_squared_error(y_fit, y_pred)
        r2 = r2_score(y_fit, y_pred)

        regression_results.append({
            'country': country_name,
            'country_code': country_code,
            'slope': model.coef_[0],
            'intercept': model.intercept_,
            'mse': mse,
            'r2': r2
        })

regression_df = pd.DataFrame(regression_results)
# Sort by R^2 in descending order to see which countries have the best fit
regression_summary = regression_df.sort_values('r2', ascending=False).reset_index(drop=True)

# -------------------------------------------------------------------------------
# 5. Output / view sample results
# -------------------------------------------------------------------------------
print("=== Clustering Sample (first 10 rows) ===")
print(cluster_summary.head(10).to_string(index=False))
cluster_summary.to_csv("cluster_summary.csv")
regression_summary.to_csv("regression_summary.csv")
print("\nSilhouette Score: {:.4f}".format(sil_score))

print("\n=== Regression Analysis Sample (first 10 rows) ===")
print(regression_summary.head(10).to_string(index=False))
