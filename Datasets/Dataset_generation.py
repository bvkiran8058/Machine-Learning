import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_regression

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
data_size = 5000

# 1. Property Characteristics
property_size = np.random.randint(500, 5000, data_size)  # sqft
num_bedrooms = np.random.randint(1, 6, data_size)
num_bathrooms = np.random.randint(1, 4, data_size)
construction_year = np.random.randint(1950, 2023, data_size)
furnished = np.random.choice(["Fully", "Partially", "None"], data_size, p=[0.3, 0.4, 0.3])

# 2. Demographic and Economic Factors
median_income = np.round(np.random.normal(75, 15, data_size), 2)  # in $1000s
unemployment_rate = np.round(np.random.uniform(2, 10, data_size), 2)  # %
population_density = np.random.randint(500, 10000, data_size)  # people per km^2
school_quality = np.random.choice(["Excellent", "Good", "Average", "Poor"], data_size, p=[0.2, 0.4, 0.3, 0.1])

# 3. Environmental and Accessibility Features
distance_city_center = np.round(np.random.uniform(1, 50, data_size), 2)  # km
distance_nearest_park = np.round(np.random.uniform(0.1, 10, data_size), 2)  # km
air_quality_index = np.random.randint(50, 200, data_size)  # AQI scale
public_transport_access = np.random.choice(["High", "Medium", "Low"], data_size, p=[0.4, 0.4, 0.2])

# 4. Market Trends
regional_price_growth = np.round(np.random.uniform(-5, 15, data_size), 2)  # %
historical_price_volatility = np.round(np.random.uniform(1, 10, data_size), 2)  # arbitrary scale

# 5. Target Variable: Selling Price
# Simulate selling price with a more intricate linear and nonlinear combination of features
selling_price = (
    property_size * np.random.uniform(50, 150) +
    num_bedrooms * np.random.uniform(10000, 30000) +
    num_bathrooms * np.random.uniform(15000, 40000) -
    distance_city_center * np.random.uniform(500, 2000) -
    distance_nearest_park * np.random.uniform(200, 800) +
    median_income * np.random.uniform(2000, 4000) -
    air_quality_index * np.random.uniform(50, 100) +
    regional_price_growth * 1000 +
    np.log1p(population_density) * np.random.uniform(500, 1500) -
    unemployment_rate * np.random.uniform(2000, 5000) +
    (2023 - construction_year) * np.random.uniform(100, 300) +
    np.random.normal(0, 50000, data_size)
)


# Convert to pandas Series for mapping
public_transport_access = pd.Series(public_transport_access)
school_quality = pd.Series(school_quality)

# Add interactions and nonlinearity
selling_price += (
    (distance_city_center * public_transport_access.map({"High": -0.8, "Medium": -0.5, "Low": 0.2})) * 500 +
    (school_quality.map({"Excellent": 1.2, "Good": 1.0, "Average": 0.8, "Poor": 0.5}) * 20000)
)
# Handle extreme luxury properties explicitly
luxury_indices = np.random.choice(data_size, size=int(0.05 * data_size), replace=False)
selling_price[luxury_indices] *= np.random.uniform(1.5, 3.0, len(luxury_indices))

# Convert air_quality_index to float to allow NaN values
air_quality_index = air_quality_index.astype(float)

# Introduce missing values
median_income[np.random.choice(data_size, 250, replace=False)] = np.nan
school_quality[np.random.choice(data_size, 150, replace=False)] = np.nan
air_quality_index[np.random.choice(data_size, 200, replace=False)] = np.nan

# Create DataFrame
data = pd.DataFrame({
    "Property Size (sqft)": property_size,
    "Number of Bedrooms": num_bedrooms,
    "Number of Bathrooms": num_bathrooms,
    "Construction Year": construction_year,
    "Furnished": furnished,
    "Median Household Income": median_income,
    "Unemployment Rate": unemployment_rate,
    "Population Density": population_density,
    "Nearby School Quality Score": school_quality,
    "Distance to City Center (km)": distance_city_center,
    "Distance to Nearest Park (km)": distance_nearest_park,
    "Air Quality Index": air_quality_index,
    "Public Transport Access": public_transport_access,
    "Regional Price Growth (%)": regional_price_growth,
    "Historical Price Volatility": historical_price_volatility,
    "Selling Price (in $)": selling_price
})

# Save to CSV
data.to_csv("real_estate_dataset.csv", index=False)

print("Dataset generated and saved as 'real_estate_dataset.csv'")
