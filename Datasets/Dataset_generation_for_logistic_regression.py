import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
data_size = 5000

# 1. Demographics
age = np.random.randint(18, 70, data_size)  # Age
gender = np.random.choice(["Male", "Female"], data_size, p=[0.5, 0.5])
income = np.round(np.random.normal(50, 20, data_size), 2)  # in $1000s

# 2. Behavioral Factors
online_hours_per_week = np.random.randint(0, 50, data_size)
purchase_frequency_last_month = np.random.randint(0, 20, data_size)
is_loyal_customer = np.random.choice([1, 0], data_size, p=[0.3, 0.7])  # 1: Loyal, 0: New/Non-loyal
preferred_shopping_platform = np.random.choice(
    ["Website", "Mobile App", "In-store"], 
    data_size, 
    p=[0.4, 0.4, 0.2]
)

# 3. Marketing and Economic Influence
ad_clicks_last_month = np.random.randint(0, 15, data_size)
discounts_used_last_month = np.random.randint(0, 10, data_size)
competitor_pricing_index = np.round(np.random.uniform(0.8, 1.2, data_size), 2)  # Relative index to 1.0
economic_conditions = np.random.choice(
    ["Favorable", "Neutral", "Unfavorable"], 
    data_size, 
    p=[0.5, 0.3, 0.2]
)

# 4. Environmental and Behavioral Interaction
distance_to_nearest_store = np.round(np.random.uniform(0.5, 50, data_size), 2)  # in km
shopping_frequency = (
    20 - distance_to_nearest_store / 2 + 
    is_loyal_customer * 5 +
    purchase_frequency_last_month / 2 +
    np.random.normal(0, 2, data_size)
)

# Normalize shopping frequency and add noise
shopping_frequency = np.clip(shopping_frequency, 1, 30)

# 5. Target Variable: Purchase Decision (1 = Purchased, 0 = Not Purchased)
logit = (
    -2 + 
    0.05 * age +
    0.03 * income +
    0.1 * online_hours_per_week +
    0.4 * is_loyal_customer -
    0.02 * distance_to_nearest_store +
    0.3 * (ad_clicks_last_month + discounts_used_last_month) -
    0.5 * (competitor_pricing_index - 1) +
    np.random.normal(0, 0.5, data_size)
)

# Convert logit to probabilities and then to binary outcomes
purchase_probability = 1 / (1 + np.exp(-logit))
purchase_decision = np.random.binomial(1, purchase_probability, data_size)

# Introduce missing values
income[np.random.choice(data_size, 200, replace=False)] = np.nan
economic_conditions[np.random.choice(data_size, 150, replace=False)] = np.nan

# Create DataFrame
data = pd.DataFrame({
    "Age": age,
    "Gender": gender,
    "Income (in $1000s)": income,
    "Online Hours Per Week": online_hours_per_week,
    "Purchase Frequency (Last Month)": purchase_frequency_last_month,
    "Is Loyal Customer": is_loyal_customer,
    "Preferred Shopping Platform": preferred_shopping_platform,
    "Ad Clicks (Last Month)": ad_clicks_last_month,
    "Discounts Used (Last Month)": discounts_used_last_month,
    "Competitor Pricing Index": competitor_pricing_index,
    "Economic Conditions": economic_conditions,
    "Distance to Nearest Store (km)": distance_to_nearest_store,
    "Shopping Frequency": shopping_frequency,
    "Purchase Decision (1=Yes, 0=No)": purchase_decision
})

# Save to CSV
data.to_csv("customer_behavior_dataset.csv", index=False)

print("Dataset generated and saved as 'customer_behavior_dataset.csv'")
