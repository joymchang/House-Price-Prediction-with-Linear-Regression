import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(416)  # For reproducible results

# Load dataset
sales = pd.read_csv('home_data.csv')
print(sales.head())

# Target variable
y = sales["price"]

# Calculate average price of 3-bedroom houses
avg_price_3_bed = sales[sales["bedrooms"] == 3]["price"].mean()

# Calculate percentage of houses with living area between 2000 and 4000 sqft
correct_sqft = (sales["sqft_living"] >= 2000) & (sales["sqft_living"] < 4000)
percent_q3 = correct_sqft.mean()

# Split data: 70% train, 15% validation, 15% test
train_data, val_and_test_data = train_test_split(sales, test_size=0.3)
val_data, test_data = train_test_split(val_and_test_data, test_size=0.5)

# Plot sqft_living vs price for train and validation sets
plt.scatter(train_data['sqft_living'], train_data['price'], marker='+', label='Train')
plt.scatter(val_data['sqft_living'], val_data['price'], marker='.', label='Validation')
plt.legend()
plt.xlabel('Sqft Living')
plt.ylabel('Price')
plt.show()

# Define basic and advanced feature sets
basic_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']

advanced_features = basic_features + [
    'condition', 'grade', 'waterfront', 'view', 'sqft_above', 'sqft_basement',
    'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15',
]

# Prepare training data
basic_x = train_data[basic_features]
train_y = train_data["price"]

# Train basic linear regression model
basic_model = LinearRegression()
basic_model.fit(basic_x, train_y)

# Train advanced linear regression model
adv_x = train_data[advanced_features]
advanced_model = LinearRegression()
advanced_model.fit(adv_x, train_y)

# Calculate training RMSE for basic model
pred_basic = basic_model.predict(basic_x)
train_rmse_basic = np.sqrt(mean_squared_error(train_y, pred_basic))

# Calculate training RMSE for advanced model
pred_adv = advanced_model.predict(adv_x)
train_rmse_advanced = np.sqrt(mean_squared_error(train_y, pred_adv))

# Calculate validation RMSE for basic model
val_pred_basic = basic_model.predict(val_data[basic_features])
val_rmse_basic = np.sqrt(mean_squared_error(val_data["price"], val_pred_basic))

# Calculate validation RMSE for advanced model
val_pred_adv = advanced_model.predict(val_data[advanced_features])
val_rmse_advanced = np.sqrt(mean_squared_error(val_data["price"], val_pred_adv))

# Choose better model based on validation RMSE
if val_rmse_basic < val_rmse_advanced:
    better_model = basic_model
    test_features = basic_features
else:
    better_model = advanced_model
    test_features = advanced_features

# Evaluate chosen model on test data
test_pred = better_model.predict(test_data[test_features])
test_rmse = np.sqrt(mean_squared_error(test_data["price"], test_pred))

# Print summary of results
print("=== Model Performance Summary ===")
print(f"Training RMSE (Basic Features): {train_rmse_basic:.2f}")
print(f"Training RMSE (Advanced Features): {train_rmse_advanced:.2f}")
print(f"Validation RMSE (Basic Features): {val_rmse_basic:.2f}")
print(f"Validation RMSE (Advanced Features): {val_rmse_advanced:.2f}")
print(f"Test RMSE (Selected Model): {test_rmse:.2f}")

if val_rmse_basic < val_rmse_advanced:
    print("\nConclusion: Basic features model performs better on validation data.")
else:
    print("\nConclusion: Advanced features model performs better on validation data.")

print("Test RMSE estimates model performance on unseen data.")
print("Further improvements could include feature engineering or using more complex models.")