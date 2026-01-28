# House Price Prediction with Linear Regression
This project demonstrates a machine learning workflow to predict house prices using linear regression models. It uses a real-world housing dataset and compares two models based on different feature sets:

- **Basic model:** Uses fundamental features such as bedrooms, bathrooms, square footage, floors, and zipcode.
- **Advanced model:** Includes all basic features plus additional property details like condition, grade, waterfront, year built, and location coordinates.

The dataset is split into training, validation, and test sets to ensure fair evaluation. Root Mean Squared Error (RMSE) is used to measure model accuracy. The best-performing model on the validation set is evaluated on the test set to estimate real-world performance.

## Dataset
The data is loaded from `home_data.csv`, containing various attributes of houses sold, including price, size, condition, and location.

## Features Used
- Bedrooms
- Bathrooms
- Sqft Living
- Sqft Lot
- Floors
- Zipcode
- Condition
- Grade
- Waterfront
- View
- Sqft Above
- Sqft Basement
- Year Built
- Year Renovated
- Latitude
- Longitude
- Sqft Living15 (average of 15 nearest neighbors)
- Sqft Lot15 (average lot size of 15 nearest neighbors)

## Installation
Make sure you have Python 3 installed. Install the required packages with:

```bash
pip install pandas numpy matplotlib scikit-learn
