# Import necessary libraries

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score



# Load your dataset (assuming it's in a CSV file)

# Replace 'dataset.csv' with your actual dataset file

data = pd.read_csv('dataset.csv')



# Assume 'features' contains the columns you want to use for prediction

features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors']

X = data[features]

y = data['price']



# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Create a linear regression model

model = LinearRegression()



# Train the model

model.fit(X_train, y_train)



# Make predictions on the test set

predictions = model.predict(X_test)



# Calculate and print metrics

mse = mean_squared_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')

print(f'R-squared: {r2}')

 # Now you can use the trained model to make predictions for new data

