import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)

# Generate random whole numbers for the number of rooms (1 to 5 rooms)
X = np.random.randint(1, 6, size=(100, 1))  # Random integers for the number of rooms
# Linear relationship with noise, prices in thousands (e.g., base price 4000, slope 3000/room)
y = 4000 + 3000 * X + np.random.randn(100, 1) * 500  # Adding noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the original data and the regression line
plt.scatter(X_test, y_test, color='black', label='Actual data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regression line')
plt.xticks(np.arange(1, 6))  # Set x-ticks for room numbers
plt.xlabel('Number of Rooms')
plt.ylabel('House Price (in Thousands)')
plt.title('Simple Linear Regression Example')
plt.legend()
plt.show()
