import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load your data
data = np.loadtxt('./gbdt_plain_layerwise.csv', delimiter=',', skiprows=1)
pos = data[:, 0].reshape(-1, 1)  # Ensure pos is a 2D array for sklearn
feature_importance = data[:, 1]

# Generate polynomial features
degree = 5
poly_features = PolynomialFeatures(degree=degree)
pos_poly = poly_features.fit_transform(pos)  # Transform pos to polynomial features

# Split dataset 
pos_poly_train, pos_poly_test, feature_importance_train, feature_importance_test = train_test_split(pos_poly, feature_importance, test_size=0.2, random_state=42)


# Fit a linear regression model
model = LinearRegression()
model.fit(pos_poly, feature_importance)  # Fit model to the polynomial features and target

# Predict and evaluate the model
feature_importance_pred = model.predict(pos_poly_test)
mse = mean_squared_error(feature_importance_test, feature_importance_pred)
r2 = r2_score(feature_importance_test, feature_importance_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Extract the model's coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Constructing the polynomial equation
equation = f"y = {intercept:.4f}"
for i, coeff in enumerate(coefficients):
    if i == 0:  # Skip the first coeff, it's already captured as the intercept
        continue
    # Only add terms with non-zero coefficients
    if coeff != 0:
        equation += f" + {coeff:.4f}*x^{i}"

print("Polynomial Equation:")
print(equation)

def polynomial_equation(x):
    return 0.2917 + -0.0475*(x**1) + 0.0025*(x**2) + -0.0001*(x**3)

# plot the equation and real data  
import matplotlib.pyplot as plt

x_values = np.linspace(0, 100, 400)
y_values = polynomial_equation(x_values)


# Re-plotting with the updated x range
plt.figure(figsize=(10, 6))
# plt.plot(x_values, y_values, label='Polynomial Regression')
plt.scatter(pos, feature_importance, color='blue', label='Real data')
plt.xlabel('Position (pos)')
plt.ylabel('Feature Importance')
plt.title('Polynomial Regression: Feature Importance vs Position (0 to 100)')
plt.legend()
plt.grid(True)
plt.savefig('polynomial_regression.png')