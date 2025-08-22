# House Price Prediction Using Ridge Regression

## Overview
This project aims to predict house prices based on various features (like size, location, number of rooms, etc.) using **Ridge Regression**, a regularized linear regression technique. Ridge Regression helps reduce overfitting by adding a penalty term to the linear regression cost function.

## Features
- Predict house prices based on multiple input features
- Handles multicollinearity in data
- Regularization to prevent overfitting
- Train-test split for model evaluation

## Dataset
The dataset should contain features such as:

| Feature | Description |
|---------|-------------|
| `Size` | Size of the house in square feet |
| `Bedrooms` | Number of bedrooms |
| `Bathrooms` | Number of bathrooms |
| `Location` | Categorical variable for location |
| `Year_Built` | Year the house was built |
| `Price` | Target variable (House price) |

> You can use any CSV dataset containing house prices and relevant features.

## Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd house-price-prediction
```
2. Create a virtual environment (optional but recommended):
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
```
3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
1. Load the dataset:
```python
import pandas as pd

data = pd.read_csv('house_prices.csv')
```

2. Preprocess the data (handle missing values, encode categorical features):
```python
from sklearn.model_selection import train_test_split

X = data.drop('Price', axis=1)
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

3. Train Ridge Regression model:
```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
```

4. Save the trained model (optional):
```python
import joblib
joblib.dump(ridge, 'ridge_model.pkl')
```

## Evaluation
The model can be evaluated using metrics such as:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

## Dependencies
- Python 3.x
- pandas
- scikit-learn
- numpy
- joblib (optional, for saving models)

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

