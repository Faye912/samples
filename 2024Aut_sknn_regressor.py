#%%
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

print("Ready to continue")


#%%
class sknn:
    def __init__(self, 
                 data_x, 
                 data_y, 
                 classifier=False, 
                 k=7, 
                 kmax=33, 
                 zscale=True, 
                 max_iter=100, 
                 seed=1, 
                 scoredigits=6, 
                 learning_rate_init=0.1, 
                 atol=1e-8):
        """
        Modified sknn class for K-NN regression with preprocessing.
        """
        self.classifier = classifier  # True for classification, False for regression
        self.k = k
        self.kmax = kmax
        self.max_iter = max_iter
        self.seed = seed
        self.scoredigits = scoredigits
        self.learning_rate = abs(learning_rate_init)
        self.atol = atol

        # Preprocess data (only feature data)
        self.data_x = data_x.copy()  # Make a copy to avoid modifying the original dataset
        self.data_y = data_y.copy()  # Make a copy of the target data
        self.zscale = zscale
        self.preprocess_data()  # Apply preprocessing (scaling, encoding, etc.)

        # Initialize k-NN models
        self.knnmodels = [None] * (self.kmax + 1)
        for i in range(2, self.kmax + 1):
            if self.classifier:
                self.knnmodels[i] = KNeighborsClassifier(n_neighbors=i).fit(self.data_x, self.data_y)
            else:
                self.knnmodels[i] = KNeighborsRegressor(n_neighbors=i).fit(self.data_x, self.data_y)

        # Benchmark scores for data
        self.benchmarkScores = [None] * (self.kmax + 1)
        for i in range(2, self.kmax + 1):
            self.benchmarkScores[i] = self.scorethis(k=i, use='train')

        # Track results from optimization
        self.results = []

    def preprocess_data(self):
        """
        Preprocess the data by handling missing values, encoding, and scaling.
        This function now only processes feature columns and does not affect the target column (`SalePrice`).
        """
        self.data_x, self.data_y = self.data_x.align(self.data_y, join='inner', axis=0)

        # Drop categorical columns without encoding
        categorical_var = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 
                        'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 
                        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 
                        'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'Electrical', 'GarageType', 
                        'GarageFinish', 'PavedDrive', 'MiscFeature', 'SaleType', 'MoSold', 'YrSold', 
                        'SaleCondition', 'BsmtExposure']
        
        # Drop the categorical columns from the feature data
        self.data_x = self.data_x.drop(columns=categorical_var, errors='ignore')  # Drop columns if they exist

        # Ordinal encoding for specific columns
        self.data_x['OverallQual'] = self.data_x['OverallQual'].map({1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9})
        self.data_x['OverallCond'] = self.data_x['OverallCond'].map({1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9})
        self.data_x['BsmtFinType1'] = self.data_x['BsmtFinType1'].map({'NA': 0, 'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5})
        self.data_x['BsmtFinType2'] = self.data_x['BsmtFinType2'].map({'NA': 0, 'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5})
        self.data_x['Functional'] = self.data_x['Functional'].map({'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7})
        self.data_x['Fence'] = self.data_x['Fence'].map({'NA': 0, 'MnWw': 0, 'GdWo': 1, 'MnPrv': 2, 'GdPrv': 3})

        # self.data_x_test['OverallQual'] = self.data_x_test['OverallQual'].map({1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9})
        # self.data_x_test['OverallCond'] = self.data_x_test['OverallCond'].map({1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9})
        # self.data_x_test['BsmtFinType1'] = self.data_x_test['BsmtFinType1'].map({'NA': 0, 'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5})
        # self.data_x_test['BsmtFinType2'] = self.data_x_test['BsmtFinType2'].map({'NA': 0, 'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5})
        # self.data_x_test['Functional'] = self.data_x_test['Functional'].map({'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7})
        # self.data_x_test['Fence'] = self.data_x_test['Fence'].map({'NA': 0, 'MnWw': 0, 'GdWo': 1, 'MnPrv': 2, 'GdPrv': 3})

        # Quality mappings
        quality_mapping = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
        for col in ['ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual']:
            self.data_x[col] = self.data_x[col].map(quality_mapping)

        # Handle 'NA' for specific columns
        na_quality_mapping = {'NA': 0, 'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
        for col in ['BsmtQual', 'BsmtCond', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']:
            self.data_x[col] = self.data_x[col].map(na_quality_mapping)

        # Standardize numerical features
        numerical_var = ['LotArea', 'LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
                        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 
                        'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                        'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 
                        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

        # Impute missing values (using mean for numerical columns)
        imputer = SimpleImputer(strategy='mean')
        self.data_x[numerical_var] = imputer.fit_transform(self.data_x[numerical_var])

        # Scale the features
        scaler = StandardScaler()
        self.data_x[numerical_var] = scaler.fit_transform(self.data_x[numerical_var])

        # Ensure no NaN values remain
        if self.data_x.isna().sum().sum() > 0:
            print("Error: Missing values remain in the training dataset after preprocessing.")
            self.data_x = self.data_x.fillna(0)  # Handle NaNs if any remain (shouldn't happen if imputed correctly)
        self.data_y = self.data_y[self.data_x.index]


    def scorethis(self, k=None, use='train'):
        """
        Evaluate model performance using either training data.
        Args:
            k (int): Number of neighbors.
            use (str): 'train' dataset to evaluate.
        Returns:
            float: Model performance (R² for regression, accuracy for classification).
        """
        if k is None:
            k = self.k

        model = self.knnmodels[k]
        if use == 'train':
            X, y = self.data_x, self.data_y
        y_pred = model.predict(X)
        if self.classifier:
            return round(accuracy_score(y, y_pred), 6)
        else:
            return round(r2_score(y, y_pred), 6)
        

    def optimize_scaling(self):
        """
        Optimize feature scaling factors to improve model performance.
        """
        print("Starting feature scaling optimization...")
        num_features = self.data_x.shape[1]
        scaling_factors = np.ones(num_features)  # Start with uniform scaling
        prev_score = -np.inf

        for i in range(self.max_iter):
            gradients = self._compute_gradients(scaling_factors)
            scaling_factors += self.learning_rate * gradients

            # Update train scores
            X_train_scaled = self.data_x * scaling_factors
            train_score = self._evaluate_scaled_model(X_train_scaled, use='train')

            # Log results
            self.results.append({
                'iteration': i,
                'scaling_factors': scaling_factors.copy(),
                'train_score': train_score,
            })

            # Save results every 100 epochs
            if i % 100 == 0:
                self.save_results()

            # Early stopping
            if abs(train_score - prev_score) < self.atol:
                print(f"Convergence reached at iteration {i}.")
                break
            prev_score = train_score

        print("Optimization complete.")
        self.save_results()

    def _compute_gradients(self, scaling_factors):
        """
        Placeholder gradient computation for scaling factors.
        In practice, this would be based on partial derivatives of the loss function.
        """
        gradients = np.random.uniform(-0.1, 0.1, size=len(scaling_factors))  # Example: Random gradients
        return gradients

    def _evaluate_scaled_model(self, X_scaled, use='train'):
        """Evaluate the scaled model's performance using training data."""
        model = self.knnmodels[self.k]
        
        # Predict values using the trained model
        y_pred = model.predict(X_scaled)
        
        if use == 'train':
            # Compare predictions with actual training target values
            return r2_score(self.data_y, y_pred) if not self.classifier else accuracy_score(self.y_train, y_pred)
        
    def save_results(self, filename="optimization_results.csv"):
        """Save optimization results to a CSV file."""
        pd.DataFrame(self.results).to_csv(filename, index=False)

    def compare_with_feature_importance(self):
        """
        Compare optimized scaling factors with feature importance from a RandomForest model.
        """
        rf = RandomForestRegressor(random_state=1)
        rf.fit(self.data_x, self.data_y)
        feature_importance = rf.feature_importances_

        # Print comparison
        print("Feature Importance vs. Optimized Scaling Factors")
        for i, (importance, scaling) in enumerate(zip(feature_importance, self.results[-1]['scaling_factors'])):
            print(f"Feature {i + 1}: Importance = {importance:.4f}, Scaling Factor = {scaling:.4f}")

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_importance)), feature_importance, alpha=0.7, label="Feature Importance")
        plt.bar(range(len(feature_importance)), self.results[-1]['scaling_factors'], alpha=0.7, label="Scaling Factors")
        plt.legend()
        plt.title("Feature Importance vs Scaling Factors")
        plt.show()

#%%
train_df = pd.read_csv('train.csv')

# Separate features (X) and target (y) for training data
data_x = train_df.drop(columns=['SalePrice'])
data_y = train_df['SalePrice']


# Initialize sknn model
knn_model = sknn(data_x=data_x, data_y=data_y, classifier=False, k=5)

# Train and evaluate the model
print(f"R² Score: {knn_model.scorethis(k=5, use='train')}")
# Optimize scaling factors
knn_model.optimize_scaling()

# Compare with feature importance
knn_model.compare_with_feature_importance()


# %%
