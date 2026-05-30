import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from ucimlrepo import fetch_ucirepo 

class DataLoader:
    def __init__(self):
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')

    def load_data(self):
        # Fetch dataset
        statlog_german_credit_data = fetch_ucirepo(id=144) 
        self.X = statlog_german_credit_data.data.features 
        self.y = statlog_german_credit_data.data.targets 

    def preprocess(self):
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Run load_data() first.")

        # 1. Split data (using stratify for imbalanced credit data)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        # 2. Identify categorical columns correctly via Pandas data types
        categorical_cols = X_train.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
        numeric_cols = X_train.select_dtypes(exclude=['object', 'category', 'str']).columns.tolist()

        if categorical_cols:
            # Fit and transform categorical columns
            X_train_cat = self.encoder.fit_transform(X_train[categorical_cols])
            X_test_cat = self.encoder.transform(X_test[categorical_cols])
            
            # Get encoded column names
            encoded_colnames = self.encoder.get_feature_names_out(categorical_cols)
            
            # Reconstruct DataFrames with safe index resetting to prevent NaN mismatches
            X_train = pd.concat([
                X_train[numeric_cols].reset_index(drop=True), 
                pd.DataFrame(X_train_cat, columns=encoded_colnames)
            ], axis=1)
            
            X_test = pd.concat([
                X_test[numeric_cols].reset_index(drop=True), 
                pd.DataFrame(X_test_cat, columns=encoded_colnames)
            ], axis=1)

        # Keep column structures intact for tracking
        feature_names = X_train.columns

        # 3. Scale features (Convert back to DataFrame to preserve column headers)
        X_train = pd.DataFrame(self.scaler.fit_transform(X_train), columns=feature_names)
        X_test = pd.DataFrame(self.scaler.transform(X_test), columns=feature_names)

        # 4. Handle class imbalance on training set using SMOTE
        # Flatten y_train to 1D array to avoid warnings or shape errors
        y_train_flat = y_train.values.ravel() if isinstance(y_train, pd.DataFrame) else y_train
        
        X_train_res, y_train_res = self.smote.fit_resample(X_train, y_train_flat)

        # Convert back to DataFrame to keep feature names clean for LightGBM
        X_train_final = pd.DataFrame(X_train_res, columns=feature_names)

        return X_train_final, X_test, y_train_res, y_test.values.ravel()