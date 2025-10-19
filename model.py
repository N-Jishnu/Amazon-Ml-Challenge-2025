import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib

def load_features(dataset_folder):
    train = pd.read_csv(os.path.join(dataset_folder, 'train_features.csv'))
    test = pd.read_csv(os.path.join(dataset_folder, 'test_features.csv'))
    return train, test

def get_feature_columns(df):
    # Select all columns starting with text_emb_ or image_emb_
    feature_cols = [col for col in df.columns if col.startswith('text_emb_') or col.startswith('image_emb_')]
    return feature_cols

def train_model(train_df):
    feature_cols = get_feature_columns(train_df)
    X = train_df[feature_cols]
    y = train_df['price']
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    model = GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_preds)
    # Additional metrics
    def smape(y_true, y_pred):
        return np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100
    val_smape = smape(y_val, val_preds)
    from sklearn.metrics import mean_squared_error, r2_score
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    val_r2 = r2_score(y_val, val_preds)
    print(f'Validation MAE: {val_mae:.4f}')
    print(f'Validation SMAPE: {val_smape:.2f}%')
    print(f'Validation RMSE: {val_rmse:.4f}')
    print(f'Validation R²: {val_r2:.4f}')
    joblib.dump(model, 'product_price_model.pkl')
    return model

def main():
    DATASET_FOLDER = 'dataset/'
    train_df, test_df = load_features(DATASET_FOLDER)
    print('Training model...')
    model = train_model(train_df)
    print('Model training complete.')

if __name__ == "__main__":
    main()