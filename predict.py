import os
import pandas as pd
import joblib

def get_feature_columns(df):
    feature_cols = [col for col in df.columns if col.startswith('text_emb_') or col.startswith('image_emb_')]
    return feature_cols

def main():
    DATASET_FOLDER = 'dataset/'
    test_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'test_features.csv'))
    model = joblib.load('product_price_model.pkl')
    feature_cols = get_feature_columns(test_df)
    X_test = test_df[feature_cols]
    preds = model.predict(X_test)
    output_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': preds})
    output_df['price'] = output_df['price'].clip(lower=0.01)  # Ensure positive prices
    output_df.to_csv(os.path.join(DATASET_FOLDER, 'test_out.csv'), index=False)
    print('Predictions saved to test_out.csv')
    print(output_df.head())

if __name__ == "__main__":
    main()