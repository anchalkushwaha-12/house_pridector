import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

def train_and_evaluate_models():
    base_dir = r"C:\Users\Anchal Kushwaha\.gemini\antigravity\scratch\house_price_prediction"
    data_path = os.path.join(base_dir, "data", "house_prices.csv")
    models_dir = os.path.join(base_dir, "models")
    assets_dir = os.path.join(base_dir, "reports", "assets")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    X = df.drop(columns=['Price'])
    y = df['Price']

    # Define feature groups
    num_features = ['Area(sqft)', 'Rooms', 'Bathrooms', 'Floors', 'YearBuilt', 'Parking']
    cat_features = ['Location', 'Condition']

    # 2. Build Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define candidate models
    candidate_models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
    }

    results = {}
    best_r2 = -float('inf')
    best_model_name = None
    best_pipeline = None

    print("\n--- Training and Evaluating Models ---")
    for name, regressor in candidate_models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        results[name] = {
            'R2': float(r2),
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MAPE': float(mape)
        }
        
        print(f"{name:20s} -> R2: {r2:.4f} | MAE: ${mae:,.2f} | RMSE: ${rmse:,.2f} | MAPE: {mape:.2f}%")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline

    print(f"\nBest Model: {best_model_name} with R2 Score: {best_r2:.4f}")

    # 3. Save Best Model Pipeline
    model_save_path = os.path.join(models_dir, "house_price_model.joblib")
    joblib.dump(best_pipeline, model_save_path)
    print(f"Saved best model pipeline to {model_save_path}")

    # Save metrics JSON
    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            'best_model': best_model_name,
            'all_models': results,
            'test_sample_count': len(y_test)
        }, f, indent=4)
    print(f"Saved metrics summary to {metrics_path}")

    # 4. Generate & Save Visualizations
    best_pred = best_pipeline.predict(X_test)
    residuals = y_test - best_pred

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Plot 1: Actual vs Predicted
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test / 1000, best_pred / 1000, alpha=0.7, color='#2b5c8f', edgecolors='k')
    ax.plot([y_test.min() / 1000, y_test.max() / 1000], [y_test.min() / 1000, y_test.max() / 1000], 'r--', lw=2, label='Ideal 1:1 Line')
    ax.set_title(f'Actual vs Predicted House Prices ({best_model_name})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual Price ($ in Thousands)', fontsize=12)
    ax.set_ylabel('Predicted Price ($ in Thousands)', fontsize=12)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(assets_dir, "actual_vs_predicted.png"), dpi=300)
    plt.close()

    # Plot 2: Residuals Distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals / 1000, kde=True, color='#28a745', ax=ax, bins=30)
    ax.set_title('Residuals Distribution (Errors in $K)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Prediction Error ($ in Thousands)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(assets_dir, "residual_plot.png"), dpi=300)
    plt.close()

    # Plot 3: Feature Importance (if tree-based model)
    regressor = best_pipeline.named_steps['regressor']
    if hasattr(regressor, 'feature_importances_'):
        cat_encoder = best_pipeline.named_steps['preprocessor'].named_transformers_['cat']
        encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_features))
        all_feature_names = num_features + encoded_cat_names
        
        importances = regressor.feature_importances_
        feat_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis', ax=ax)
        ax.set_title(f'Feature Importance ({best_model_name})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Relative Importance', fontsize=12)
        plt.tight_layout()
        fig.savefig(os.path.join(assets_dir, "feature_importance.png"), dpi=300)
        plt.close()
        print("Feature importance plot generated successfully.")

if __name__ == '__main__':
    train_and_evaluate_models()
