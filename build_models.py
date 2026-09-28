import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib

sys.stdout.reconfigure(encoding='utf-8')

print("=== BUILDING & EVALUATING MAORS ML MODULE ===")
df = pd.read_excel('maternal_ai_dataset.xlsx')

# Feature Engineering
def engineer_features(data):
    df_feat = data.copy()
    if 'Height' in df_feat.columns and 'Weight' in df_feat.columns:
        height_m = df_feat['Height'] / 100.0
        df_feat['BMI'] = df_feat['Weight'] / (height_m ** 2)
    
    if 'BP_sys' in df_feat.columns and 'BP_dia' in df_feat.columns:
        df_feat['MAP'] = (2 * df_feat['BP_dia'] + df_feat['BP_sys']) / 3.0
        df_feat['Pulse_Pressure'] = df_feat['BP_sys'] - df_feat['BP_dia']
        
    if 'GA_weeks' in df_feat.columns and 'Fundal_Height' in df_feat.columns:
        df_feat['Fundal_Lag'] = df_feat['GA_weeks'] - df_feat['Fundal_Height']
        
    if 'Baby_Position' in df_feat.columns:
        df_feat['Baby_Position_NotHead'] = (df_feat['Baby_Position'] == 'NotHead').astype(int)
        df_feat = df_feat.drop(columns=['Baby_Position'])
        
    return df_feat

df_engineered = engineer_features(df)

# Targets
targets = ['Outcome_Preterm', 'Outcome_LBW', 'Outcome_Preeclampsia']
feature_cols = [c for c in df_engineered.columns if c not in targets and c != 'ID']

print(f"Total features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

os.makedirs('models', exist_ok=True)

# Save feature statistics (mean and std) for dynamic driver calculation
feature_stats = {
    'mean': df_engineered[feature_cols].mean().to_dict(),
    'std': df_engineered[feature_cols].std().to_dict()
}
joblib.dump(feature_stats, 'models/feature_stats.joblib')

trained_models = {}

for target in targets:
    print(f"\n==========================================")
    print(f"Target: {target} (Class 1 rate: {df[target].mean():.1%})")
    print(f"==========================================")
    
    X = df_engineered[feature_cols]
    y = df_engineered[target]
    
    # We will use an ImbPipeline that applies SMOTE then a GradientBoostingClassifier
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42))
    ])
    
    # Define hyperparameter grid
    param_grid = {
        'gb__n_estimators': [100, 150],
        'gb__learning_rate': [0.05, 0.1],
        'gb__max_depth': [3, 4]
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Running GridSearchCV with SMOTE & GradientBoosting...")
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=skf, 
        scoring='roc_auc',
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.3f}")
    
    # The actual GradientBoosting model is the 'gb' step in the pipeline
    final_gb_model = best_model.named_steps['gb']
    
    # Save the pipeline so we don't need to re-apply SMOTE during inference (wait, SMOTE is only during fit)
    # However, for prediction we only need the final classifier. We can just save the fitted pipeline or the classifier.
    # The current ml_engine.py expects just a classifier with predict_proba. ImbPipeline provides predict_proba directly.
    joblib.dump(best_model, f'models/{target}_model.joblib')
    trained_models[target] = best_model
    
    # Feature importances from the underlying GB model
    importances = pd.Series(final_gb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\nTop 8 Features for {target}:")
    for feat, imp in importances.head(8).items():
        print(f"  {feat:25}: {imp:.4f}")

# Save feature list
joblib.dump(feature_cols, 'models/feature_cols.joblib')
print("\nAll models, feature statistics, and metadata saved to ./models/")
