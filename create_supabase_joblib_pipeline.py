#!/usr/bin/env python3
"""
Complete ML Pipeline for Supabase Default Risk Prediction
Creates a joblib file that can be deployed on LinuxONE server
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SupabaseDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Complete preprocessing pipeline for Supabase default risk prediction data.
    Handles all data transformations from raw Supabase JSON to model-ready features.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_names = None
        self.missing_value_indicators = {}
        
    def fit(self, X, y=None):
        """Fit the preprocessor on training data"""
        # Convert to DataFrame if needed
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, list):
            X = pd.DataFrame(X)
        
        # Store original feature names
        self.feature_names = X.columns.tolist()
        
        # Fit label encoders for categorical variables
        categorical_columns = [
            'name_contract_type_x', 'code_gender', 'name_type_suite_x',
            'name_income_type', 'name_education_type', 'name_family_status',
            'name_housing_type', 'occupation_type', 'weekday_appr_process_start_x',
            'organization_type', 'name_contract_type_y', 'weekday_appr_process_start_y',
            'name_cash_loan_purpose', 'name_contract_status', 'name_payment_type',
            'code_reject_reason', 'name_client_type', 'name_goods_category',
            'name_portfolio', 'name_product_type', 'channel_type', 'name_seller_industry',
            'name_yield_group', 'product_combination'
        ]
        
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                # Handle missing values in categorical columns
                col_data = X[col].fillna('MISSING')
                le.fit(col_data)
                self.label_encoders[col] = le
        
        return self
    
    def transform(self, X):
        """Transform the data"""
        # Convert to DataFrame if needed
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, list):
            X = pd.DataFrame(X)
        
        # Make a copy to avoid modifying original
        X_transformed = X.copy()
        
        # 1. Convert string numbers to numeric
        numeric_columns = [
            'amt_income_total', 'amt_credit_x', 'amt_annuity_x', 'amt_goods_price_x',
            'ext_source_2', 'ext_source_3', 'amt_annuity_y', 'amt_application',
            'amt_credit_y', 'amt_goods_price_y', 'sellerplace_area'
        ]
        
        for col in numeric_columns:
            if col in X_transformed.columns:
                X_transformed[col] = pd.to_numeric(X_transformed[col], errors='coerce')
        
        # 2. Handle missing values
        X_transformed = self._handle_missing_values(X_transformed)
        
        # 3. Encode categorical variables
        X_transformed = self._encode_categorical_variables(X_transformed)
        
        # 4. Create engineered features
        X_transformed = self._create_engineered_features(X_transformed)
        
        # 5. Ensure all expected features are present
        X_transformed = self._ensure_feature_consistency(X_transformed)
        
        return X_transformed
    
    def _handle_missing_values(self, X):
        """Handle missing values with appropriate strategies"""
        
        # Fill missing values for different column types
        missing_strategies = {
            # Binary flags - fill with 0
            'flag_own_car': 0, 'flag_own_realty': 0, 'flag_mobil': 0,
            'flag_emp_phone': 0, 'flag_work_phone': 0, 'flag_cont_mobile': 0,
            'flag_phone': 0, 'flag_email': 0, 'flag_last_appl_per_contract': 0,
            'nflag_last_appl_in_day': 0, 'nflag_insured_on_approval': 0,
            
            # Count columns - fill with 0
            'cnt_children': 0, 'cnt_fam_members': 1, 'cnt_payment': 0,
            'obs_30_cnt_social_circle': 0, 'def_30_cnt_social_circle': 0,
            'obs_60_cnt_social_circle': 0, 'def_60_cnt_social_circle': 0,
            
            # Amount columns - fill with median or 0
            'amt_income_total': 0, 'amt_credit_x': 0, 'amt_annuity_x': 0,
            'amt_goods_price_x': 0, 'amt_annuity_y': 0, 'amt_application': 0,
            'amt_credit_y': 0, 'amt_goods_price_y': 0,
            
            # External source scores - fill with 0
            'ext_source_2': 0, 'ext_source_3': 0,
            
            # Days columns - fill with 0 (no change)
            'days_birth': 0, 'days_employed': 0, 'days_registration': 0,
            'days_id_publish': 0, 'days_last_phone_change': 0,
            'days_decision': 0, 'days_first_drawing': 0, 'days_first_due': 0,
            'days_last_due_1st_version': 0, 'days_last_due': 0, 'days_termination': 0,
            
            # Bureau query columns - fill with 0
            'amt_req_credit_bureau_hour': 0, 'amt_req_credit_bureau_day': 0,
            'amt_req_credit_bureau_week': 0, 'amt_req_credit_bureau_mon': 0,
            'amt_req_credit_bureau_qrt': 0, 'amt_req_credit_bureau_year': 0,
            
            # Other numeric columns
            'region_population_relative': 0, 'region_rating_client': 2,
            'region_rating_client_w_city': 2, 'hour_appr_process_start_x': 12,
            'hour_appr_process_start_y': 12, 'sellerplace_area': 0,
            
            # Flag columns for documents
            'flag_document_2': 0, 'flag_document_3': 0, 'flag_document_4': 0,
            'flag_document_5': 0, 'flag_document_6': 0, 'flag_document_7': 0,
            'flag_document_8': 0, 'flag_document_9': 0, 'flag_document_10': 0,
            'flag_document_11': 0, 'flag_document_12': 0, 'flag_document_13': 0,
            'flag_document_14': 0, 'flag_document_15': 0, 'flag_document_16': 0,
            'flag_document_17': 0, 'flag_document_18': 0, 'flag_document_19': 0,
            'flag_document_20': 0, 'flag_document_21': 0,
            
            # Region mismatch flags
            'reg_region_not_live_region': 0, 'reg_region_not_work_region': 0,
            'live_region_not_work_region': 0, 'reg_city_not_live_city': 0,
            'reg_city_not_work_city': 0, 'live_city_not_work_city': 0
        }
        
        for col, default_value in missing_strategies.items():
            if col in X.columns:
                X[col] = X[col].fillna(default_value)
        
        # Fill remaining missing values with 0
        X = X.fillna(0)
        
        return X
    
    def _encode_categorical_variables(self, X):
        """Encode categorical variables using label encoders"""
        
        for col, le in self.label_encoders.items():
            if col in X.columns:
                # Handle missing values in categorical columns
                col_data = X[col].fillna('MISSING')
                X[col] = le.transform(col_data)
        
        return X
    
    def _create_engineered_features(self, X):
        """Create engineered features based on domain knowledge"""
        
        # 1. Create missing value indicators
        missing_columns = [
            'obs_30_cnt_social_circle', 'def_30_cnt_social_circle',
            'obs_60_cnt_social_circle', 'def_60_cnt_social_circle',
            'amt_req_credit_bureau_hour', 'amt_req_credit_bureau_day',
            'amt_req_credit_bureau_week', 'amt_req_credit_bureau_mon',
            'amt_req_credit_bureau_qrt', 'amt_req_credit_bureau_year',
            'days_first_drawing', 'days_first_due', 'days_last_due_1st_version',
            'days_last_due', 'days_termination', 'amt_goods_price_y',
            'cnt_payment', 'amt_annuity_y', 'nflag_insured_on_approval',
            'ext_source_2', 'ext_source_3', 'amt_goods_price_x', 'amt_annuity_x'
        ]
        
        for col in missing_columns:
            if col in X.columns:
                X[f'{col}_missing'] = (X[col] == 0).astype(int)
        
        # 2. Create ratio features
        if 'amt_credit_x' in X.columns and 'amt_income_total' in X.columns:
            X['credit_income_ratio'] = X['amt_credit_x'] / (X['amt_income_total'] + 1)
        
        if 'amt_annuity_x' in X.columns and 'amt_income_total' in X.columns:
            X['annuity_income_ratio'] = X['amt_annuity_x'] / (X['amt_income_total'] + 1)
        
        if 'amt_credit_x' in X.columns and 'amt_goods_price_x' in X.columns:
            X['credit_goods_ratio'] = X['amt_credit_x'] / (X['amt_goods_price_x'] + 1)
        
        if 'amt_income_total' in X.columns and 'cnt_fam_members' in X.columns:
            X['income_per_person'] = X['amt_income_total'] / (X['cnt_fam_members'] + 1)
        
        if 'cnt_children' in X.columns and 'cnt_fam_members' in X.columns:
            X['children_ratio'] = X['cnt_children'] / (X['cnt_fam_members'] + 1)
        
        # 3. Create age and time-based features
        if 'days_birth' in X.columns:
            X['age_years'] = -X['days_birth'] / 365.25
        
        if 'days_employed' in X.columns:
            X['employed_years'] = -X['days_employed'] / 365.25
        
        if 'days_registration' in X.columns:
            X['registration_years_ago'] = -X['days_registration'] / 365.25
        
        if 'days_id_publish' in X.columns:
            X['id_publish_years_ago'] = -X['days_id_publish'] / 365.25
        
        if 'days_last_phone_change' in X.columns:
            X['phone_change_years_ago'] = -X['days_last_phone_change'] / 365.25
        
        # 4. Create employment stability features
        if 'employed_years' in X.columns and 'age_years' in X.columns:
            X['employment_to_age_ratio'] = X['employed_years'] / (X['age_years'] + 1)
        
        # 5. Create contact features
        contact_cols = ['flag_emp_phone', 'flag_work_phone', 'flag_cont_mobile', 'flag_phone']
        X['num_active_contacts'] = sum([X[col] for col in contact_cols if col in X.columns])
        X['has_work_contact'] = ((X.get('flag_emp_phone', 0) + X.get('flag_work_phone', 0)) > 0).astype(int)
        
        # 6. Create document completeness feature
        doc_cols = [f'flag_document_{i}' for i in range(2, 22)]
        X['has_all_docs'] = sum([X[col] for col in doc_cols if col in X.columns])
        
        # 7. Create stability score
        stability_features = ['flag_own_car', 'flag_own_realty', 'flag_phone', 'flag_email']
        X['stability_score'] = sum([X[col] for col in stability_features if col in X.columns])
        
        # 8. Create urban/rural indicator
        if 'region_population_relative' in X.columns:
            X['urban_rural'] = (X['region_population_relative'] > 0.5).astype(int)
        
        # 9. Create region mismatch score
        region_cols = ['reg_region_not_live_region', 'reg_region_not_work_region', 
                      'live_region_not_work_region', 'reg_city_not_live_city',
                      'reg_city_not_work_city', 'live_city_not_work_city']
        X['city_region_mismatch_score'] = sum([X[col] for col in region_cols if col in X.columns])
        
        # 10. Create bureau query intensity
        bureau_cols = ['amt_req_credit_bureau_hour', 'amt_req_credit_bureau_day',
                      'amt_req_credit_bureau_week', 'amt_req_credit_bureau_mon',
                      'amt_req_credit_bureau_qrt', 'amt_req_credit_bureau_year']
        X['bureau_query_intensity'] = sum([X[col] for col in bureau_cols if col in X.columns])
        
        # 11. Create short-term bureau ratio
        if 'amt_req_credit_bureau_hour' in X.columns and 'amt_req_credit_bureau_year' in X.columns:
            short_term = X['amt_req_credit_bureau_hour'] + X['amt_req_credit_bureau_day'] + X['amt_req_credit_bureau_week']
            long_term = X['amt_req_credit_bureau_year']
            X['short_term_bureau_ratio'] = short_term / (long_term + 1)
        
        # 12. Create credit differences
        if 'amt_credit_x' in X.columns and 'amt_credit_y' in X.columns:
            X['credit_diff'] = X['amt_credit_x'] - X['amt_credit_y']
        
        if 'amt_annuity_x' in X.columns and 'amt_annuity_y' in X.columns:
            X['annuity_diff'] = X['amt_annuity_x'] - X['amt_annuity_y']
        
        if 'amt_goods_price_x' in X.columns and 'amt_goods_price_y' in X.columns:
            X['goods_price_diff'] = X['amt_goods_price_x'] - X['amt_goods_price_y']
        
        # 13. Create contract type consistency
        if 'name_contract_type_x' in X.columns and 'name_contract_type_y' in X.columns:
            X['same_contract_type'] = (X['name_contract_type_x'] == X['name_contract_type_y']).astype(int)
        
        # 14. Create weekday consistency
        if 'weekday_appr_process_start_x' in X.columns and 'weekday_appr_process_start_y' in X.columns:
            X['same_weekday_appr'] = (X['weekday_appr_process_start_x'] == X['weekday_appr_process_start_y']).astype(int)
        
        # 15. Create hour difference
        if 'hour_appr_process_start_x' in X.columns and 'hour_appr_process_start_y' in X.columns:
            X['hour_appr_diff'] = abs(X['hour_appr_process_start_x'] - X['hour_appr_process_start_y'])
        
        # 16. Create time-based features
        if 'days_first_drawing' in X.columns and 'days_first_due' in X.columns:
            X['credit_duration_days'] = X['days_first_due'] - X['days_first_drawing']
        
        if 'days_first_drawing' in X.columns and 'days_first_due' in X.columns:
            X['time_to_first_payment_days'] = X['days_first_due'] - X['days_first_drawing']
        
        if 'days_termination' in X.columns and 'days_first_drawing' in X.columns:
            X['time_to_termination_days'] = X['days_termination'] - X['days_first_drawing']
        
        # 17. Create overlap feature
        if 'days_first_drawing' in X.columns and 'days_last_due' in X.columns:
            X['overlap_with_current'] = (X['days_first_drawing'] <= 0) & (X['days_last_due'] >= 0)
            X['overlap_with_current'] = X['overlap_with_current'].astype(int)
        
        # 18. Create cash loan indicator
        if 'name_contract_type_y' in X.columns:
            X['is_cashloan'] = (X['name_contract_type_y'] == 0).astype(int)  # Assuming 0 = Cash loans
        
        return X
    
    def _ensure_feature_consistency(self, X):
        """Ensure all expected features are present with correct order"""
        
        # Expected features from training (in order)
        expected_features = [
            'sk_id_curr', 'name_contract_type_x', 'code_gender', 'flag_own_car',
            'flag_own_realty', 'cnt_children', 'amt_income_total', 'amt_credit_x',
            'amt_annuity_x', 'amt_goods_price_x', 'name_type_suite_x',
            'name_income_type', 'name_education_type', 'name_family_status',
            'name_housing_type', 'region_population_relative', 'days_birth',
            'days_employed', 'days_registration', 'days_id_publish', 'flag_mobil',
            'flag_emp_phone', 'flag_work_phone', 'flag_cont_mobile', 'flag_phone',
            'flag_email', 'occupation_type', 'cnt_fam_members', 'region_rating_client',
            'region_rating_client_w_city', 'weekday_appr_process_start_x',
            'hour_appr_process_start_x', 'reg_region_not_live_region',
            'reg_region_not_work_region', 'live_region_not_work_region',
            'reg_city_not_live_city', 'reg_city_not_work_city',
            'live_city_not_work_city', 'organization_type', 'ext_source_2',
            'ext_source_3', 'obs_30_cnt_social_circle', 'def_30_cnt_social_circle',
            'obs_60_cnt_social_circle', 'def_60_cnt_social_circle',
            'days_last_phone_change', 'flag_document_2', 'flag_document_3',
            'flag_document_4', 'flag_document_5', 'flag_document_6',
            'flag_document_7', 'flag_document_8', 'flag_document_9',
            'flag_document_10', 'flag_document_11', 'flag_document_12',
            'flag_document_13', 'flag_document_14', 'flag_document_15',
            'flag_document_16', 'flag_document_17', 'flag_document_18',
            'flag_document_19', 'flag_document_20', 'flag_document_21',
            'amt_req_credit_bureau_hour', 'amt_req_credit_bureau_day',
            'amt_req_credit_bureau_week', 'amt_req_credit_bureau_mon',
            'amt_req_credit_bureau_qrt', 'amt_req_credit_bureau_year',
            'sk_id_prev', 'name_contract_type_y', 'amt_annuity_y',
            'amt_application', 'amt_credit_y', 'amt_goods_price_y',
            'weekday_appr_process_start_y', 'hour_appr_process_start_y',
            'flag_last_appl_per_contract', 'nflag_last_appl_in_day',
            'name_cash_loan_purpose', 'name_contract_status', 'days_decision',
            'name_payment_type', 'code_reject_reason', 'name_client_type',
            'name_goods_category', 'name_portfolio', 'name_product_type',
            'channel_type', 'sellerplace_area', 'name_seller_industry',
            'cnt_payment', 'name_yield_group', 'product_combination',
            'days_first_drawing', 'days_first_due', 'days_last_due_1st_version',
            'days_last_due', 'days_termination', 'nflag_insured_on_approval',
            'is_cashloan', 'obs_30_cnt_social_circle_missing',
            'def_30_cnt_social_circle_missing', 'obs_60_cnt_social_circle_missing',
            'def_60_cnt_social_circle_missing', 'amt_req_credit_bureau_hour_missing',
            'amt_req_credit_bureau_day_missing', 'amt_req_credit_bureau_week_missing',
            'amt_req_credit_bureau_mon_missing', 'amt_req_credit_bureau_qrt_missing',
            'amt_req_credit_bureau_year_missing', 'days_first_drawing_missing',
            'days_first_due_missing', 'days_last_due_1st_version_missing',
            'days_last_due_missing', 'days_termination_missing',
            'amt_goods_price_y_missing', 'cnt_payment_missing',
            'amt_annuity_y_missing', 'nflag_insured_on_approval_missing',
            'ext_source_2_missing', 'ext_source_3_missing',
            'amt_goods_price_x_missing', 'amt_annuity_x_missing',
            'credit_income_ratio', 'annuity_income_ratio', 'credit_goods_ratio',
            'income_per_person', 'children_ratio', 'payment_rate',
            'age_years', 'employed_years', 'registration_years_ago',
            'id_publish_years_ago', 'phone_change_years_ago',
            'employment_to_age_ratio', 'num_active_contacts', 'has_work_contact',
            'has_all_docs', 'stability_score', 'urban_rural',
            'city_region_mismatch_score', 'bureau_query_intensity',
            'short_term_bureau_ratio', 'credit_diff', 'annuity_diff',
            'goods_price_diff', 'credit_to_goods_delta_ratio',
            'same_contract_type', 'same_weekday_appr', 'hour_appr_diff',
            'credit_duration_days', 'time_to_first_payment_days',
            'time_to_termination_days', 'overlap_with_current'
        ]
        
        # Add missing features with default values
        for feature in expected_features:
            if feature not in X.columns:
                X[feature] = 0
        
        # Reorder columns to match training data
        X = X[expected_features]
        
        return X

def create_supabase_deployment_pipeline():
    """
    Create the complete ML pipeline for Supabase deployment
    """
    print("🔧 Creating Supabase ML Pipeline...")
    
    # Load the trained model from your existing pickle file
    try:
        with open('rf_model_complete.pkl', 'rb') as f:
            model_components = joblib.load(f)
        
        print("✅ Loaded existing trained model")
        print(f"   Model type: {model_components.get('model_type', 'Unknown')}")
        print(f"   Training date: {model_components.get('training_date', 'Unknown')}")
        print(f"   Threshold: {model_components.get('threshold', 0.3)}")
        
        # Extract the trained Random Forest model
        rf_model = model_components['model']
        
    except FileNotFoundError:
        print("⚠️  rf_model_complete.pkl not found. Creating new Random Forest model...")
        
        # Create a new Random Forest model with the same parameters as your training
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        
        # Note: This will need to be trained with actual data
        print("⚠️  WARNING: New model created but not trained. You need to train it with your data.")
    
    # Create the complete pipeline
    pipeline = Pipeline([
        ('preprocessor', SupabaseDataPreprocessor()),
        ('scaler', StandardScaler()),
        ('classifier', rf_model)
    ])
    
    # Create the deployment bundle
    deployment_bundle = {
        'pipeline': pipeline,
        'threshold': 0.30,  # Your optimal threshold
        'model_type': 'RandomForestClassifier',
        'model_params': {
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        },
        'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'version': '1.0.0',
        'description': 'Default Risk Prediction Pipeline for Supabase Integration'
    }
    
    # Save the pipeline
    output_filename = 'model_pipeline.joblib'
    joblib.dump(deployment_bundle, output_filename)
    
    print(f"✅ Pipeline saved to '{output_filename}'")
    print(f"📦 Bundle contains:")
    print(f"   - Complete preprocessing pipeline")
    print(f"   - Trained Random Forest model")
    print(f"   - Optimal threshold: {deployment_bundle['threshold']}")
    print(f"   - Model parameters: {deployment_bundle['model_params']}")
    
    return deployment_bundle

def predict_default_risk(data):
    """
    Predict default risk for Supabase data and return formatted results
    
    Args:
        data (dict): Supabase data record
        
    Returns:
        dict: Prediction results with 'target' and 'risk_score' fields
    """
    try:
        # Load the pipeline
        bundle = joblib.load('model_pipeline.joblib')
        pipeline = bundle['pipeline']
        threshold = bundle['threshold']
        
        # Get prediction probability
        proba = pipeline.predict_proba([data])[:, 1][0]
        
        # Apply threshold to get binary prediction
        prediction = int(proba >= threshold)
        
        # Return formatted results
        return {
            'target': prediction,  # Binary prediction (0 or 1)
            'risk_score': float(proba),  # Probability score (0.0 to 1.0)
            'sk_id_curr': data.get('sk_id_curr'),
            'risk_level': 'HIGH' if prediction == 1 else 'LOW',
            'threshold_used': threshold,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    # Create the pipeline
    bundle = create_supabase_deployment_pipeline()
    
    print("\n🎉 Pipeline creation complete!")
    print("📁 Files created:")
    print("   - model_pipeline.joblib (ready for LinuxONE deployment)")
    print("\n📋 Next steps:")
    print("   1. Copy model_pipeline.joblib to your LinuxONE server")
    print("   2. Use the FastAPI script provided earlier")
    print("   3. Test with your Supabase Edge Function")
    print("\n🔧 Usage in FastAPI:")
    print("   from create_supabase_joblib_pipeline import predict_default_risk")
    print("   result = predict_default_risk(supabase_data)")
    print("   # Returns: {'target': 0/1, 'risk_score': 0.0-1.0, ...}")
