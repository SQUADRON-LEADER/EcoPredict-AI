import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_input(input_df):
    """
    Preprocess input data for the GHG emissions prediction model.
    
    Args:
        input_df (pd.DataFrame): Input dataframe with raw features
        
    Returns:
        pd.DataFrame: Preprocessed dataframe ready for model prediction
    """
    # Create a copy to avoid modifying the original
    df = input_df.copy()
    
    # Define expected columns and their order (based on typical ML model training)
    expected_columns = [
        'Substance',
        'Unit', 
        'Supply Chain Emission Factors without Margins',
        'Margins of Supply Chain Emission Factors',
        'DQ ReliabilityScore of Factors without Margins',
        'DQ TemporalCorrelation of Factors without Margins',
        'DQ GeographicalCorrelation of Factors without Margins',
        'DQ TechnologicalCorrelation of Factors without Margins',
        'DQ DataCollection of Factors without Margins',
        'Source'
    ]
    
    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in df.columns:
            # Add missing columns with default values
            if col in ['Substance', 'Unit', 'Source']:
                df[col] = 'Unknown'
            else:
                df[col] = 0.0
    
    # Label encode categorical variables
    label_encoders = {}
    categorical_columns = ['Substance', 'Unit', 'Source']
    
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            # Define expected categories to handle unseen values
            if col == 'Substance':
                expected_categories = ['carbon dioxide', 'methane', 'nitrous oxide', 'other GHGs']
            elif col == 'Unit':
                expected_categories = ['kg/2018 USD, purchaser price', 'kg CO2e/2018 USD, purchaser price']
            elif col == 'Source':
                expected_categories = ['Commodity', 'Industry']
            else:
                expected_categories = df[col].unique().tolist()
            
            # Fit the encoder with expected categories
            le.fit(expected_categories)
            
            # Transform the data, handling unseen categories
            df[col] = df[col].apply(lambda x: x if x in expected_categories else expected_categories[0])
            df[col] = le.transform(df[col])
            
            label_encoders[col] = le
    
    # Ensure numeric columns are properly typed
    numeric_columns = [
        'Supply Chain Emission Factors without Margins',
        'Margins of Supply Chain Emission Factors',
        'DQ ReliabilityScore of Factors without Margins',
        'DQ TemporalCorrelation of Factors without Margins',
        'DQ GeographicalCorrelation of Factors without Margins',
        'DQ TechnologicalCorrelation of Factors without Margins',
        'DQ DataCollection of Factors without Margins'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # Reorder columns to match expected order (important for some models)
    df = df[expected_columns]
    
    return df

def reverse_preprocess_output(prediction, label_encoders=None):
    """
    Convert model output back to interpretable format if needed.
    
    Args:
        prediction: Model prediction output
        label_encoders: Dictionary of label encoders used in preprocessing
        
    Returns:
        Processed prediction
    """
    # For regression models, usually no reverse preprocessing needed
    # Just ensure it's a proper numeric value
    if hasattr(prediction, '__iter__') and not isinstance(prediction, str):
        return float(prediction[0]) if len(prediction) > 0 else 0.0
    else:
        return float(prediction)

# Additional utility functions
def validate_input_data(input_data):
    """
    Validate input data before preprocessing.
    
    Args:
        input_data (dict): Input data dictionary
        
    Returns:
        bool: True if valid, False otherwise
        str: Error message if invalid
    """
    required_fields = [
        'Substance',
        'Unit',
        'Supply Chain Emission Factors without Margins',
        'Margins of Supply Chain Emission Factors',
        'Source'
    ]
    
    # Check required fields
    for field in required_fields:
        if field not in input_data:
            return False, f"Missing required field: {field}"
        
        if input_data[field] is None:
            return False, f"Field cannot be None: {field}"
    
    # Validate numeric fields
    numeric_fields = [
        'Supply Chain Emission Factors without Margins',
        'Margins of Supply Chain Emission Factors'
    ]
    
    for field in numeric_fields:
        if field in input_data:
            try:
                val = float(input_data[field])
                if val < 0:
                    return False, f"Field must be non-negative: {field}"
            except (ValueError, TypeError):
                return False, f"Field must be numeric: {field}"
    
    # Validate DQ scores (should be between 0 and 1)
    dq_fields = [
        'DQ ReliabilityScore of Factors without Margins',
        'DQ TemporalCorrelation of Factors without Margins',
        'DQ GeographicalCorrelation of Factors without Margins',
        'DQ TechnologicalCorrelation of Factors without Margins',
        'DQ DataCollection of Factors without Margins'
    ]
    
    for field in dq_fields:
        if field in input_data:
            try:
                val = float(input_data[field])
                if not (0.0 <= val <= 1.0):
                    return False, f"DQ field must be between 0 and 1: {field}"
            except (ValueError, TypeError):
                return False, f"DQ field must be numeric: {field}"
    
    return True, "Valid"

def get_feature_importance_info():
    """
    Return information about feature importance for user guidance.
    
    Returns:
        dict: Feature importance information
    """
    return {
        "most_important": [
            "Supply Chain Emission Factors without Margins",
            "Margins of Supply Chain Emission Factors",
            "DQ ReliabilityScore of Factors without Margins"
        ],
        "categorical_features": [
            "Substance",
            "Unit", 
            "Source"
        ],
        "quality_features": [
            "DQ ReliabilityScore of Factors without Margins",
            "DQ TemporalCorrelation of Factors without Margins",
            "DQ GeographicalCorrelation of Factors without Margins",
            "DQ TechnologicalCorrelation of Factors without Margins",
            "DQ DataCollection of Factors without Margins"
        ],
        "tips": {
            "substance": "Carbon dioxide typically has different emission factors than methane or nitrous oxide",
            "unit": "CO2e units include all greenhouse gases converted to CO2 equivalent",
            "reliability": "Higher reliability scores lead to more confident predictions",
            "margins": "Safety margins should typically be 5-20% of base emission factors"
        }
    }
