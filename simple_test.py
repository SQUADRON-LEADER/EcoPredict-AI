import pandas as pd
import joblib
import numpy as np

def preprocess_input(input_df):
    """Preprocess input data for the GHG emissions prediction model."""
    df = input_df.copy()
    
    # Map categorical variables to numerical values
    substance_map = {'carbon dioxide': 0, 'methane': 1, 'nitrous oxide': 2, 'other GHGs': 3}
    unit_map = {'kg/2018 USD, purchaser price': 0, 'kg CO2e/2018 USD, purchaser price': 1}
    source_map = {'Commodity': 0, 'Industry': 1}
    
    # Apply mappings
    df['Substance'] = df['Substance'].map(substance_map)
    df['Unit'] = df['Unit'].map(unit_map)
    df['Source'] = df['Source'].map(source_map)
    
    # Column order as used in training (after dropping Name, Code, Year, and target variable)
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
    
    return df[expected_columns]

# Test the prediction pipeline
try:
    print("Loading models...")
    model = joblib.load('models/LR_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Models loaded successfully")
    
    # Test input data
    input_data = {
        'Substance': 'carbon dioxide',
        'Unit': 'kg/2018 USD, purchaser price',
        'Supply Chain Emission Factors without Margins': 0.1,
        'Margins of Supply Chain Emission Factors': 0.01,
        'DQ ReliabilityScore of Factors without Margins': 0.5,
        'DQ TemporalCorrelation of Factors without Margins': 0.5,
        'DQ GeographicalCorrelation of Factors without Margins': 0.5,
        'DQ TechnologicalCorrelation of Factors without Margins': 0.5,
        'DQ DataCollection of Factors without Margins': 0.5,
        'Source': 'Industry',
    }
    
    print("Preprocessing input...")
    input_df = preprocess_input(pd.DataFrame([input_data]))
    print("✅ Input preprocessed")
    print("Input shape:", input_df.shape)
    print("Input columns:", input_df.columns.tolist())
    print("Input values:", input_df.iloc[0].values)
    
    print("Scaling input...")
    input_scaled = scaler.transform(input_df)
    print("✅ Input scaled")
    print("Scaled shape:", input_scaled.shape)
    
    print("Making prediction...")
    prediction = model.predict(input_scaled)
    print("✅ Prediction successful!")
    print("Prediction:", prediction[0])
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
