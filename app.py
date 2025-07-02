import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Import preprocessing function
try:
    from utils.preprocessor import preprocess_input, validate_input_data
except ImportError:
    st.error("❌ Error: Missing preprocessor module. Please ensure utils/preprocessor.py exists.")
    st.stop()

# Import plotting libraries with fallback
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("📊 Plotly not available. Charts will use fallback display.")

# Page Configuration
st.set_page_config(
    page_title="EcoPredict AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and scaler with error handling
@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/LR_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler, True
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        return None, None, False
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, False

model, scaler, models_loaded = load_models()

if not models_loaded:
    st.stop()

# Helper Functions
def create_gauge_chart(value, title, color):
    """Create a gauge chart for metrics"""
    if not PLOTLY_AVAILABLE:
        return None
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20, 'color': '#2C3E50'}},
        gauge = {
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.4], 'color': 'lightgray'},
                {'range': [0.4, 0.7], 'color': 'yellow'},
                {'range': [0.7, 1], 'color': 'lightgreen'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9}}))
    
    fig.update_layout(
        height=300,
        font={'color': "#2C3E50"},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_emission_bar_chart(prediction_value):
    """Create a bar chart showing emission levels"""
    if not PLOTLY_AVAILABLE:
        return None
    
    categories = ['Low Impact\n(<0.05)', 'Medium Impact\n(0.05-0.1)', 'High Impact\n(>0.1)', 'Your Prediction']
    values = [0.03, 0.075, 0.15, prediction_value]
    colors = ['#2ECC71', '#F39C12', '#E74C3C', '#9B59B6']
    
    fig = go.Figure(data=[
        go.Bar(x=categories, y=values, marker_color=colors, text=[f'{v:.4f}' for v in values], textposition='auto')
    ])
    
    fig.update_layout(
        title={'text': '🌍 Emission Factor Comparison', 'x': 0.5, 'font': {'size': 20, 'color': '#2C3E50'}},
        xaxis_title="Impact Categories",
        yaxis_title="Emission Factor (kg CO2e/USD)",
        template="plotly_white",
        height=400,
        showlegend=False
    )
    return fig

def create_radar_chart(dq_scores):
    """Create radar chart for data quality"""
    if not PLOTLY_AVAILABLE:
        return None
    
    categories = ['Reliability', 'Temporal', 'Geographic', 'Technology', 'Data Collection']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=dq_scores,
        theta=categories,
        fill='toself',
        name='Data Quality',
        line=dict(color='#27AE60', width=3),
        fillcolor='rgba(39, 174, 96, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=12),
                gridcolor='rgba(46, 204, 113, 0.3)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        title={
            'text': '📊 Data Quality Assessment',
            'x': 0.5,
            'font': {'size': 18, 'color': '#2C3E50'}
        },
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_fallback_chart_display(prediction_value, dq_scores, overall_quality):
    """Create fallback display when Plotly is not available"""
    # Emission comparison display
    st.markdown("""
    <div class="modern-card">
        <h3 style="color: #2C3E50; text-align: center; margin-bottom: 1rem;">🌍 Emission Factor Comparison</h3>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
    """, unsafe_allow_html=True)
    
    categories = [
        ('Low Impact', 0.03, '#2ECC71', '🟢'),
        ('Medium Impact', 0.075, '#F39C12', '🟡'),
        ('High Impact', 0.15, '#E74C3C', '🔴'),
        ('Your Prediction', prediction_value, '#9B59B6', '🎯')
    ]
    
    for name, value, color, emoji in categories:
        st.markdown(f"""
        <div style="background: {color}; color: white; padding: 1.5rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{emoji}</div>
            <h4 style="margin: 0; color: white; font-size: 0.9rem;">{name}</h4>
            <h2 style="margin: 0.5rem 0; color: white; font-size: 1.5rem;">{value:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Data quality display
    st.markdown("""
    <div class="modern-card">
        <h3 style="color: #2C3E50; text-align: center; margin-bottom: 1rem;">📊 Data Quality Metrics</h3>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem;">
    """, unsafe_allow_html=True)
    
    dq_categories = [
        ('🎯 Reliability', dq_scores[0]),
        ('⏰ Temporal', dq_scores[1]), 
        ('🌍 Geographic', dq_scores[2]),
        ('🔬 Technology', dq_scores[3]),
        ('📊 Collection', dq_scores[4])
    ]
    
    for name, value in dq_categories:
        color = '#2ECC71' if value >= 0.7 else '#F39C12' if value >= 0.4 else '#E74C3C'
        st.markdown(f"""
        <div style="background: {color}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
            <h4 style="margin: 0; color: white; font-size: 0.8rem;">{name}</h4>
            <h2 style="margin: 0.5rem 0; color: white;">{value:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

# Modern CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* Simplified Global Styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        font-family: 'Poppins', sans-serif;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Completely hide and override unwanted Streamlit emotion cache classes */
    .st-emotion-cache-iwohja {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
        overflow: hidden !important;
        position: absolute !important;
        left: -9999px !important;
        background: none !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        font-size: 0 !important;
        line-height: 0 !important;
    }
    
    .e1chbk302 {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
        overflow: hidden !important;
        position: absolute !important;
        left: -9999px !important;
    }
    
    /* Hide any elements with these class patterns */
    [class*="st-emotion-cache-iwohja"],
    [class*="e1chbk302"],
    div[class*="st-emotion-cache-iwohja"],
    div[class*="e1chbk302"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
        overflow: hidden !important;
        position: absolute !important;
        left: -9999px !important;
    }
    
    /* Override any code block styling that might leak through */
    pre, code {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 8px !important;
        color: #2C3E50 !important;
        font-family: 'Poppins', sans-serif !important;
        padding: 0.5rem !important;
    }
    
    /* Fix monospace fonts in specific elements */
    .st-emotion-cache-hpex6h {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700 !important;
        color: #2C3E50 !important;
    }
    
    /* Simplified Modern Card Design */
    .modern-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: all 0.3s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.5);
    }
    
    /* Simplified Header Styling */
    .main-header {
        text-align: center;
        margin: 2rem 0;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Simplified Input Styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid #27AE60 !important;
        border-radius: 15px !important;
        transition: all 0.3s ease !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #2ECC71 !important;
        box-shadow: 0 0 15px rgba(46, 204, 113, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Selectbox text and options styling */
    .stSelectbox > div > div > div {
        color: #2C3E50 !important;
        font-weight: 600 !important;
    }
    
    .stSelectbox > div > div > div > div {
        color: #2C3E50 !important;
        font-weight: 600 !important;
    }
    
    /* Selected value text */
    .stSelectbox [data-baseweb="select"] > div {
        color: #2C3E50 !important;
        font-weight: 600 !important;
    }
    
    /* Dropdown options */
    .stSelectbox [data-baseweb="popover"] {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        border: 2px solid #27AE60 !important;
    }
    
    .stSelectbox [data-baseweb="popover"] li {
        color: #2C3E50 !important;
        font-weight: 600 !important;
        background: transparent !important;
    }
    
    .stSelectbox [data-baseweb="popover"] li:hover {
        background: rgba(46, 204, 113, 0.1) !important;
        color: #27AE60 !important;
    }
    
    /* Fix for selected option display */
    .stSelectbox div[data-testid="stSelectbox"] > div > div {
        color: #2C3E50 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Fix selectbox text visibility */
    .stSelectbox > div > div > div {
        color: #2C3E50 !important;
        font-weight: 600 !important;
    }
    
    .stSelectbox > div > div > div[data-baseweb="select"] > div {
        color: #2C3E50 !important;
        font-weight: 600 !important;
    }
    
    /* Ensure selected text is visible */
    .stSelectbox > div > div input {
        color: #2C3E50 !important;
    }
    
    /* Fix dropdown option text */
    .stSelectbox div[data-baseweb="select"] > div {
        color: #2C3E50 !important;
        font-weight: 600 !important;
    }
    
    /* Enhanced dropdown options visibility */
    div[data-baseweb="popover"] {
        background: rgba(255,255,255,0.98) !important;
        backdrop-filter: blur(20px) !important;
        border: 2px solid #27AE60 !important;
        border-radius: 15px !important;
        box-shadow: 0 20px 40px rgba(0,0,0,0.15) !important;
        z-index: 9999 !important;
    }
    
    div[data-baseweb="popover"] li {
        background: transparent !important;
        color: #2C3E50 !important;
        font-weight: 600 !important;
        padding: 12px 20px !important;
        transition: all 0.2s ease !important;
    }
    
    div[data-baseweb="popover"] li:hover {
        background: linear-gradient(135deg, #27AE60, #2ECC71) !important;
        color: white !important;
        transform: translateX(5px) !important;
    }
    
    div[data-baseweb="popover"] li[aria-selected="true"] {
        background: linear-gradient(135deg, #27AE60, #2ECC71) !important;
        color: white !important;
        font-weight: 700 !important;
    }
    
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid #3498DB !important;
        border-radius: 15px !important;
        color: #2C3E50 !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput > div > div > input:hover {
        border-color: #5DADE2 !important;
        box-shadow: 0 0 15px rgba(52, 152, 219, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Enhanced Slider Styling with Better Value Display */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(248, 249, 250, 0.95)) !important;
        border: 4px solid transparent !important;
        background-clip: padding-box !important;
        border-radius: 25px !important;
        padding: 2rem !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.12),
            0 2px 16px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.5) !important;
        backdrop-filter: blur(20px) !important;
        position: relative !important;
        overflow: visible !important;
    }
    
    .stSlider > div > div > div::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 25px;
        padding: 4px;
        background: linear-gradient(135deg, #E67E22, #F39C12, #F7DC6F);
        mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        mask-composite: xor;
        -webkit-mask-composite: xor;
    }
    
    .stSlider > div > div > div:hover {
        transform: translateY(-8px) scale(1.02) !important;
        box-shadow: 
            0 20px 60px rgba(230, 126, 34, 0.25),
            0 8px 32px rgba(0, 0, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.7) !important;
        background: linear-gradient(135deg, rgba(255, 255, 255, 1), rgba(255, 251, 230, 0.98)) !important;
    }
    
    /* Fix slider value display visibility */
    .st-emotion-cache-hpex6h {
        line-height: 1.4 !important;
        font-weight: 800 !important;
        font-family: 'Poppins', sans-serif !important;
        color: #2C3E50 !important;
        font-size: 1.3rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1) !important;
        background: linear-gradient(135deg, #E67E22, #F39C12) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        display: block !important;
        text-align: center !important;
        padding: 0.8rem 1.5rem !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
        position: relative !important;
        margin-bottom: 1rem !important;
    }
    
    .st-emotion-cache-hpex6h::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(230, 126, 34, 0.1), rgba(243, 156, 18, 0.1));
        border-radius: 15px;
        border: 2px solid rgba(230, 126, 34, 0.3);
        z-index: -1;
    }
    
    /* Enhanced slider value display with animation */
    .stSlider > div > div > div > div:first-child {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        border-radius: 20px !important;
        padding: 1.2rem 2rem !important;
        margin-bottom: 2rem !important;
        border: 3px solid rgba(255, 255, 255, 0.4) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
        backdrop-filter: blur(15px) !important;
        text-align: center !important;
        color: white !important;
        font-weight: 800 !important;
        font-size: 1.4rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5) !important;
        transition: all 0.3s ease !important;
    }
    
    .stSlider > div > div > div > div:first-child:hover {
        background: linear-gradient(135deg, #764ba2, #667eea) !important;
        transform: scale(1.05) !important;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Slider track styling */
    .stSlider > div > div > div > div > div > div {
        height: 8px !important;
        border-radius: 10px !important;
        background: linear-gradient(90deg, #E74C3C 0%, #F39C12 50%, #27AE60 100%) !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Slider thumb (handle) styling */
    .stSlider > div > div > div > div > div > div > div {
        width: 24px !important;
        height: 24px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #E67E22, #F39C12) !important;
        border: 3px solid white !important;
        box-shadow: 0 4px 12px rgba(230, 126, 34, 0.6) !important;
        transition: all 0.2s ease !important;
        cursor: grab !important;
    }
    
    .stSlider > div > div > div > div > div > div > div:hover {
        transform: scale(1.3) !important;
        box-shadow: 0 6px 20px rgba(230, 126, 34, 0.8) !important;
        cursor: grabbing !important;
    }
    
    .stSlider > div > div > div > div > div > div > div:active {
        transform: scale(1.4) !important;
        box-shadow: 0 8px 25px rgba(230, 126, 34, 1) !important;
        cursor: grabbing !important;
    }
    
    /* Slider value display enhancement */
    .stSlider > div > div > div > div:first-child {
        background: rgba(230, 126, 34, 0.1) !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
        margin-bottom: 1rem !important;
        border: 2px solid rgba(230, 126, 34, 0.3) !important;
    }
    
    /* Individual slider customization by data attribute */
    .stSlider[data-testid*="stSlider"]:nth-of-type(1) > div > div > div > div > div > div > div {
        background: linear-gradient(135deg, #27AE60, #2ECC71) !important;
        box-shadow: 0 4px 12px rgba(39, 174, 96, 0.6) !important;
    }
    
    .stSlider[data-testid*="stSlider"]:nth-of-type(2) > div > div > div > div > div > div > div {
        background: linear-gradient(135deg, #3498DB, #5DADE2) !important;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.6) !important;
    }
    
    .stSlider[data-testid*="stSlider"]:nth-of-type(3) > div > div > div > div > div > div > div {
        background: linear-gradient(135deg, #9B59B6, #BB8FCE) !important;
        box-shadow: 0 4px 12px rgba(155, 89, 182, 0.6) !important;
    }
    
    .stSlider[data-testid*="stSlider"]:nth-of-type(4) > div > div > div > div > div > div > div {
        background: linear-gradient(135deg, #E67E22, #F39C12) !important;
        box-shadow: 0 4px 12px rgba(230, 126, 34, 0.6) !important;
    }
    
    .stSlider[data-testid*="stSlider"]:nth-of-type(5) > div > div > div > div > div > div > div {
        background: linear-gradient(135deg, #E74C3C, #EC7063) !important;
        box-shadow: 0 4px 12px rgba(231, 76, 60, 0.6) !important;
    }
    
    /* Simplified Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #27AE60, #2ECC71) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 1rem 2rem !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 20px rgba(46, 204, 113, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 12px 30px rgba(46, 204, 113, 0.6) !important;
        background: linear-gradient(135deg, #2ECC71, #27AE60) !important;
    }
    
    /* Labels */
    .stSelectbox label, .stNumberInput label {
        color: #2C3E50 !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1) !important;
        margin-bottom: 1rem !important;
        display: block !important;
    }
    
    /* Super Enhanced Slider Labels for Maximum Visibility */
    .stSlider label {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 1.4rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.9) !important;
        margin-bottom: 1.5rem !important;
        display: block !important;
        line-height: 1.4 !important;
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        padding: 1.5rem 2rem !important;
        border-radius: 20px !important;
        border: 3px solid rgba(255, 255, 255, 0.4) !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5) !important;
        backdrop-filter: blur(15px) !important;
        text-align: center !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stSlider label::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stSlider label:hover::before {
        left: 100%;
    }
    
    .stSlider label:hover {
        background: linear-gradient(135deg, #764ba2, #667eea) !important;
        border-color: rgba(255, 255, 255, 0.7) !important;
        transform: translateY(-8px) scale(1.05) !important;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.7) !important;
        text-shadow: 3px 3px 6px rgba(0,0,0,1) !important;
    }
    
    /* Individual slider label colors for better distinction */
    .stSlider:nth-of-type(1) label {
        background: linear-gradient(135deg, #27AE60, #2ECC71) !important;
        box-shadow: 0 10px 30px rgba(39, 174, 96, 0.5) !important;
    }
    
    .stSlider:nth-of-type(1) label:hover {
        background: linear-gradient(135deg, #2ECC71, #58D68D) !important;
        box-shadow: 0 20px 40px rgba(39, 174, 96, 0.7) !important;
    }
    
    .stSlider:nth-of-type(2) label {
        background: linear-gradient(135deg, #3498DB, #5DADE2) !important;
        box-shadow: 0 10px 30px rgba(52, 152, 219, 0.5) !important;
    }
    
    .stSlider:nth-of-type(2) label:hover {
        background: linear-gradient(135deg, #5DADE2, #85C1E9) !important;
        box-shadow: 0 20px 40px rgba(52, 152, 219, 0.7) !important;
    }
    
    .stSlider:nth-of-type(3) label {
        background: linear-gradient(135deg, #9B59B6, #BB8FCE) !important;
        box-shadow: 0 10px 30px rgba(155, 89, 182, 0.5) !important;
    }
    
    .stSlider:nth-of-type(3) label:hover {
        background: linear-gradient(135deg, #BB8FCE, #D7BDE2) !important;
        box-shadow: 0 20px 40px rgba(155, 89, 182, 0.7) !important;
    }
    
    .stSlider:nth-of-type(4) label {
        background: linear-gradient(135deg, #E67E22, #F39C12) !important;
        box-shadow: 0 10px 30px rgba(230, 126, 34, 0.5) !important;
    }
    
    .stSlider:nth-of-type(4) label:hover {
        background: linear-gradient(135deg, #F39C12, #F7DC6F) !important;
        box-shadow: 0 20px 40px rgba(230, 126, 34, 0.7) !important;
    }
    
    .stSlider:nth-of-type(5) label {
        background: linear-gradient(135deg, #E74C3C, #EC7063) !important;
        box-shadow: 0 10px 30px rgba(231, 76, 60, 0.5) !important;
    }
    
    .stSlider:nth-of-type(5) label:hover {
        background: linear-gradient(135deg, #EC7063, #F1948A) !important;
        box-shadow: 0 20px 40px rgba(231, 76, 60, 0.7) !important;
    }
    
    /* Simplified Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.1), rgba(39, 174, 96, 0.1));
        border: 2px solid rgba(46, 204, 113, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        border-color: #27AE60;
        box-shadow: 0 8px 25px rgba(46, 204, 113, 0.3);
    }
    
    /* Simplified Results styling */
    .prediction-box {
        background: linear-gradient(135deg, #27AE60, #2ECC71);
        color: white;
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(46, 204, 113, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #27AE60, #2ECC71, #58D68D, #82E0AA);
        border-radius: 20px;
        z-index: -1;
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 0.8; }
        50% { opacity: 1; }
    }
    
    /* Eco-friendly theme colors */
    .eco-green { 
        color: #27AE60 !important; 
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1) !important;
    }
    .eco-blue { 
        color: #3498DB !important; 
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1) !important;
    }
    .eco-orange { 
        color: #E67E22 !important; 
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1) !important;
    }
    .eco-purple { 
        color: #9B59B6 !important; 
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1) !important;
    }
    
    /* Enhanced section header styling */
    .modern-card h3 {
        color: #2C3E50 !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1) !important;
        margin-top: 0 !important;
        margin-bottom: 1rem !important;
    }
    
    .modern-card h3.eco-green {
        background: linear-gradient(135deg, #27AE60, #2ECC71) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-weight: 800 !important;
    }
    
    .modern-card h3.eco-blue {
        background: linear-gradient(135deg, #3498DB, #5DADE2) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-weight: 800 !important;
    }
    
    .modern-card h3.eco-orange {
        background: linear-gradient(135deg, #E67E22, #F39C12) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="main-header">
    <h1 style="font-size: 3.5rem; margin: 0; color: white; font-weight: 800; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
        🌿 EcoPredict AI
    </h1>
    <h3 style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-weight: 400;">
        Advanced Supply Chain Emissions Intelligence
    </h3>
</div>
""", unsafe_allow_html=True)

# Create columns for layout
col1, col2 = st.columns([1.2, 1])

# Input form
with col1:
    st.markdown("""
    <div class="modern-card">
        <h2 style="color: #2C3E50; text-align: center; margin-bottom: 2rem; font-weight: 700;">
            🔧 Smart Configuration Panel
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        # Basic Parameters Section
        st.markdown("""
        <div class="modern-card" style="border-left: 5px solid #27AE60; background: linear-gradient(135deg, rgba(46, 204, 113, 0.05), rgba(255, 255, 255, 0.95)); box-shadow: 0 8px 32px rgba(46, 204, 113, 0.2);">
            <h3 class="eco-green" style="margin-top: 0; font-size: 1.8rem; font-weight: 800; color: #27AE60 !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);">🧪 Basic Parameters</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col_basic1, col_basic2 = st.columns(2)
        with col_basic1:
            substance = st.selectbox(
                "🧪 Greenhouse Gas Type", 
                ['carbon dioxide', 'methane', 'nitrous oxide', 'other GHGs'],
                help="Select the primary greenhouse gas for analysis"
            )
            source = st.selectbox(
                "🏭 Data Source", 
                ['Commodity', 'Industry'],
                help="Choose your data source type"
            )
        
        with col_basic2:
            unit = st.selectbox(
                "📏 Measurement Unit", 
                ['kg/2018 USD, purchaser price', 'kg CO2e/2018 USD, purchaser price'],
                help="CO2e includes all GHGs in CO2 equivalent"
            )
            
            # Display current config
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0; color: #27AE60;">Current Setup</h4>
                <p style="margin: 0.5rem 0; color: #2C3E50; font-weight: 600;">
                    {substance.title()} • {source}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Emission Factors Section
        st.markdown("""
        <div class="modern-card" style="border-left: 5px solid #3498DB; background: linear-gradient(135deg, rgba(52, 152, 219, 0.05), rgba(255, 255, 255, 0.95)); box-shadow: 0 8px 32px rgba(52, 152, 219, 0.2);">
            <h3 class="eco-blue" style="margin-top: 0; font-size: 1.8rem; font-weight: 800; color: #3498DB !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);">💰 Emission Factors</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col_factors1, col_factors2 = st.columns(2)
        with col_factors1:
            supply_wo_margin = st.number_input(
                "🏭 Base Emission Factor", 
                min_value=0.0, 
                value=0.05, 
                step=0.001,
                format="%.6f",
                help="Base emission factor without margins"
            )
        
        with col_factors2:
            margin = st.number_input(
                "⚖️ Safety Margin", 
                min_value=0.0, 
                value=0.005, 
                step=0.001,
                format="%.6f",
                help="Additional uncertainty margin"
            )
        
        # Display total factor
        total_factor = supply_wo_margin + margin
        st.markdown(f"""
        <div class="prediction-box" style="margin: 1rem 0; padding: 1.5rem;">
            <h4 style="margin: 0; color: white;">📊 Total Expected Factor</h4>
            <h2 style="margin: 0.5rem 0; color: white; font-size: 2.5rem;">{total_factor:.6f}</h2>
            <p style="margin: 0; color: rgba(255,255,255,0.9);">kg CO2e/USD</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Data Quality Section
        st.markdown("""
        <div class="modern-card" style="border-left: 5px solid #E67E22; background: linear-gradient(135deg, rgba(230, 126, 34, 0.05), rgba(255, 255, 255, 0.95)); box-shadow: 0 8px 32px rgba(230, 126, 34, 0.2);">
            <h3 class="eco-orange" style="margin-top: 0; font-size: 1.8rem; font-weight: 800; color: #E67E22 !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);">📊 Data Quality Metrics</h3>
            <p style="color: #7F8C8D; margin-bottom: 1.5rem; font-weight: 500; font-size: 1.1rem;">Rate each aspect from 0.0 (Poor) to 1.0 (Excellent)</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_dq1, col_dq2 = st.columns(2)
        
        with col_dq1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #27AE60, #2ECC71); 
                        padding: 2rem; border-radius: 20px; margin-bottom: 2rem; text-align: center;
                        box-shadow: 0 10px 30px rgba(39, 174, 96, 0.3);">
                <h3 style="color: white; margin: 0; font-weight: 800; font-size: 1.4rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    🎯 Primary Quality Metrics
                </h3>
                <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1rem;">
                    Core data quality indicators
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            dq_reliability = st.slider("🎯 Reliability", 0.0, 1.0, 0.8, 0.1, 
                                     help="Rate the reliability of your data source (0.0 = Poor, 1.0 = Excellent)")
            
            dq_temporal = st.slider("⏰ Temporal Correlation", 0.0, 1.0, 0.8, 0.1,
                                  help="How well does your data match the time period? (0.0 = Poor match, 1.0 = Perfect match)")
            
            dq_geo = st.slider("🌍 Geographic Correlation", 0.0, 1.0, 0.8, 0.1,
                             help="Geographic relevance of your data (0.0 = Different region, 1.0 = Same location)")
            
            # Display values for Set 1 with enhanced styling
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(39, 174, 96, 0.15), rgba(46, 204, 113, 0.1)); 
                        padding: 1.5rem; border-radius: 15px; margin-top: 1.5rem; 
                        border: 2px solid rgba(39, 174, 96, 0.3); backdrop-filter: blur(10px);">
                <h4 style="color: #27AE60; margin: 0 0 1rem 0; font-weight: 700; text-align: center;">
                    📊 Current Settings
                </h4>
                <div style="display: grid; gap: 0.8rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; 
                                background: rgba(255,255,255,0.7); padding: 0.8rem; border-radius: 10px;">
                        <span style="color: #2C3E50; font-weight: 600;">🎯 Reliability:</span>
                        <span style="color: #27AE60; font-weight: 800; font-size: 1.2rem;">{dq_reliability:.1f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; 
                                background: rgba(255,255,255,0.7); padding: 0.8rem; border-radius: 10px;">
                        <span style="color: #2C3E50; font-weight: 600;">⏰ Temporal:</span>
                        <span style="color: #3498DB; font-weight: 800; font-size: 1.2rem;">{dq_temporal:.1f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; 
                                background: rgba(255,255,255,0.7); padding: 0.8rem; border-radius: 10px;">
                        <span style="color: #2C3E50; font-weight: 600;">🌍 Geographic:</span>
                        <span style="color: #9B59B6; font-weight: 800; font-size: 1.2rem;">{dq_geo:.1f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_dq2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #E67E22, #F39C12); 
                        padding: 2rem; border-radius: 20px; margin-bottom: 2rem; text-align: center;
                        box-shadow: 0 10px 30px rgba(230, 126, 34, 0.3);">
                <h3 style="color: white; margin: 0; font-weight: 800; font-size: 1.4rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    � Advanced Quality Metrics
                </h3>
                <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1rem;">
                    Technical & collection quality
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            dq_tech = st.slider("🔬 Technology Correlation", 0.0, 1.0, 0.8, 0.1,
                              help="Technology relevance of your data (0.0 = Different tech, 1.0 = Same technology)")
            
            dq_data = st.slider("📊 Data Collection", 0.0, 1.0, 0.8, 0.1,
                              help="Quality of data collection methods (0.0 = Poor methods, 1.0 = Excellent methods)")
            
            # Display values for Set 2 with enhanced styling
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(230, 126, 34, 0.15), rgba(243, 156, 18, 0.1)); 
                        padding: 1.5rem; border-radius: 15px; margin-top: 1.5rem; 
                        border: 2px solid rgba(230, 126, 34, 0.3); backdrop-filter: blur(10px);">
                <h4 style="color: #E67E22; margin: 0 0 1rem 0; font-weight: 700; text-align: center;">
                    📊 Current Settings
                </h4>
                <div style="display: grid; gap: 0.8rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; 
                                background: rgba(255,255,255,0.7); padding: 0.8rem; border-radius: 10px;">
                        <span style="color: #2C3E50; font-weight: 600;">🔬 Technology:</span>
                        <span style="color: #E67E22; font-weight: 800; font-size: 1.2rem;">{dq_tech:.1f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; 
                                background: rgba(255,255,255,0.7); padding: 0.8rem; border-radius: 10px;">
                        <span style="color: #2C3E50; font-weight: 600;">📊 Collection:</span>
                        <span style="color: #E74C3C; font-weight: 800; font-size: 1.2rem;">{dq_data:.1f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Overall Quality Score with Enhanced Visual Display
        overall_quality = np.mean([dq_reliability, dq_temporal, dq_geo, dq_tech, dq_data])
        quality_status = "Excellent" if overall_quality >= 0.8 else "Good" if overall_quality >= 0.6 else "Fair" if overall_quality >= 0.4 else "Poor"
        
        # Quality color based on score
        if overall_quality >= 0.8:
            quality_color = "#27AE60"
            quality_bg = "linear-gradient(135deg, #27AE60, #2ECC71)"
        elif overall_quality >= 0.6:
            quality_color = "#3498DB"
            quality_bg = "linear-gradient(135deg, #3498DB, #5DADE2)"
        elif overall_quality >= 0.4:
            quality_color = "#F39C12"
            quality_bg = "linear-gradient(135deg, #F39C12, #F7DC6F)"
        else:
            quality_color = "#E74C3C"
            quality_bg = "linear-gradient(135deg, #E74C3C, #EC7063)"
        
        st.markdown(f"""
        <div class="modern-card" style="background: {quality_bg}; color: white; border: none; text-align: center; 
                    box-shadow: 0 15px 35px rgba(0,0,0,0.2); transform: scale(1.02); transition: all 0.3s ease;">
            <div style="position: relative; padding: 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🏆</div>
                <h2 style="margin: 0; color: white; font-size: 1.8rem; font-weight: 800; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    Overall Data Quality Score
                </h2>
                <div style="margin: 2rem 0;">
                    <div style="font-size: 5rem; font-weight: 900; color: white; text-shadow: 3px 3px 6px rgba(0,0,0,0.4); 
                                line-height: 1; margin-bottom: 0.5rem;">
                        {overall_quality:.2f}
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: rgba(255,255,255,0.95); 
                                text-transform: uppercase; letter-spacing: 2px;">
                        {quality_status} Rating
                    </div>
                </div>
                
                <div style="background: rgba(255,255,255,0.15); padding: 1.5rem; border-radius: 15px; 
                            margin-top: 2rem; backdrop-filter: blur(10px);">
                    <h4 style="color: white; margin: 0 0 1rem 0; font-size: 1.2rem; font-weight: 600;">
                        Quality Breakdown
                    </h4>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.8rem 0;">
                        <span style="color: rgba(255,255,255,0.9); font-weight: 500;">🎯 Reliability:</span>
                        <span style="color: white; font-weight: 700; font-size: 1.1rem;">{dq_reliability:.1f}/1.0</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.8rem 0;">
                        <span style="color: rgba(255,255,255,0.9); font-weight: 500;">⏰ Temporal:</span>
                        <span style="color: white; font-weight: 700; font-size: 1.1rem;">{dq_temporal:.1f}/1.0</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.8rem 0;">
                        <span style="color: rgba(255,255,255,0.9); font-weight: 500;">🌍 Geographic:</span>
                        <span style="color: white; font-weight: 700; font-size: 1.1rem;">{dq_geo:.1f}/1.0</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.8rem 0;">
                        <span style="color: rgba(255,255,255,0.9); font-weight: 500;">🔬 Technology:</span>
                        <span style="color: white; font-weight: 700; font-size: 1.1rem;">{dq_tech:.1f}/1.0</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.8rem 0;">
                        <span style="color: rgba(255,255,255,0.9); font-weight: 500;">📊 Collection:</span>
                        <span style="color: white; font-weight: 700; font-size: 1.1rem;">{dq_data:.1f}/1.0</span>
                    </div>
                </div>
                
                <div style="margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.1); 
                            border-radius: 10px; font-size: 1rem; color: rgba(255,255,255,0.95);">
                    💡 <strong>Tip:</strong> Use the sliders above to adjust ratings in real-time!
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Submit Button
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("🚀 Generate AI Prediction", use_container_width=True)

# Results Column
with col2:
    st.markdown("""
    <div class="modern-card">
        <h2 style="color: #2C3E50; text-align: center; margin-bottom: 2rem; font-weight: 700;">
            📈 AI Analytics Dashboard
        </h2>
    </div>
    """, unsafe_allow_html=True)

    if submit:
        # Prepare input data
        input_data = {
            'Substance': substance,
            'Unit': unit,
            'Supply Chain Emission Factors without Margins': supply_wo_margin,
            'Margins of Supply Chain Emission Factors': margin,
            'DQ ReliabilityScore of Factors without Margins': dq_reliability,
            'DQ TemporalCorrelation of Factors without Margins': dq_temporal,
            'DQ GeographicalCorrelation of Factors without Margins': dq_geo,
            'DQ TechnologicalCorrelation of Factors without Margins': dq_tech,
            'DQ DataCollection of Factors without Margins': dq_data,
            'Source': source,
        }

        try:
            # Validate input data
            is_valid, error_msg = validate_input_data(input_data)
            if not is_valid:
                st.error(f"❌ Input Validation Error: {error_msg}")
                st.stop()
            
            # Make prediction
            with st.spinner('🤖 AI is analyzing your data...'):
                input_df = preprocess_input(pd.DataFrame([input_data]))
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)
                prediction_value = prediction[0]

            # Display prediction result
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="color: white; margin-bottom: 1rem;">🎯 AI Prediction Result</h2>
                <h1 style="color: white; font-size: 4rem; margin: 1rem 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    {prediction_value:.6f}
                </h1>
                <p style="font-size: 1.5rem; color: rgba(255,255,255,0.9); margin: 0;">
                    kg CO2e per USD
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Impact Level Assessment
            if prediction_value > 0.1:
                impact_level, impact_color, impact_emoji = "High", "#E74C3C", "🔴"
            elif prediction_value > 0.05:
                impact_level, impact_color, impact_emoji = "Medium", "#F39C12", "🟡"  
            else:
                impact_level, impact_color, impact_emoji = "Low", "#27AE60", "🟢"
            
            st.markdown(f"""
            <div class="modern-card" style="border-left: 5px solid {impact_color}; text-align: center;">
                <h3 style="color: {impact_color}; margin: 0;">
                    {impact_emoji} Environmental Impact: {impact_level}
                </h3>
                <p style="color: #2C3E50; margin: 1rem 0 0 0;">
                    {"⚠️ Consider emission reduction strategies" if impact_level == "High" else 
                     "📊 Monitor and optimize where possible" if impact_level == "Medium" else
                     "✅ Great job maintaining sustainable practices!"}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create and display charts
            st.markdown("### 📊 Visual Analytics")
            
            if PLOTLY_AVAILABLE:
                # Emission comparison chart
                try:
                    fig_bar = create_emission_bar_chart(prediction_value)
                    if fig_bar:
                        st.plotly_chart(fig_bar, use_container_width=True)
                except Exception as e:
                    st.warning(f"Chart error: {e}")
                    create_fallback_chart_display(prediction_value, [dq_reliability, dq_temporal, dq_geo, dq_tech, dq_data], overall_quality)
                
                # Data Quality Radar Chart
                st.markdown("### 🎨 Data Quality Radar")
                try:
                    dq_scores = [dq_reliability, dq_temporal, dq_geo, dq_tech, dq_data]
                    fig_radar = create_radar_chart(dq_scores)
                    if fig_radar:
                        st.plotly_chart(fig_radar, use_container_width=True)
                except Exception as e:
                    st.warning(f"Radar chart error: {e}")
                    # Fallback: Simple metric display
                    cols = st.columns(5)
                    metrics = ["🎯 Reliability", "⏰ Temporal", "🌍 Geographic", "🔬 Technology", "📊 Collection"]
                    values = [dq_reliability, dq_temporal, dq_geo, dq_tech, dq_data]
                    
                    for i, (col, metric, value) in enumerate(zip(cols, metrics, values)):
                        with col:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="margin: 0; color: #27AE60; font-size: 0.9rem;">{metric}</h4>
                                <h2 style="margin: 0.5rem 0; color: #2C3E50;">{value:.1f}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Quality Gauge
                st.markdown("### 🎯 Quality Gauge")
                try:
                    fig_gauge = create_gauge_chart(overall_quality, "Overall Data Quality", "#27AE60")
                    if fig_gauge:
                        st.plotly_chart(fig_gauge, use_container_width=True)
                except Exception as e:
                    st.warning(f"Gauge error: {e}")
                    st.markdown(f"""
                    <div class="modern-card" style="text-align: center;">
                        <h2 style="margin: 0; color: #27AE60;">Overall Quality: {overall_quality:.2f}</h2>
                        <p style="margin: 0.5rem 0; color: #2C3E50;">Status: {quality_status}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Use fallback displays when Plotly is not available
                create_fallback_chart_display(prediction_value, [dq_reliability, dq_temporal, dq_geo, dq_tech, dq_data], overall_quality)
                
                st.markdown("### 🎯 Quality Overview")
                st.markdown(f"""
                <div class="modern-card" style="text-align: center;">
                    <h2 style="margin: 0; color: #27AE60;">Overall Quality: {overall_quality:.2f}</h2>
                    <p style="margin: 0.5rem 0; color: #2C3E50;">Status: {quality_status}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Smart Recommendations
            st.markdown("### 💡 AI Recommendations")
            recommendations = []
            
            if impact_level == "High":
                recommendations.extend([
                    "🔄 Consider switching to cleaner alternatives",
                    "🌱 Explore renewable energy sources",
                    "📉 Implement carbon reduction technologies"
                ])
            elif impact_level == "Medium":
                recommendations.extend([
                    "⚡ Optimize supply chain processes",
                    "🎯 Set emission reduction targets",
                    "📊 Monitor performance regularly"
                ])
            else:
                recommendations.extend([
                    "✨ Maintain current sustainable practices",
                    "🤝 Share best practices with others",
                    "🔄 Continue monitoring and improvement"
                ])
            
            if overall_quality < 0.6:
                recommendations.append("📈 Improve data collection methods for better accuracy")
            
            for i, rec in enumerate(recommendations):
                st.markdown(f"""
                <div class="modern-card" style="border-left: 4px solid #9B59B6; margin: 0.5rem 0;">
                    <p style="margin: 0; color: #2C3E50; font-weight: 500;">
                        <strong>💡 Recommendation #{i+1}:</strong> {rec}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            st.info("🔧 Please check your input values and try again.")
    
    else:
        # Welcome message when no prediction yet
        st.markdown("""
        <div class="modern-card" style="text-align: center;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🚀</div>
            <h2 style="color: #2C3E50; margin-bottom: 1rem;">Ready for AI Analysis!</h2>
            <p style="font-size: 1.2rem; color: #7F8C8D; margin-bottom: 2rem;">
                Configure your parameters and click 'Generate AI Prediction' to see intelligent insights
            </p>
            
            <div style="background: linear-gradient(135deg, #3498DB, #2980B9); padding: 2rem; border-radius: 15px; color: white; margin: 2rem 0;">
                <h3 style="color: white; margin-bottom: 1rem;">🎯 Quick Start Guide</h3>
                <div style="text-align: left; max-width: 300px; margin: 0 auto;">
                    <p style="margin: 0.8rem 0; color: white;"><strong>1.</strong> 🧪 Select greenhouse gas type</p>
                    <p style="margin: 0.8rem 0; color: white;"><strong>2.</strong> 💰 Input emission factors</p>
                    <p style="margin: 0.8rem 0; color: white;"><strong>3.</strong> 📊 Rate data quality</p>
                    <p style="margin: 0.8rem 0; color: white;"><strong>4.</strong> 🚀 Generate prediction</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample predictions showcase
        st.markdown("""
        <div class="modern-card">
            <h3 style="color: #2C3E50; text-align: center; margin-bottom: 1.5rem;">📊 Sample Scenarios</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div style="background: linear-gradient(135deg, #27AE60, rgba(46, 204, 113, 0.8)); padding: 1.5rem; border-radius: 12px; color: white; text-align: center;">
                    <h4 style="margin: 0; color: white;">🌱 Low Impact</h4>
                    <p style="margin: 0.5rem 0; color: white; font-size: 1.2rem; font-weight: 600;">~0.030</p>
                    <small style="color: rgba(255,255,255,0.8);">Sustainable practices</small>
                </div>
                <div style="background: linear-gradient(135deg, #E67E22, rgba(243, 156, 18, 0.8)); padding: 1.5rem; border-radius: 12px; color: white; text-align: center;">
                    <h4 style="margin: 0; color: white;">⚡ Medium Impact</h4>
                    <p style="margin: 0.5rem 0; color: white; font-size: 1.2rem; font-weight: 600;">~0.075</p>
                    <small style="color: rgba(255,255,255,0.8);">Needs optimization</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Sidebar with enhanced information
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
    <h2 style="color: white; text-align: center; margin-bottom: 1rem;">🤖 AI Model Info</h2>
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
        <p style="margin: 0.5rem 0; color: white;"><strong>🔬 Algorithm:</strong> Linear Regression</p>
        <p style="margin: 0.5rem 0; color: white;"><strong>📊 Training:</strong> US Industries 2010-2016</p>
        <p style="margin: 0.5rem 0; color: white;"><strong>🎯 Features:</strong> 10 parameters</p>
        <p style="margin: 0.5rem 0; color: white;"><strong>⚡ Status:</strong> Optimized</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #27AE60, #2ECC71); padding: 2rem; border-radius: 15px; color: white;">
    <h3 style="color: white; text-align: center; margin-bottom: 1rem;">🌍 Impact Scale</h3>
    <div style="color: white;">
        <div style="margin: 1rem 0; padding: 0.8rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
            <strong>🟢 Low (< 0.05)</strong><br>
            <small>Sustainable operations</small>
        </div>
        <div style="margin: 1rem 0; padding: 0.8rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
            <strong>🟡 Medium (0.05-0.1)</strong><br>
            <small>Moderate environmental impact</small>
        </div>
        <div style="margin: 1rem 0; padding: 0.8rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
            <strong>🔴 High (> 0.1)</strong><br>
            <small>Requires immediate attention</small>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
           padding: 3rem; border-radius: 20px; text-align: center; color: white; margin: 2rem 0;">
    <h2 style="color: white; margin-bottom: 1rem;">🌿 EcoPredict AI</h2>
    <p style="font-size: 1.2rem; margin-bottom: 1.5rem; color: rgba(255,255,255,0.9);">
        Empowering sustainable decisions through intelligent analytics
    </p>
    <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin: 1.5rem 0;">
        <div style="background: rgba(255,255,255,0.1); padding: 1rem 1.5rem; border-radius: 20px;">
            <strong>🤖 AI-Powered</strong>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 1rem 1.5rem; border-radius: 20px;">
            <strong>🌍 Eco-Friendly</strong>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 1rem 1.5rem; border-radius: 20px;">
            <strong>📊 Data-Driven</strong>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Copyright Section
st.markdown("""
<div style="background: linear-gradient(135deg, #2C3E50, #34495E); 
           padding: 2rem; border-radius: 15px; text-align: center; color: white; margin: 1rem 0;
           border-top: 3px solid #3498DB;">
    <div style="margin-bottom: 1rem;">
        <p style="font-size: 1rem; margin: 0.5rem 0; color: rgba(255,255,255,0.9);">
            <strong>© 2025 EcoPredict AI - Advanced GHG Emissions Intelligence Platform</strong>
        </p>
        <p style="font-size: 0.9rem; margin: 0.5rem 0; color: rgba(255,255,255,0.7);">
            Developed with ❤️ for Environmental Sustainability
        </p>
    </div>
    
    <div style="border-top: 1px solid rgba(255,255,255,0.2); padding-top: 1rem; margin-top: 1rem;">
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; font-size: 0.85rem;">
            <span style="color: rgba(255,255,255,0.8);">
                <strong>🔬 Algorithm:</strong> Linear Regression ML Model
            </span>
            <span style="color: rgba(255,255,255,0.8);">
                <strong>📊 Dataset:</strong> US Supply Chain Emissions 2010-2016
            </span>
            <span style="color: rgba(255,255,255,0.8);">
                <strong>🌱 Purpose:</strong> Carbon Footprint Analysis
            </span>
        </div>
    </div>
    
    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);">
        <p style="font-size: 0.8rem; color: rgba(255,255,255,0.6); margin: 0;">
            This software is provided for educational and research purposes. All predictions are estimates based on historical data.<br>
            For commercial use, please ensure compliance with applicable environmental regulations and standards.
        </p>
    </div>
    
    <div style="margin-top: 1rem;">
        <p style="font-size: 0.75rem; color: rgba(255,255,255,0.5); margin: 0;">
            Built with Streamlit 🚀 | Powered by scikit-learn 🤖 | Enhanced with Plotly 📊
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
