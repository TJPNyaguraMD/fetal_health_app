"""
Fetal Health Classification - Simplified Interface
Focus on most important CTG features for quick online predictions
"""

import streamlit as st
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Fetal Health Predictor",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# API Configuration
API_URL = "https://fetalhealthapplication.streamlit.app/"

# Custom CSS for beautiful, simple interface
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header */
    .main-header {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }
    
    .main-header h1 {
        color: #667eea;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .main-header p {
        color: #666;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Input container */
    .input-container {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        margin-bottom: 1.5rem;
    }
    
    /* Section headers */
    .section-header {
        color: #667eea;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    /* Prediction result */
    .prediction-result {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        text-align: center;
        margin: 2rem 0;
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .prediction-badge {
        display: inline-block;
        padding: 1.5rem 3rem;
        border-radius: 50px;
        font-size: 2rem;
        font-weight: 700;
        margin: 1rem 0;
        color: white;
    }
    
    .badge-normal {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 5px 25px rgba(17, 153, 142, 0.5);
    }
    
    .badge-suspect {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 5px 25px rgba(245, 87, 108, 0.5);
    }
    
    .badge-pathological {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        box-shadow: 0 5px 25px rgba(250, 112, 154, 0.5);
    }
    
    /* Confidence meter */
    .confidence-meter {
        margin: 2rem 0;
    }
    
    .confidence-bar {
        background: #f0f0f0;
        height: 40px;
        border-radius: 20px;
        overflow: hidden;
        position: relative;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        transition: width 1s ease;
    }
    
    /* Info boxes */
    .info-box {
        background: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #0d47a1;
    }
    
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #856404;
    }
    
    .danger-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #721c24;
    }
    
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #155724;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 3rem;
        font-size: 1.2rem;
        font-weight: 600;
        width: 100%;
        margin-top: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 35px rgba(102, 126, 234, 0.6);
    }
    
    /* Input fields */
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Selectbox */
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Feature importance */
    .feature-importance {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .importance-high {
        color: #dc3545;
        font-weight: 600;
    }
    
    .importance-medium {
        color: #ffc107;
        font-weight: 600;
    }
    
    .importance-low {
        color: #28a745;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None


def get_prediction(features):
    """Get prediction from API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=features,
            timeout=10
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.text}"
    except Exception as e:
        return None, f"Connection Error: {str(e)}"


def display_prediction_result(result):
    """Display prediction result beautifully."""
    prediction = result['prediction']
    prediction_label = result['prediction_label']
    confidence = result['confidence']
    risk_level = result['risk_level']
    probabilities = result['probabilities']
    
    # Determine badge class
    if prediction == 1:
        badge_class = "badge-normal"
        box_class = "success-box"
        icon = "✅"
    elif prediction == 2:
        badge_class = "badge-suspect"
        box_class = "warning-box"
        icon = "⚠️"
    else:
        badge_class = "badge-pathological"
        box_class = "danger-box"
        icon = "🚨"
    
    st.markdown(f"""
        <div class="prediction-result">
            <h2 style="color: #333; margin-bottom: 1rem;">Prediction Result</h2>
            <div class="prediction-badge {badge_class}">
                {icon} {prediction_label}
            </div>
            <p style="color: #666; font-size: 1.2rem; margin-top: 0.5rem;">{risk_level}</p>
            
            <div class="confidence-meter">
                <p style="color: #666; margin-bottom: 0.5rem;">Confidence Level</p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence*100}%;">
                        {confidence*100:.1f}%
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Display interpretation
    if prediction == 1:
        st.markdown(f"""
            <div class="{box_class}">
                <h4 style="margin-top: 0;">✅ Normal Fetal Health</h4>
                <p><strong>Recommendation:</strong> Continue routine monitoring. No immediate intervention required.</p>
                <ul>
                    <li>Fetal heart rate patterns are within normal limits</li>
                    <li>No signs of distress detected</li>
                    <li>Maintain standard care protocol</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    elif prediction == 2:
        st.markdown(f"""
            <div class="{box_class}">
                <h4 style="margin-top: 0;">⚠️ Suspect Fetal Health</h4>
                <p><strong>Recommendation:</strong> Increased monitoring and clinical review recommended.</p>
                <ul>
                    <li>Some abnormal CTG patterns detected</li>
                    <li>Increase monitoring frequency</li>
                    <li>Consult with obstetrician</li>
                    <li>Consider additional diagnostic tests</li>
                    <li>Reassess in 30-60 minutes</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="{box_class}">
                <h4 style="margin-top: 0;">🚨 Pathological - Immediate Attention Required</h4>
                <p><strong>⚠️ URGENT ACTION NEEDED:</strong></p>
                <ul>
                    <li><strong>Immediate obstetric consultation required</strong></li>
                    <li>Initiate continuous fetal monitoring</li>
                    <li>Prepare for potential emergency intervention</li>
                    <li>Consider expedited delivery if distress persists</li>
                    <li>Document all findings and actions taken</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Show probabilities
    with st.expander("📊 View Detailed Probabilities", expanded=False):
        for class_name, prob in probabilities.items():
            st.progress(prob, text=f"{class_name}: {prob*100:.1f}%")


def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🏥 Fetal Health Predictor</h1>
            <p>Quick CTG Analysis - Enter Key Features</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Check API
    api_available, health_data = check_api_health()
    
    if not api_available:
        st.error("⚠️ **API Not Available** - Please start the API server first")
        st.code("python api_server.py", language="bash")
        st.stop()
    
    # Quick load examples
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    st.markdown("### 🚀 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📗 Load Normal Example", use_container_width=True):
            st.session_state.example = "normal"
    
    with col2:
        if st.button("📙 Load Suspect Example", use_container_width=True):
            st.session_state.example = "suspect"
    
    with col3:
        if st.button("📕 Load Pathological Example", use_container_width=True):
            st.session_state.example = "pathological"
    
    # Example data
    examples = {
        "normal": {
            "baseline_value": 120.0,
            "abnormal_short_term_variability": 20.0,
            "mean_value_of_short_term_variability": 2.0,
            "percentage_of_time_with_abnormal_long_term_variability": 5.0,
            "accelerations": 0.003,
            "light_decelerations": 0.002,
            "severe_decelerations": 0.0,
            "histogram_mean": 135.0
        },
        "suspect": {
            "baseline_value": 140.0,
            "abnormal_short_term_variability": 60.0,
            "mean_value_of_short_term_variability": 0.8,
            "percentage_of_time_with_abnormal_long_term_variability": 40.0,
            "accelerations": 0.0,
            "light_decelerations": 0.005,
            "severe_decelerations": 0.0,
            "histogram_mean": 145.0
        },
        "pathological": {
            "baseline_value": 160.0,
            "abnormal_short_term_variability": 85.0,
            "mean_value_of_short_term_variability": 0.3,
            "percentage_of_time_with_abnormal_long_term_variability": 75.0,
            "accelerations": 0.0,
            "light_decelerations": 0.008,
            "severe_decelerations": 0.002,
            "histogram_mean": 165.0
        }
    }
    
    # Get example if selected
    if 'example' in st.session_state:
        example_data = examples[st.session_state.example]
    else:
        example_data = examples["normal"]
    
    st.markdown("---")
    
    # Most Important Features Section
    st.markdown('<div class="section-header">🎯 Essential CTG Features</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <strong>ℹ️ These are the most important features for prediction</strong><br>
            Fill in the values from the CTG monitor reading
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        baseline_value = st.number_input(
            "⭐ Baseline Heart Rate (bpm)",
            min_value=100.0,
            max_value=180.0,
            value=example_data["baseline_value"],
            step=1.0,
            help="Normal range: 110-160 bpm. Most important predictor!"
        )
        
        abnormal_short_term_variability = st.number_input(
            "⭐ Abnormal Short-Term Variability (%)",
            min_value=0.0,
            max_value=100.0,
            value=example_data["abnormal_short_term_variability"],
            step=1.0,
            help="Percentage of time with abnormal beat-to-beat variability"
        )
        
        mean_value_of_short_term_variability = st.number_input(
            "Mean Short-Term Variability",
            min_value=0.0,
            max_value=10.0,
            value=example_data["mean_value_of_short_term_variability"],
            step=0.1,
            help="Average beat-to-beat variability"
        )
        
        accelerations = st.number_input(
            "⭐ Accelerations (/sec)",
            min_value=0.0,
            max_value=0.1,
            value=example_data["accelerations"],
            step=0.001,
            format="%.3f",
            help="Indicates fetal well-being. Higher is better!"
        )
    
    with col2:
        percentage_of_time_with_abnormal_long_term_variability = st.number_input(
            "Abnormal Long-Term Variability (%)",
            min_value=0.0,
            max_value=100.0,
            value=example_data["percentage_of_time_with_abnormal_long_term_variability"],
            step=1.0,
            help="Percentage of time with abnormal long-term patterns"
        )
        
        light_decelerations = st.number_input(
            "Light Decelerations (/sec)",
            min_value=0.0,
            max_value=0.1,
            value=example_data["light_decelerations"],
            step=0.001,
            format="%.3f",
            help="Temporary decreases in heart rate"
        )
        
        severe_decelerations = st.number_input(
            "⭐ Severe Decelerations (/sec)",
            min_value=0.0,
            max_value=0.1,
            value=example_data["severe_decelerations"],
            step=0.001,
            format="%.3f",
            help="WARNING: Indicates potential distress"
        )
        
        histogram_mean = st.number_input(
            "Histogram Mean (bpm)",
            min_value=50.0,
            max_value=200.0,
            value=example_data["histogram_mean"],
            step=1.0,
            help="Average heart rate from histogram"
        )
    
    # Additional features (collapsed by default)
    with st.expander("➕ Additional Features (Optional - Auto-filled)", expanded=False):
        col3, col4 = st.columns(2)
        
        with col3:
            fetal_movement = st.number_input(
                "Fetal Movement (/sec)",
                min_value=0.0,
                max_value=0.1,
                value=0.0,
                step=0.001,
                format="%.3f"
            )
            
            uterine_contractions = st.number_input(
                "Uterine Contractions (/sec)",
                min_value=0.0,
                max_value=0.1,
                value=0.005,
                step=0.001,
                format="%.3f"
            )
            
            prolongued_decelerations = st.number_input(
                "Prolonged Decelerations (/sec)",
                min_value=0.0,
                max_value=0.1,
                value=0.0,
                step=0.001,
                format="%.3f"
            )
        
        with col4:
            mean_value_of_long_term_variability = st.number_input(
                "Mean Long-Term Variability",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=0.5
            )
            
            histogram_number_of_peaks = st.number_input(
                "Histogram Peaks",
                min_value=0.0,
                max_value=20.0,
                value=3.0,
                step=1.0
            )
            
            histogram_number_of_zeroes = st.number_input(
                "Histogram Zeroes",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=1.0
            )
    
    # Predict button
    if st.button("🔍 Analyze Fetal Health", use_container_width=True, type="primary"):
        # Prepare features
        features = {
            "baseline_value": baseline_value,
            "accelerations": accelerations,
            "fetal_movement": fetal_movement,
            "uterine_contractions": uterine_contractions,
            "light_decelerations": light_decelerations,
            "severe_decelerations": severe_decelerations,
            "prolongued_decelerations": prolongued_decelerations,
            "abnormal_short_term_variability": abnormal_short_term_variability,
            "mean_value_of_short_term_variability": mean_value_of_short_term_variability,
            "percentage_of_time_with_abnormal_long_term_variability": percentage_of_time_with_abnormal_long_term_variability,
            "mean_value_of_long_term_variability": mean_value_of_long_term_variability,
            "histogram_number_of_peaks": histogram_number_of_peaks,
            "histogram_number_of_zeroes": histogram_number_of_zeroes,
            "histogram_mean": histogram_mean
        }
        
        # Show loading
        with st.spinner("🔄 Analyzing CTG data..."):
            result, error = get_prediction(features)
        
        if error:
            st.error(f"❌ {error}")
        else:
            display_prediction_result(result)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer disclaimer
    st.markdown("""
        <div class="info-box" style="margin-top: 2rem;">
            <h4 style="margin-top: 0;">⚠️ Medical Disclaimer</h4>
            <p>
                This tool provides <strong>decision support only</strong> and should not replace 
                clinical judgment. All predictions must be reviewed by qualified healthcare 
                professionals. This system is not a diagnostic device.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature importance guide
    with st.expander("📖 Feature Importance Guide"):
        st.markdown("""
            ### Most Important Features (marked with ⭐)
            
            **1. Baseline Heart Rate**
            - Normal: 110-160 bpm
            - High importance for classification
            
            **2. Abnormal Short-Term Variability**
            - Measures beat-to-beat heart rate changes
            - Higher percentage indicates more concern
            
            **3. Accelerations**
            - Temporary increases in heart rate
            - Presence indicates fetal well-being
            - Absence may be concerning
            
            **4. Severe Decelerations**
            - Most critical warning sign
            - Even small values can indicate distress
            
            ### Secondary Features
            
            - **Long-Term Variability**: Overall pattern changes
            - **Light Decelerations**: Usually benign
            - **Histogram Mean**: Overall heart rate distribution
            
            ### Reading Tips
            
            1. Always enter the most accurate values from CTG monitor
            2. Use example data to understand normal vs. abnormal patterns
            3. Pay special attention to starred (⭐) features
            4. When in doubt, consult with medical staff
        """)


if __name__ == "__main__":
    main()
