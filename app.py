import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# File paths
BASE_PATH = r"C:\Users\dinne\Diabetic-Readmissions-"
MODEL_PATH = BASE_PATH + r"\Model\diabetic_readmission_model.joblib"
FEATURE_INFO_PATH = BASE_PATH + r"\Model\feature_info.joblib"

# Page config
st.set_page_config(page_title="30-Day Diabetes Readmission Predictor", layout="wide")
st.title("🏥 30-Day Diabetes Readmission Risk Predictor")

class Predictor:
    def __init__(self):
        self.model = None
        self.features = None
        
    def load_model(self):
        try:
            self.model = joblib.load(MODEL_PATH)
            feature_info = joblib.load(FEATURE_INFO_PATH)
            self.features = feature_info['feature_names']
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def predict(self, input_data):
        input_df = pd.DataFrame([input_data])
        
        # Ensure all expected features are present
        for feature in self.features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        input_df = input_df[self.features]
        prediction = self.model.predict(input_df)[0]
        probability = self.model.predict_proba(input_df)[0]
        return prediction, probability
    
    def get_feature_importance(self, input_data):
        """Calculate feature importance using baseline comparison"""
        if hasattr(self.model, 'feature_importances_'):
            classifier = self.model
        elif hasattr(self.model, 'named_steps'):
            classifier = self.model.named_steps['classifier']
        else:
            return self._get_fallback_importance()
        
        if not hasattr(classifier, 'feature_importances_'):
            return self._get_fallback_importance()
            
        global_importance = classifier.feature_importances_
        feature_importance_map = dict(zip(self.features, global_importance))
        
        # Create input dataframe
        input_df = pd.DataFrame([input_data])
        for feature in self.features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        input_df = input_df[self.features]
        
        # Baseline values for common features
        baseline_values = {
            'age': 50, 'time_in_hospital': 4, 'num_medications': 8,
            'num_lab_procedures': 45, 'number_diagnoses': 6,
            'number_emergency': 1, 'number_inpatient': 1, 'num_procedures': 1
        }
        
        patient_importance = {}
        
        for feature in self.features:
            importance = feature_importance_map.get(feature, 0)
            feature_value = input_df[feature].iloc[0]
            
            if feature in baseline_values:
                baseline = baseline_values[feature]
                deviation = (feature_value - baseline) / (baseline + 1)
                contribution = importance * deviation
            else:
                contribution = importance * feature_value
            
            patient_importance[feature] = abs(contribution)
        
        # Get top 15 features
        top_features = sorted(patient_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        top_dict = dict(top_features)
        
        # Normalize to percentages
        total = sum(top_dict.values())
        if total > 0:
            return {k: 100 * v / total for k, v in top_dict.items()}
        else:
            return self._get_fallback_importance()
    
    def _get_fallback_importance(self):
        """Fallback importance values"""
        return {
            'time_in_hospital': 18, 'number_inpatient': 15, 'number_emergency': 12,
            'num_medications': 10, 'number_diagnoses': 9, 'age': 8,
            'num_lab_procedures': 7, 'insulin': 5, 'diabetesMed': 5,
            'A1Cresult_>8': 3, 'max_glu_serum_>300': 3, 'change': 2,
            'num_procedures': 2, 'race_Caucasian': 1, 'gender_Female': 1
        }

def create_input_data(user_inputs, model_features):
    """Create input data dictionary from user inputs"""
    input_data = {feature: 0 for feature in model_features}
    
    # Numerical features
    numerical_map = {
        'age': user_inputs['age'],
        'time_in_hospital': user_inputs['time_in_hospital'],
        'num_medications': user_inputs['num_medications'],
        'num_lab_procedures': user_inputs['num_lab_procedures'],
        'number_diagnoses': user_inputs['number_diagnoses'],
        'number_emergency': user_inputs['number_emergency'],
        'number_inpatient': user_inputs['number_inpatient'],
        'num_procedures': user_inputs['num_procedures']
    }
    
    for feature, value in numerical_map.items():
        if feature in model_features:
            input_data[feature] = value
    
    # Binary features
    binary_map = {
        'diabetesMed': 1 if user_inputs['diabetesMed'] == "Yes" else 0,
        'insulin': 1 if user_inputs['insulin'] == "Yes" else 0,
        'change': 1 if user_inputs['change'] == "Yes" else 0
    }
    
    for feature, value in binary_map.items():
        if feature in model_features:
            input_data[feature] = value
    
    # Categorical features
    categorical_map = {
        f'race_{user_inputs["race"]}': 1,
        f'gender_{user_inputs["gender"]}': 1,
        f'A1Cresult_{user_inputs["a1c_result"]}': 1,
        f'max_glu_serum_{user_inputs["max_glu_serum"]}': 1
    }
    
    for feature, value in categorical_map.items():
        if feature in model_features:
            input_data[feature] = value
    
    # Medications
    medications = ['metformin', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone']
    for med in medications:
        feature = f'{med}_{user_inputs[med]}'
        if feature in model_features:
            input_data[feature] = 1
    
    # Set common medications to "No"
    common_meds = ['acarbose', 'nateglinide', 'chlorpropamide', 'glimepiride', 
                   'acetohexamide', 'tolazamide', 'glipizide-metformin', 'glyburide-metformin']
    
    for med in common_meds:
        feature = f'{med}_No'
        if feature in model_features:
            input_data[feature] = 1
    
    return input_data

def prediction_page(predictor):
    st.header("Patient Risk Assessment")
    
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            age = st.slider("Age", 0, 100, 50)
            race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            
            st.subheader("Hospital Stay")
            time_in_hospital = st.slider("Current Hospital Stay (days)", 1, 30, 3)
            num_lab_procedures = st.slider("Lab Procedures This Stay", 0, 100, 50)
            num_procedures = st.slider("Number of Procedures", 0, 10, 1)
            
        with col2:
            st.subheader("Medical History")
            number_diagnoses = st.slider("Number of Diagnoses", 1, 20, 5)
            number_emergency = st.slider("ER Visits (past year)", 0, 20, 0)
            number_inpatient = st.slider("Inpatient Stays (past year)", 0, 20, 0)
            
            st.subheader("Diabetes Management")
            diabetesMed = st.selectbox("On Diabetes Medication", ["No", "Yes"])
            insulin = st.selectbox("Insulin Use", ["No", "Yes"])
            change = st.selectbox("Medication Change", ["No", "Yes"])
            
        with col3:
            st.subheader("Lab Results")
            a1c_result = st.selectbox("A1C Result", ["Norm", ">7", ">8", "None"])
            max_glu_serum = st.selectbox("Max Glucose Serum", ["Norm", ">200", ">300", "None"])
            
            st.subheader("Medications")
            num_medications = st.slider("Number of Medications", 0, 50, 10)
            metformin = st.selectbox("Metformin", ["No", "Steady", "Up", "Down"])
            glipizide = st.selectbox("Glipizide", ["No", "Steady", "Up", "Down"])
            glyburide = st.selectbox("Glyburide", ["No", "Steady", "Up", "Down"])
            pioglitazone = st.selectbox("Pioglitazone", ["No", "Steady", "Up", "Down"])
            rosiglitazone = st.selectbox("Rosiglitazone", ["No", "Steady", "Up", "Down"])
        
        submitted = st.form_submit_button("Predict 30-Day Readmission Risk")
    
    if submitted:
        user_inputs = {
            'age': age, 'time_in_hospital': time_in_hospital, 'num_medications': num_medications,
            'num_lab_procedures': num_lab_procedures, 'number_diagnoses': number_diagnoses,
            'number_emergency': number_emergency, 'number_inpatient': number_inpatient,
            'diabetesMed': diabetesMed, 'insulin': insulin, 'change': change,
            'race': race, 'gender': gender, 'a1c_result': a1c_result, 'max_glu_serum': max_glu_serum,
            'metformin': metformin, 'glipizide': glipizide, 'glyburide': glyburide,
            'pioglitazone': pioglitazone, 'rosiglitazone': rosiglitazone, 'num_procedures': num_procedures
        }
        
        input_data = create_input_data(user_inputs, predictor.features)
        prediction, probability = predictor.predict(input_data)
        feature_importance = predictor.get_feature_importance(input_data)
        show_results(prediction, probability, feature_importance)

def show_results(prediction, probability, feature_importance):
    risk_prob = probability[1]
    
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "30-Day Readmission Risk"},
            number={'suffix': '%'},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if risk_prob > 0.5 else "darkgreen"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgreen"},
                    {'range': [20, 50], 'color': "yellow"},
                    {'range': [50, 100], 'color': "lightcoral"}
                ],
                'threshold': {'value': 50}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk level
        if risk_prob > 0.5:
            st.error(f"🚨 HIGH RISK: {risk_prob:.1%} probability")
        elif risk_prob > 0.2:
            st.warning(f"⚠️ MEDIUM RISK: {risk_prob:.1%} probability")
        else:
            st.success(f"✅ LOW RISK: {risk_prob:.1%} probability")
    
    with col2:
        st.subheader("Prevention Plan")
        if risk_prob > 0.5:
            st.write("""
            **High-Risk Interventions:**
            - Follow-up within 3-5 days
            - Medication reconciliation
            - Care coordination
            - Home health evaluation
            """)
        elif risk_prob > 0.2:
            st.write("""
            **Medium-Risk Actions:**
            - Follow-up within 7 days
            - Medication review
            - Weekly check-ins
            - Diabetes education
            """)
        else:
            st.write("""
            **Standard Care:**
            - Routine follow-up
            - Continue medications
            - Self-monitoring guidance
            """)
    
    # Feature Importance
    if feature_importance:
        st.subheader("Feature Contribution Analysis")
        
        features, percentages = zip(*sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        readable_features = [f.replace('_', ' ').title().replace('Num ', 'Number of ') for f in features]
        
        fig = px.bar(
            x=percentages, y=readable_features, orientation='h',
            title='Top Features Contributing to Readmission Risk',
            labels={'x': 'Contribution (%)', 'y': 'Risk Factors'},
            color=percentages, color_continuous_scale='RdYlBu_r'
        )
        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

def insights_page():
    st.header("Model Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        metrics_data = {
            'Metric': ['AUC-ROC', 'Recall', 'Specificity', 'Precision'],
            'Score': [0.678, 0.627, 0.632, 0.177]
        }
        
        for metric, score in zip(metrics_data['Metric'], metrics_data['Score']):
            st.metric(metric, f"{score:.3f}")
        
        metrics_df = pd.DataFrame(metrics_data)
        fig = px.bar(metrics_df, x='Metric', y='Score', title='Model Performance', range_y=[0,1])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Key Predictive Features")
        common_features = [
            'Time in Hospital', 'Prior Inpatient Stays', 'ER Visits',
            'Number of Medications', 'Number of Diagnoses', 'Age',
            'Lab Procedures', 'Insulin Use', 'Diabetes Medications'
        ]
        
        features_df = pd.DataFrame({
            'Feature': common_features, 
            'Importance': range(len(common_features), 0, -1)
        })
        fig = px.bar(features_df, y='Feature', x='Importance', orientation='h', 
                    title='Key Risk Factors')
        st.plotly_chart(fig, use_container_width=True)

def about_page():
    st.header("About")
    st.write("""
    ## 30-Day Diabetes Readmission Risk Predictor
    
    **Clinical Purpose**: Predicts likelihood of diabetic patients being readmitted within 30 days of discharge.
    
    **Key Features**:
    - Individual risk scoring
    - Feature contribution analysis  
    - Personalized prevention plans
    
    **Performance**: 
    - AUC-ROC: 0.68
    - Recall: 0.63
    
    ⚠️ **Important**: This tool supports clinical decision-making but does not replace professional medical judgment.
    """)

def main():
    predictor = Predictor()
    
    if not predictor.model:
        with st.spinner("Loading model..."):
            predictor.load_model()
    
    page = st.sidebar.selectbox("Navigate", ["30-Day Prediction", "Risk Analysis", "About"])
    
    if page == "30-Day Prediction":
        prediction_page(predictor)
    elif page == "Risk Analysis":
        insights_page()
    else:
        about_page()

if __name__ == "__main__":
    main()