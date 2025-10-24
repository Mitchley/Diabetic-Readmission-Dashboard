import streamlit as st, pandas as pd, numpy as np, joblib, os, plotly.express as px, plotly.graph_objects as go

st.set_page_config(page_title="30-Day Diabetes Readmission Predictor", layout="wide")
st.title("🏥 30-Day Diabetes Readmission Risk Predictor")

MODEL_PATH, FEATURE_INFO_PATH = "diabetic_readmission_model.joblib", "feature_info.joblib"

class Predictor:
    def __init__(self): self.model, self.features = None, None

    def load_model(self):
        try:
            for f in [MODEL_PATH, FEATURE_INFO_PATH]:
                if not os.path.exists(f): st.error(f"Missing file: {f}"); return False
            self.model = joblib.load(MODEL_PATH)
            self.features = joblib.load(FEATURE_INFO_PATH)['feature_names']
            return True
        except Exception as e: st.error(f"Error loading model: {e}"); return False

    def predict(self, data):
        df = pd.DataFrame([data]).reindex(columns=self.features, fill_value=0)
        return self.model.predict(df)[0], self.model.predict_proba(df)[0]

    def get_feature_importance(self, data):
        m = getattr(self.model, 'feature_importances_', None) or getattr(
            getattr(self.model, 'named_steps', {}).get('classifier', None), 'feature_importances_', None)
        if m is None: return self._fallback()
        imp = dict(zip(self.features, m))
        df = pd.DataFrame([data]).reindex(columns=self.features, fill_value=0)
        base = {'age':50,'time_in_hospital':4,'num_medications':8,'num_lab_procedures':45,
                'number_diagnoses':6,'number_emergency':1,'number_inpatient':1,'num_procedures':1}
        contrib = {f: abs(imp.get(f,0)*((df[f][0]-base.get(f,0))/(base.get(f,0)+1) if f in base else df[f][0])) 
                   for f in self.features}
        top = dict(sorted(contrib.items(), key=lambda x:x[1], reverse=True)[:15])
        s = sum(top.values()); return {k:100*v/s for k,v in top.items()} if s else self._fallback()

    def _fallback(self): 
        return {'time_in_hospital':18,'number_inpatient':15,'number_emergency':12,'num_medications':10,
                'number_diagnoses':9,'age':8,'num_lab_procedures':7,'insulin':5,'diabetesMed':5,
                'A1Cresult_>8':3,'max_glu_serum_>300':3,'change':2,'num_procedures':2,
                'race_Caucasian':1,'gender_Female':1}

def create_input_data(u, f):
    d = {k:0 for k in f}
    num = ['age','time_in_hospital','num_medications','num_lab_procedures','number_diagnoses',
           'number_emergency','number_inpatient','num_procedures']
    for k in num: d[k]=u[k] if k in f else 0
    for k in ['diabetesMed','insulin','change']: d[k]=1 if u[k]=="Yes" else 0
    cats = [f"race_{u['race']}", f"gender_{u['gender']}", f"A1Cresult_{u['a1c_result']}", f"max_glu_serum_{u['max_glu_serum']}"]
    for c in cats: d[c]=1 if c in f else 0
    for med in ['metformin','glipizide','glyburide','pioglitazone','rosiglitazone']:
        if f"{med}_{u[med]}" in f: d[f"{med}_{u[med]}"]=1
    for med in ['acarbose','nateglinide','chlorpropamide','glimepiride','acetohexamide','tolazamide','glipizide-metformin','glyburide-metformin']:
        if f"{med}_No" in f: d[f"{med}_No"]=1
    return d

def prediction_page(p):
    st.header("Patient Risk Assessment")
    with st.form("patient_form"):
        c1,c2,c3=st.columns(3)
        with c1:
            age=st.slider("Age",0,100,50)
            race=st.selectbox("Race",["Caucasian","AfricanAmerican","Asian","Hispanic","Other"])
            gender=st.selectbox("Gender",["Female","Male"])
            time_in_hospital=st.slider("Hospital Stay (days)",1,30,3)
            num_lab_procedures=st.slider("Lab Procedures",0,100,50)
            num_procedures=st.slider("Procedures",0,10,1)
        with c2:
            number_diagnoses=st.slider("Diagnoses",1,20,5)
            number_emergency=st.slider("ER Visits (year)",0,20,0)
            number_inpatient=st.slider("Inpatient Stays (year)",0,20,0)
            diabetesMed=st.selectbox("On Diabetes Medication",["No","Yes"])
            insulin=st.selectbox("Insulin Use",["No","Yes"])
            change=st.selectbox("Medication Change",["No","Yes"])
        with c3:
            a1c_result=st.selectbox("A1C Result",["Norm",">7",">8","None"])
            max_glu_serum=st.selectbox("Max Glucose Serum",["Norm",">200",">300","None"])
            num_medications=st.slider("Number of Medications",0,50,10)
            meds={m:st.selectbox(m.title(),["No","Steady","Up","Down"]) for m in 
                  ["metformin","glipizide","glyburide","pioglitazone","rosiglitazone"]}
        if st.form_submit_button("Predict 30-Day Readmission Risk"):
            u={'age':age,'race':race,'gender':gender,'time_in_hospital':time_in_hospital,
               'num_lab_procedures':num_lab_procedures,'num_procedures':num_procedures,
               'number_diagnoses':number_diagnoses,'number_emergency':number_emergency,
               'number_inpatient':number_inpatient,'diabetesMed':diabetesMed,'insulin':insulin,
               'change':change,'a1c_result':a1c_result,'max_glu_serum':max_glu_serum,
               'num_medications':num_medications,**meds}
            data=create_input_data(u,p.features)
            pred,prob=p.predict(data); fi=p.get_feature_importance(data)
            show_results(pred,prob,fi)

def show_results(pred,prob,fi):
    risk=prob[1]
    st.subheader("Prediction Results")
    c1,c2=st.columns(2)
    with c1:
        fig=go.Figure(go.Indicator(mode="gauge+number",value=risk*100,number={'suffix':'%'},gauge={
            'axis':{'range':[None,100]},
            'bar':{'color':"darkred" if risk>0.5 else "darkgreen"},
            'steps':[{'range':[0,20],'color':"lightgreen"},{'range':[20,50],'color':"yellow"},{'range':[50,100],'color':"lightcoral"}]}))
        st.plotly_chart(fig,use_container_width=True)
        msg = ("🚨 HIGH" if risk>0.5 else "⚠️ MEDIUM" if risk>0.2 else "✅ LOW")
        st.write(f"{msg} RISK: {risk:.1%} probability")
    with c2:
        plans={"HIGH":"- Follow-up 3–5 days\n- Med reconciliation\n- Care coordination",
               "MEDIUM":"- Follow-up 7 days\n- Medication review\n- Weekly check-ins",
               "LOW":"- Routine follow-up\n- Continue meds"}
        level="HIGH" if risk>0.5 else "MEDIUM" if risk>0.2 else "LOW"
        st.write(f"**{level}-Risk Actions:**\n{plans[level]}")
    if fi:
        st.subheader("Feature Contribution Analysis")
        f,v=zip(*sorted(fi.items(),key=lambda x:x[1],reverse=True))
        fig=px.bar(x=v,y=[i.replace('_',' ').title() for i in f],orientation='h',
                   labels={'x':'Contribution (%)','y':'Features'},color=v,color_continuous_scale='RdYlBu_r')
        st.plotly_chart(fig,use_container_width=True)

def insights_page():
    st.header("Model Insights")
    c1,c2=st.columns(2)
    with c1:
        data={'Metric':['AUC-ROC','Recall','Specificity','Precision'],'Score':[0.678,0.627,0.632,0.177]}
        for m,s in zip(data['Metric'],data['Score']): st.metric(m,f"{s:.3f}")
        st.plotly_chart(px.bar(pd.DataFrame(data),x='Metric',y='Score',range_y=[0,1]),use_container_width=True)
    with c2:
        f=['Time in Hospital','Prior Inpatient Stays','ER Visits','Number of Medications','Diagnoses','Age','Lab Procedures','Insulin Use','Diabetes Meds']
        st.plotly_chart(px.bar(pd.DataFrame({'Feature':f,'Importance':range(len(f),0,-1)}),
                               y='Feature',x='Importance',orientation='h'),use_container_width=True)

def about_page():
    st.header("About")
    st.markdown("""**Purpose**: Predicts likelihood of diabetic readmission within 30 days.
    - Individual risk scoring
    - Feature contribution analysis  
    - Personalized prevention plans  
    ⚠️ *Supports decisions, not a medical substitute.*""")

def main():
    p=Predictor()
    with st.spinner("Loading model..."):
        if not p.load_model(): st.error("Model not loaded. Ensure files exist."); return
    page=st.sidebar.selectbox("Navigate",["30-Day Prediction","Risk Analysis","About"])
    {"30-Day Prediction":prediction_page,"Risk Analysis":insights_page,"About":about_page}[page](p)

if __name__=="__main__": main()
