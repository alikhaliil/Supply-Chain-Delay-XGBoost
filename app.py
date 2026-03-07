import streamlit as st
import pandas as pd
import joblib

# إعدادات الصفحة
st.set_page_config(
    page_title="Shipment Delay Prediction | Port Logistics",
    page_icon="🚢",
    layout="wide"
)

# تحميل الموديل والأعمدة مرة واحدة بس لتسريع الأبلكيشن
@st.cache_resource
def load_model():
    model = joblib.load('deploy_xgb_model.pkl')
    cols = joblib.load('deploy_model_columns.pkl')
    return model, cols

model, expected_cols = load_model()

# عنوان الصفحة
st.title("🚢 Supply Chain Logistics: Shipment Delay Predictor")
st.markdown("""
This tool predicts the probability of a shipment being delayed based on scheduled days, shipping mode, and geographical risk scores. 
Used for proactive logistics management.
""")
st.divider()

# الشريط الجانبي للمدخلات
st.sidebar.header("📦 Input Shipment Details")

# قاموس لتحويل النصوص لأرقام زي ما الـ LabelEncoder عملها في الجوبيتر
shipping_dict = {
    'First Class': 0, 
    'Same Day': 1, 
    'Second Class': 2, 
    'Standard Class': 3
}

shipping_type = st.sidebar.selectbox('Shipping Type', list(shipping_dict.keys()))
scheduled_days = st.sidebar.number_input('Days for shipment (scheduled)', min_value=0, max_value=30, value=3)
geo_risk_score = st.sidebar.slider('Geographical Risk Score', min_value=0.0, max_value=1.0, value=0.5, help="0 = Low Risk, 1 = High Risk")

# زرار التوقع
if st.button("Predict Delivery Status", type="primary"):
    
    # 1. تجهيز الداتا اللي هتدخل للموديل
    shipping_mode_encoded = shipping_dict[shipping_type]
    interaction = shipping_mode_encoded * geo_risk_score # حساب الميزة الإضافية
    
    # عمل Dataframe بنفس ترتيب الأعمدة المحفوظة
    input_data = pd.DataFrame([[
        scheduled_days, 
        shipping_mode_encoded, 
        geo_risk_score, 
        interaction
    ]], columns=expected_cols)
    
    # 2. التوقع
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1] * 100
    
    # 3. عرض النتائج بشكل احترافي
    st.divider()
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("⚠️ High Risk of Delay")
            st.markdown("This shipment is **likely to be delayed**. Recommend proactive intervention.")
        else:
            st.success("✅ On Time")
            st.markdown("This shipment is **expected to arrive on schedule**.")
            
    with col2:
        st.metric(label="Delay Probability", value=f"{prediction_proba:.1f}%")