import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Shipment Delay Risk Predictor", layout="wide")

st.title("📦 Shipment Delay Risk Predictor")
st.markdown("""
This tool predicts the risk of a shipment being delivered late, based on historical patterns 
from 180,000+ real supply chain records. Built with a focus on logistics operations in the 
Middle East and globally.
""")

model = joblib.load('late_delivery_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.header("Enter Shipment Details")

col1, col2 = st.columns(2)

with col1:
    shipping_mode = st.selectbox(
        "Shipping Mode",
        ["Standard Class", "First Class", "Second Class", "Same Day"]
    )
    scheduled_days = st.slider("Days for Shipment (Scheduled)", 0, 6, 2)
    order_region = st.selectbox(
        "Order Region",
        ["West Asia", "Western Europe", "Southeast Asia", "South Asia", "East Africa",
         "West Africa", "North Africa", "South America", "Central America", "Caribbean",
         "Eastern Europe", "Northern Europe", "Southern Europe", "Eastern Asia",
         "Central Asia", "Oceania", "Canada", "US Center", "East of USA", "West of USA",
         "South of USA", "Central Africa", "Southern Africa"]
    )

with col2:
    discount_rate = st.slider("Order Item Discount Rate", 0.0, 0.5, 0.1, 0.01)
    product_price = st.number_input("Order Item Product Price ($)", min_value=0.0, value=100.0)
    customer_segment = st.selectbox("Customer Segment", ["Consumer", "Corporate", "Home Office"])

# The Predict button - only runs the code below when clicked
if st.button("Predict Delay Risk", type="primary"):

    # Step 1: Build a single-row dictionary matching our raw feature names
    input_dict = {
        'Days for shipment (scheduled)': scheduled_days,
        'Benefit per order': 0,  # unknown at prediction time, use neutral default
        'Sales per customer': product_price,  # reasonable proxy
        'Order Item Discount Rate': discount_rate,
        'Order Item Product Price': product_price,
        'Is_West_Asia': 1 if order_region == 'West Asia' else 0,
        'High_Discount': 1 if discount_rate > 0.15 else 0,
        'Order_Month': 6,  # neutral default (we showed month barely matters)
        'Order_DayOfWeek': 2,  # neutral default (we showed this barely matters)
        'Order_IsWeekend': 0,
        'Category Name': 'Cleats',  # most common category as default
        'Customer Segment': customer_segment,
        'Order Region': order_region,
        'Shipping Mode': shipping_mode,
    }

    # Step 2: Convert to a DataFrame (a table with 1 row)
    input_df = pd.DataFrame([input_dict])

    # Step 3: One-hot encode it the exact same way we did during training
    input_encoded = pd.get_dummies(input_df, columns=['Category Name', 'Customer Segment', 'Order Region', 'Shipping Mode'])

    # Step 4: Align columns to match exactly what the model expects
    # Any column the model expects but we don't have gets filled with 0
    # Any extra column we have but the model doesn't expect gets dropped
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Step 5: Get the prediction
    prediction = model.predict(input_final)[0]
    probability = model.predict_proba(input_final)[0][1]

    # Step 6: Display results
    st.divider()
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK of Late Delivery — {probability*100:.1f}% probability")
    else:
        st.success(f"✅ LOW RISK of Late Delivery — {probability*100:.1f}% probability")
        # SHAP explanation for this specific prediction
    st.subheader("Why this prediction?")

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_final)
    shap_vals_late = shap_vals[0, :, 1]  # this row, class 1 (late)

    # Build a readable table of the top contributing factors
    contrib_df = pd.DataFrame({
        'Feature': input_final.columns,
        'Impact': shap_vals_late
    })
    contrib_df['Abs_Impact'] = contrib_df['Impact'].abs()
    top_contrib = contrib_df.sort_values('Abs_Impact', ascending=False).head(8)

    # Only show features that actually have a nonzero value for this input (more readable)
    top_contrib = top_contrib[top_contrib['Feature'].apply(lambda f: input_final[f].iloc[0] != 0 or 'Days' in f)]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#d62728' if x > 0 else '#2ca02c' for x in top_contrib['Impact']]
    ax.barh(top_contrib['Feature'], top_contrib['Impact'], color=colors)
    ax.set_xlabel('Impact on Late Delivery Risk (red = increases risk, green = decreases risk)')
    ax.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig)