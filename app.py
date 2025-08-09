import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import IsolationForest
from kpi_generator import generate_kpis
from recommender import recommend_similar_products
from nl_query import run_local_llm_query
import os

st.set_page_config(page_title="DataVista BI", layout="wide")

# Custom CSS for layout and fonts
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .metric-container {
            display: flex;
            justify-content: space-between;
        }
        .metric-container > div {
            flex: 1;
            margin-right: 1rem;
        }
        .metric-container > div:last-child {
            margin-right: 0;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("---")
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=150)
else:
    st.sidebar.markdown("🚀 **DataVista BI**", unsafe_allow_html=True)

st.sidebar.markdown("📊 **Business Intelligence Dashboard**", unsafe_allow_html=True)
st.sidebar.markdown("---")

# How to Use
st.sidebar.markdown("### 📝 How to Use")
st.sidebar.markdown("""
1. 📂 **Upload `Orders.csv`**  
2. 📂 **Upload `Products.csv`**  
3. 📊 **Explore KPIs, Charts, and Recommendations**  
4. 💬 **Ask questions to your data**
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Use Cases
st.sidebar.markdown("### 📌 Use Cases")
st.sidebar.markdown("""
- 📈 Monitor real-time KPIs  
- 📉 Detect revenue anomalies with ML  
- 🛒 Identify best- & worst-performing products  
- 🤖 Get smart product recommendations  
- 💬 Ask natural language questions with LLaMA 3  
- 🧠 Make faster decisions  
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# --- Centered Title ---
st.markdown(
    """
    <h1 style='text-align: center; color: #4A90E2; font-family: "Segoe UI", sans-serif;'>
        📊 DataVista BI Dashboard
    </h1>
    """,
    unsafe_allow_html=True
)

# File upload section
col1, col2 = st.columns(2)
with col1:
    orders_file = st.file_uploader("📂 Upload Orders CSV", type=["csv"], key="orders")
with col2:
    products_file = st.file_uploader("📂 Upload Products CSV", type=["csv"], key="products")

# Process uploaded files
if orders_file is not None and products_file is not None:
    try:
        orders_df = pd.read_csv(orders_file)
        products_df = pd.read_csv(products_file)
        merged_df = pd.merge(orders_df, products_df, on="product_id")

        st.markdown("---")

        # KPI Section
        st.subheader("📌 KPIs")
        kpis = generate_kpis(orders_df, products_df)
        metric_cols = st.columns(len(kpis))
        for idx, (kpi_name, kpi_value) in enumerate(kpis.items()):
            metric_cols[idx].metric(label=kpi_name, value=kpi_value)

        st.markdown("---")
        
        # Revenue Trend & Category Revenue Charts
        st.subheader("📊 Revenue Insights")
        merged_df["revenue"] = merged_df["quantity"] * merged_df["price"]
        merged_df["order_date"] = pd.to_datetime(merged_df["order_date"])
        merged_df["month"] = merged_df["order_date"].dt.to_period("M").astype(str)

        col3, col4 = st.columns(2)
        with col3:
            monthly_revenue = merged_df.groupby("month")["revenue"].sum().reset_index()
            fig = px.line(monthly_revenue, x="month", y="revenue", title="Monthly Revenue", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            category_revenue = merged_df.groupby("category")["revenue"].sum().reset_index().sort_values(by="revenue", ascending=False)
            fig2 = px.pie(category_revenue, values='revenue', names='category', title='Revenue by Category')
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        # Product Recommendations
        st.subheader("🔍 Product Insights")
        col5, col6 = st.columns(2)

        with col5:
            top_products = merged_df.groupby("name")["quantity"].sum().reset_index().sort_values(by="quantity", ascending=False).head(5)
            st.markdown("**🔥 Top 5 Best-Selling Products:**")
            for _, row in top_products.iterrows():
                st.write(f"➡️ {row['name']} (Sold: {int(row['quantity'])})")

        with col6:
            bottom_products = merged_df.groupby("name")["quantity"].sum().reset_index().sort_values(by="quantity", ascending=True).head(5)
            st.markdown("**🧊 Consider Promoting These Low-Sellers:**")
            for _, row in bottom_products.iterrows():
                st.write(f"⚠️ {row['name']} (Sold: {int(row['quantity'])})")

        st.markdown("---")

        # Anomaly Detection
        st.subheader("📉 Revenue Anomaly Detection")
        daily_revenue = merged_df.groupby('order_date')['revenue'].sum().reset_index()
        model = IsolationForest(contamination=0.05, random_state=42)
        daily_revenue['anomaly'] = model.fit_predict(daily_revenue[['revenue']])
        daily_revenue['is_anomaly'] = daily_revenue['anomaly'] == -1
        fig3 = px.line(daily_revenue, x='order_date', y='revenue', title='Revenue Over Time')
        fig3.add_scatter(x=daily_revenue[daily_revenue['is_anomaly']]['order_date'],
                         y=daily_revenue[daily_revenue['is_anomaly']]['revenue'],
                         mode='markers', name='Anomaly', marker=dict(color='red', size=10))
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")

        # Product Recommender
        st.subheader("🤖 Smart Recommender")
        selected_product = st.selectbox("Select a product to get similar recommendations:", merged_df["name"].unique())
        if selected_product:
            similar_products = recommend_similar_products(selected_product, merged_df)
            st.success(f"Recommended Products: {similar_products}")

        st.markdown("---")

        # Downloads
        st.subheader("⬇️ Downloads")
        kpi_df = pd.DataFrame(kpis.items(), columns=["KPI", "Value"])
        st.download_button("📥 Download KPI Summary", data=kpi_df.to_csv(index=False).encode('utf-8'), file_name="kpi_summary.csv", mime="text/csv")
        st.download_button("📥 Download Processed Data", data=merged_df.to_csv(index=False).encode('utf-8'), file_name='merged_data.csv', mime='text/csv')

        st.markdown("---")

        # Natural Language Queries
        st.subheader("💬 Ask Your Data (LLaMA 3)")
        user_query = st.text_input("Type a question about your data:")
        if user_query:
            response = run_local_llm_query(user_query, orders_df, products_df)
            st.info(response)

  

        # Footer
        st.markdown(
            "<hr style='border-top: 1px solid #999;'>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align: center; color: gray;'>🚀 DataVista BI Dashboard — © 2025 Rudraksh Zodage</p>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"❌ Error processing files: {e}")
else:
    st.warning("📌 Please upload both Orders and Products CSV files to proceed.")
