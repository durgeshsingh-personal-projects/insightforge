# src/streamlit_app.py

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

from src.data_prep import load_data, basic_aggregates, filter_data
from src.knowledge_base import build_documents, build_vectorstore
from src.retriever import HybridRetriever
from src.rag_chain import build_rag_chain, run_rag_answer

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="InsightForge", layout="wide")

# -------------------------------
# Load assets (cached)
# -------------------------------
@st.cache_resource
def load_assets():
    df = load_data("data/sales_data.csv")
    docs = build_documents(df)
    vs = build_vectorstore(df, persist=True)
    return df, vs

df, vs = load_assets()

# Initialize retriever and RAG chain
retriever = HybridRetriever(vs, df)
chain = build_rag_chain(vs)

# -------------------------------
# Sidebar filters
# -------------------------------
st.sidebar.header("Filters")

product = st.sidebar.selectbox("Product", options=['All'] + list(df['Product'].unique()))
region = st.sidebar.selectbox("Region", options=['All'] + list(df['Region'].unique()))

min_date = df['Date'].min()
max_date = df['Date'].max()
date_range = st.sidebar.date_input("Date range", [min_date, max_date])

# -------------------------------
# Filter data based on user selection
# -------------------------------
df_filtered = filter_data(
    df,
    product=None if product=='All' else product,
    region=None if region=='All' else region,
    start_date=date_range[0].strftime("%Y-%m-%d") if date_range else None,
    end_date=date_range[1].strftime("%Y-%m-%d") if date_range else None
)

# -------------------------------
# Main UI
# -------------------------------
st.title("InsightForge — AI Business Intelligence Assistant")
st.markdown("Ask questions about your sales data or use the filters to generate charts and insights.")

col1, col2 = st.columns([2,1])

with col1:
    query = st.text_input("Ask InsightForge", value="How did sales perform in Q1 2022?")

    if st.button("Run Query"):
        filters_dict = {}
        if product != 'All':
            filters_dict['Product'] = product
        if region != 'All':
            filters_dict['Region'] = region

        # Retrieve context rows + stats
        docs, stats = retriever.retrieve(query, k=6, filters=filters_dict)

        # Generate AI answer using RAG
        answer = run_rag_answer(chain, docs, stats, query)

        # Display AI answer
        st.subheader("AI Answer")
        st.write(answer)

        # Reference rows
        st.subheader("Reference Rows")
        for d in docs:
            st.write(d.page_content)

with col2:
    st.subheader("Key Visualizations")

    aggs = basic_aggregates(df_filtered)

    # ----- Revenue Over Time -----
    df_month = aggs['sales_by_month']
    chart = alt.Chart(df_month).mark_line(point=True).encode(
        x='month', y='revenue'
    ).properties(width=350, height=250, title="Revenue Over Time")
    st.altair_chart(chart)

    # ----- Top Products -----
    st.subheader("Top Products")
    st.table(aggs['sales_by_product'].head(10))

    # ----- Revenue by Region -----
    st.subheader("Revenue by Region")
    df_region = aggs['sales_by_region']
    if not df_region.empty:
        chart_region = alt.Chart(df_region).mark_bar().encode(
            x='Region', y='revenue', color='Region'
        ).properties(width=350, height=250)
        st.altair_chart(chart_region)
    else:
        st.write("Regional data not available.")

    # ----- Customer Segmentation -----
    st.subheader("Customer Segmentation")

    # Age distribution
    chart_age = alt.Chart(df_filtered).mark_bar().encode(
        x=alt.X("Customer_Age:Q", bin=alt.Bin(maxbins=20)),
        y='count()'
    ).properties(width=350, height=200, title="Customer Age Distribution")
    st.altair_chart(chart_age)

    # Gender distribution
    chart_gender = alt.Chart(df_filtered).mark_bar().encode(
        x='Customer_Gender:N',
        y='count()',
        color='Customer_Gender:N'
    ).properties(width=350, height=200, title="Customer Gender Distribution")
    st.altair_chart(chart_gender)

    # Satisfaction distribution
    chart_satisfaction = alt.Chart(df_filtered).mark_bar().encode(
        x=alt.X("Customer_Satisfaction:Q", bin=alt.Bin(maxbins=10)),
        y='count()'
    ).properties(width=350, height=200, title="Customer Satisfaction Distribution")
    st.altair_chart(chart_satisfaction)
