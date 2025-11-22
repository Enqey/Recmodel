# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:31:34 2024

@author: Lenovo
"""

# Updated Recommendation System with Your Dataset

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data_url = 'https://raw.githubusercontent.com/Enqey/Recmodel/main/Sdata.csv'  # Corrected raw GitHub link

st.title("🛒 Product Recommendation System")
st.write("""
    **Find products similar to your favorites!**  
    Select a product, and we'll recommend others you might like.
""")

# Load dataset with error handling
try:
    df = pd.read_csv(data_url, encoding='Windows-1252', on_bad_lines='skip')
    st.success("Dataset loaded successfully!")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Check for missing required columns
required_columns = ['Category', 'Sub-Category', 'Product Name', 'Product ID']
if not all(col in df.columns for col in required_columns):
    st.error(f"Error: Dataset is missing required columns: {required_columns}")
    st.stop()

# Handle missing data
if df[required_columns].isnull().any().any():
    st.warning("Missing values detected. Filling with placeholder values...")
    df.fillna('Unknown', inplace=True)

# Feature engineering: Combine text features
df['combined_features'] = df['Category'] + ' ' + df['Sub-Category'] + ' ' + df['Product Name'] + ' ' + df['Product ID']

# Vectorize combined features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# Sidebar for product selection
product_names = df['Product Name'].tolist()
selected_product = st.sidebar.selectbox("Select a product:", product_names)

if selected_product:
    st.subheader(f"Products similar to: {selected_product}")
    
    # Get the index of the selected product
    product_idx = df[df['Product Name'] == selected_product].index[0]
    
    # Compute similarity scores for the selected product
    sim_scores = list(enumerate(cosine_sim[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:8]  # Top 7 recommendations
    
    # Display recommendations
    if sim_scores:
        for idx, (product_index, score) in enumerate(sim_scores):
            recommended_product = df.iloc[product_index]['Product Name']
            st.write(f"{idx + 1}. {recommended_product}")
    else:
        st.write("No similar products found.")
        
st.write("---")
st.write("**Developed by Nana Ekow Okusu**")

