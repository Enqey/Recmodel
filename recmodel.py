# Updated Recommendation System with Your Dataset

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data_url = 'https://github.com/Enqey/Recmodel/blob/main/Superstore.csv'

st.title("🛒 Product Recommendation System")
st.write("""
    **Find products similar to your favorites!**  
    Select a product, and we'll recommend others you might like.
""")

try:
  df = pd.read_csv(data_url, encoding='Windows-1252')
    st.success("Dataset loaded successfully!")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Check for missing values
required_columns = ['Category', 'Sub-Category', 'Product Name']
missing_data = df[required_columns].isnull().any().any()
if missing_data:
    st.warning("Missing values detected. Filling with placeholder values...")
    df.fillna('Unknown', inplace=True)

# Feature engineering
df['combined_features'] = df['Category'] + ' ' + df['Sub-Category'] + ' ' + df['Product Name']

# Vectorize features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# Sidebar product selection
product_names = df['Product Name'].tolist()
selected_product = st.sidebar.selectbox("Select a product:", product_names)

if selected_product:
    st.subheader(f"Products similar to: {selected_product}")
    
    # Get index of the selected product
    product_idx = df[df['Product Name'] == selected_product].index[0]
    
    # Compute similarity scores
    sim_scores = list(enumerate(cosine_sim[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:8]  # Top 7
    
    # Display recommendations
    for idx, (product_index, score) in enumerate(sim_scores):
        st.write(f"{idx + 1}. {df.iloc[product_index]['Product Name']}")

st.write("---")
st.write("**Developed by Nana Ekow Okusu**")
