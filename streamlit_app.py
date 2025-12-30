import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Streamlit App Title
st.title("Air Quality Monitoring & Clustering")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Dataset")
    st.write(df)

    # Check for missing values
    if df[['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']].isnull().values.any():
        st.warning("⚠️ Some missing values were detected and will be handled automatically.")

    # Extract relevant features
    features = df[['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']]

    # Handle missing values by dropping rows (you can switch to imputation if preferred)
    features = features.dropna()

    # Apply KMeans clustering
    model = KMeans(n_clusters=3, random_state=42)
    labels = model.fit_predict(features)

    # Add cluster labels back to the main dataframe (matching index after dropna)
    cleaned_df = df.loc[features.index].copy()
    cleaned_df['Cluster'] = labels

    st.subheader("Clustered Data")
    st.write(cleaned_df)

    # Visualize clusters using PM2.5 and PM10 (you can choose others too)
    st.subheader("PM2.5 vs PM10 Clustering")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=cleaned_df, x='PM2.5', y='PM10', hue='Cluster', palette='viridis')
    plt.title("KMeans Clustering based on PM2.5 and PM10")
    st.pyplot(plt)

    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(features.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)

else:
    st.info("👆 Please upload a CSV file to begin.")
