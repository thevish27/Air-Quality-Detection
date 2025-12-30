# src/clustering.py
from sklearn.cluster import KMeans
import pickle

def perform_clustering(df, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    features = df.select_dtypes(include=[float, int]).values
    labels = model.fit_predict(features)
    df['cluster'] = labels
    return df, model

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
