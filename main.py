# src/main.py
from preprocessing import load_data, preprocess_data
from clustering import perform_clustering, save_model
from frequent_patterns import generate_frequent_patterns

def main():
    filepath = 'dataset/city_day.csv'
    
    # Step 1: Load and Preprocess
    df = load_data(filepath)
    df = preprocess_data(df)
    
    # Step 2: Clustering
    df, model = perform_clustering(df, n_clusters=3)
    save_model(model, 'models/air_quality_kmeans.pkl')
    
    # Step 3: Frequent Pattern Mining
    rules = generate_frequent_patterns(df.select_dtypes(include=[float, int]))
    print("Top Frequent Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence']])
    
    # Step 4: Save clustered results
    df.to_csv('outputs/clustered_air_quality.csv', index=False)
    
if __name__ == "__main__":
    main()
