# src/frequent_patterns.py
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

def generate_frequent_patterns(df, min_support=0.1):
    df_bin = df.copy()
    for col in df_bin.columns:
        df_bin[col] = df_bin[col] > df_bin[col].mean()
    
    patterns = apriori(df_bin, min_support=min_support, use_colnames=True)
    rules = association_rules(patterns, metric="confidence", min_threshold=0.7)
    return rules
