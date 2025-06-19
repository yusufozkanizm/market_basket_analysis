"""
This code builds and saves a fptree from a dataset
of binary transformation of groceries.csv saved as transaction_binary.csv
"""


import pandas as pd
import pickle
from mlxtend.frequent_patterns import fpgrowth

data = pd.read_csv(r"E:\piton\association\transaction_binary.csv")

frequent_itemsets = fpgrowth(data, min_support=0.05, use_colnames=True)

with open('fp_tree.pkl', 'wb') as f:
    pickle.dump(frequent_itemsets, f)

print("process completed")
