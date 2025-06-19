"""
This code invastigates association rules by applying apriori to a
dataset of the binary transformation of groceries.csv saved as transaction_binary.csv

It reads the df, calculates the frequent items and association rules with respect to support metric wit min_support %5
It logs results and total time cost to seeing and comparing performance of the system.
"""


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import time
import logging

logging.basicConfig(filename='apriori_classic.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

start = time.time()
data = pd.read_csv(r"E:\piton\association\transaction_binary.csv")

frequent_itemsets = apriori(data, min_support=0.05, use_colnames=True)
logging.info('Frequent Itemsets:\n%s', frequent_itemsets)
rules = association_rules(frequent_itemsets, metric = 'support', min_threshold=0.05)
logging.info('Association Rules:\n%s', rules)

finish = time.time()
totaltime = finish - start
logging.info('Total Time: %.2f seconds\n', totaltime)
print(f"total time= {totaltime:.2f}")
print(frequent_itemsets)
print(rules)




