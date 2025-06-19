"""
This code uses a previously saved fptree by fptree_builder.py to apply fpgrowth faster.
It saves results and total time cost to seeing and comparing performance of the system.
"""

import time
import logging
import pickle
from mlxtend.frequent_patterns import association_rules
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

logging.basicConfig(filename='fptree_usage.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

start = time.time()

with open('fp_tree.pkl', 'rb') as f:
    frequent_itemsets = pickle.load(f)

logging.info('Frequent Itemsets Loaded:\n%s', frequent_itemsets)

rules = association_rules(frequent_itemsets, metric='support', min_threshold=0.05)
logging.info('Association Rules:\n%s', rules)

finish = time.time()
totaltime = finish - start

logging.info('Total Time: %.2f seconds\n', totaltime)

print(f"Total time = {totaltime:.2f}")
print(rules)
