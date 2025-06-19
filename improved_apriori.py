import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import time
import logging
from itertools import combinations

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

start = time.time()

df=pd.read_csv("transaction_binary.csv")


# Count how many transactions include each item (i.e., number of True values per column)
item_transaction_counts = df.sum()

# Create a dictionary with each item and the list of transaction indices that include it
item_transaction_lists = {
    item: df.index[df[item] == True].tolist()
    for item in df.columns
}

# Display the item counts
print("Transaction count per item:")
print(item_transaction_counts)

threshold = 0.05 * len(df)
frequent_items = item_transaction_counts[item_transaction_counts >= threshold].index.tolist()

single_set_df = df[frequent_items]
single_set_items = frequent_items

print("Items appearing in at least 5% of transactions:")
print(single_set_items)

two_itemsets = list(combinations(frequent_items, 2))
two_itemset_counts = []
for item1, item2 in two_itemsets:
    count = ((single_set_df[item1] == True) & (single_set_df[item2] == True)).sum()
    two_itemset_counts.append(((item1, item2), count))

# Convert to DataFrame for easier visualization or further processing
two_itemset_df = pd.DataFrame(two_itemset_counts, columns=["Item Pair", "Transaction Count"])

min_support_count = 0.05*len(two_itemset_df)
frequent_two_sets_df = two_itemset_df[two_itemset_df["Transaction Count"] >= min_support_count]
frequent_two_sets = frequent_two_sets_df["Item Pair"].tolist()
finish = time.time()
# Display result
print(two_itemset_df)
print(frequent_two_sets_df)

duration = finish - start
print(duration)


