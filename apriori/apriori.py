from http.client import responses

import pandas as pd
import csv
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from numpy.ma.extras import unique

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)  # Sütun genişliklerini sınırsız yap
pd.set_option('display.expand_frame_repr', False)


transactions = []
with open ('groceries.csv') as file:
    reader = csv.reader(file)
    for row in reader:
        transactions.append(row)
"""
for i in range(5):
    print(transactions[i])
"""

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
veri = pd.DataFrame(te_ary, columns=te.columns_)
veri.to_csv('transection_binary.csv', index= False)



#apriori algoritmasını kuruyoruz
frequent_itemsets = apriori(veri, min_support=0.05, use_colnames=True)
#print(frequent_itemsets)
rules = association_rules(frequent_itemsets, metric = 'lift', min_threshold=1)
#print(rules)

unique_items = sorted(te.columns_)
#print(unique_items)
def get_recommendation(item):
    rule = rules[rules['antecedents'] == {item}]
    if rule.empty:
        return None
    recommendation = rule.sort_values(by='confidence', ascending = False).head(1)
    return recommendation['consequents'].values[0]

#market uygulaması
def market_app():
    print("Markete hoş geldiniz!")
    sepet = []

    while True:
        print("Lütfen aşağıdaki ürünlerden birini seçiniz:")
        for idx, item in enumerate(unique_items):
            print(f"{idx + 1}. {item}")

        choice = int(input("Ürün numarasını girin: ")) - 1
        selected_item = unique_items[choice]
        sepet.append(selected_item)
        print(f"Sepetinize '{selected_item}' eklediniz.")

        recommendation = get_recommendation(selected_item)
        if recommendation:
            print(f"Size '{recommendation}' ürününü öneriyoruz.")
            response = input("Öneriyi kabul ediyor musunuz? (e/h): ").lower()
            if response == 'e':
                sepet.append(recommendation)
                print(f"'{recommendation}' ürününü de sepetinize eklediniz.")

        devam = input("Alışverişe devam etmek istiyor musunuz? (e/h): ").lower()
        if devam == 'h':
            break

    print("Alışverişi tamamladınız. Sepetinizdeki ürünler:")
    for item in sepet:
        print(f"- {item}")
    print("İyi alışverişler!")

market_app()