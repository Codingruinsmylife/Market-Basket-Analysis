#!/usr/bin/env python
# coding: utf-8

# In[1]:
# Import Library

get_ipython().system('pip install mlxtend')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from collections import defaultdict

# In[2]:
# Load data and study its structure

df = pd.read_csv("data.csv", encoding="latin-1")

print("The dimension of dataset is:", df.shape)

# In[4]:
# Inspect the data

print("The first five rows of dataset are:", df.head())

# In[5]:

# Convert the "invoicedate" column to datetime data type
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Categorize the data into different period data
df['Year'] = df['InvoiceDate'].dt.year
df['Quarter'] = df['InvoiceDate'].dt.to_period('Q')
df['Month'] = df['InvoiceDate'].dt.to_period('M')

# Select four quarters with the highest number of transactions
quarterly_counts = df.groupby(['Year', 'Quarter']).size().reset_index(name='Count')
quarterly_counts['ConsecutiveCount'] = (
    quarterly_counts.sort_values(['Year', 'Quarter'])
                  .groupby('Year')['Count']
                  .rolling(4, min_periods=1)
                  .sum()
                  .reset_index(drop=True)
)

top_four_quarters = quarterly_counts.sort_values('ConsecutiveCount', ascending=False).head(4)

# Filter the DataFrame to keep only the data from the top four quarters
df_top_four_quarters = df.merge(top_four_quarters[['Year', 'Quarter']], on=['Year', 'Quarter'])
data_grouped = df_top_four_quarters.groupby(['InvoiceNo', 'Year', 'Quarter'])['Description'].apply(list).reset_index()

top_quarter_transactions = []

# Iterate through the top four quarters and extract transactions
for top_quarter in sorted(top_four_quarters['Quarter']):
    top_quarter_df = data_grouped[data_grouped['Quarter'] == top_quarter]
    transactions = top_quarter_df['Description'].tolist()
    top_quarter_transactions.append(transactions)

quarter1 = top_quarter_transactions[0]
quarter2 = top_quarter_transactions[1]
quarter3 = top_quarter_transactions[2]
quarter4 = top_quarter_transactions[3]

for i, top_quarter in enumerate(sorted(top_four_quarters['Quarter'])):
    print(f"Number of transactions in {top_quarter}: {len(top_quarter_transactions[i])}")
    print()

#%%
    
# Select three consecutive months with the highest number of transactions
monthly_counts = df.groupby(['Year', 'Month']).size().reset_index(name='Count')
monthly_counts['ConsecutiveCount'] = (
    monthly_counts.sort_values(['Year', 'Month'])
                  .groupby('Year')['Count']
                  .rolling(3, min_periods=1)
                  .sum()
                  .reset_index(drop=True)
)

top_three_months = monthly_counts.sort_values('ConsecutiveCount', ascending=False).head(3)

# Filter the DataFrame to keep only the data from the top three months
df_top_three_months = df.merge(top_three_months[['Year', 'Month']], on=['Year', 'Month'])
data_grouped = df_top_three_months.groupby(['InvoiceNo', 'Year', 'Month'])['Description'].apply(list).reset_index()

top_month_transactions = []

# Iterate through the top three months and extract transactions
for top_month in sorted(top_three_months['Month']):
    top_month_df = data_grouped[data_grouped['Month'] == top_month]
    transactions = top_month_df['Description'].tolist()
    top_month_transactions.append(transactions)
    
month1 = top_month_transactions[0]
month2 = top_month_transactions[1]
month3 = top_month_transactions[2]

for i, top_month in enumerate(sorted(top_three_months['Month'])):
    print(f"Number of transactions in {top_month}: {len(top_month_transactions[i])}")
    print()
    

# In[8]:

# Initialize a dictionary to store biweek transaction counts
biweek_transaction_counts = defaultdict(int)
biweek_transactions = defaultdict(list)

biweekly_data_grouped = df_top_three_months.groupby(['InvoiceNo', 'Biweek'])['Description'].apply(list).reset_index()

# Sort biweekly_data_grouped by 'Biweek' column
biweekly_data_grouped = biweekly_data_grouped.sort_values('Biweek')

# Iterate through each biweek and calculate transaction counts
for biweek, group in biweekly_data_grouped.groupby('Biweek'):
    transaction_count = len(group)
    biweek_transaction_counts[biweek] = transaction_count
    transactions = group['Description'].tolist()
    biweek_transactions[biweek] = transactions

sorted_biweeks = sorted(biweek_transaction_counts.keys(), key=lambda biweek: pd.to_datetime(biweek.split(' - ')[0]))

# Iterate through the sorted order of biweeks
for biweek in sorted_biweeks:
    transactions = biweek_transactions[biweek]
    print(f"Number of transactions in {biweek}: {len(transactions)}")
    print()

biweek1 = biweek_transactions[sorted_biweeks[0]]
biweek2 = biweek_transactions[sorted_biweeks[1]]
biweek3 = biweek_transactions[sorted_biweeks[2]]
biweek4 = biweek_transactions[sorted_biweeks[3]]
biweek5 = biweek_transactions[sorted_biweeks[4]]
biweek6 = biweek_transactions[sorted_biweeks[5]]


# In[6]:

# Function to calculate metrics for the top items in a given DataFrame
def calculate_top_item_metrics(df, total_invoices_per_period, period_column, top_n=20):
    top_item_counts = df['Description'].value_counts().head(top_n)
    
    # Initialize lists to store metrics for each item
    top_items_metrics = []
    
    for item in top_item_counts.index:
        item_df = df[df['Description'] == item]
        total_receipts = len(item_df)
        total_sales_quantity = item_df['Quantity'].sum()
        total_sales_value = (item_df['Quantity'] * item_df['UnitPrice']).sum()
        
        # Calculate support for the item based on total invoices
        period = item_df[period_column].iloc[0]
        total_invoices = total_invoices_per_period[period]
        support = total_receipts / total_invoices
        
        top_items_metrics.append((item, total_receipts, total_sales_quantity, total_sales_value, support))
    
    return top_items_metrics


# Calculate the total number of invoices for each quarter
total_invoices_per_quarter = df_top_four_quarters.groupby('Quarter')['InvoiceNo'].nunique()

# Process transactions and calculate metrics for each quarter
for i, quarter in enumerate(total_invoices_per_quarter.index):
    quarter_df = df_top_four_quarters[df_top_four_quarters['Quarter'] == quarter]
    top_items_metrics = calculate_top_item_metrics(quarter_df, total_invoices_per_quarter, 'Quarter')
    
    print(f"Metrics for the top 20 items in Quarter {i+1} ({quarter}):")
    for item, total_receipts, total_sales_quantity, total_sales_value, support in top_items_metrics:
        print(f"Item: {item}")
        print(f"Total Receipts: {total_receipts}")
        print(f"Total Sales Quantity: {total_sales_quantity}")
        print(f"Total Sales Value: {total_sales_value:.2f}")
        print(f"Support: {support:.4f}")
        print()

#%%

def process_and_visualize_top_items(df, total_invoices_per_period, period_name):
    support_values = []

    # Process transactions and calculate metrics for each period
    for i, period in enumerate(total_invoices_per_period.index):
        period_df = df[df[period_name] == period]
        top_items_metrics = calculate_top_item_metrics(period_df, total_invoices_per_period, period_name)

        # Sort items by support value in descending order
        top_items_metrics.sort(key=lambda x: x[4], reverse=True)

        # Extract support values and item names
        item_names = [item for item, _, _, _, support in top_items_metrics]
        support_values_period = [support for _, _, _, _, support in top_items_metrics]
        support_values.append(support_values_period)

        # Visualization
        plt.figure(figsize=(10, 6))

        # Plotting in descending order
        plt.barh(item_names[::-1], support_values_period[::-1])

        plt.xlabel("Support")
        plt.ylabel("Item Name")
        plt.title(f"Support of Top 20 Items in {period_name.capitalize()} {i+1} ({period})")
        plt.tight_layout()
        plt.show()
        
# Process and visualize top items for each quarter
process_and_visualize_top_items(df_top_four_quarters, total_invoices_per_quarter, 'Quarter')

#%%

# Calculate the total number of invoices for each month
total_invoices_per_month = df_top_three_months.groupby('Month')['InvoiceNo'].nunique()

# Process transactions and calculate metrics for each month
for i, month in enumerate(total_invoices_per_month.index):
    month_df = df_top_three_months[df_top_three_months['Month'] == month]
    top_items_metrics = calculate_top_item_metrics(month_df, total_invoices_per_month, 'Month')
    
    print(f"Metrics for the top 20 items in Month {i+1} ({month}):")
    for item, total_receipts, total_sales_quantity, total_sales_value, support in top_items_metrics:
        print(f"Item: {item}")
        print(f"Total Receipts: {total_receipts}")
        print(f"Total Sales Quantity: {total_sales_quantity}")
        print(f"Total Sales Value: {total_sales_value:.2f}")
        print(f"Support: {support:.4f}")
        print()

# In[7]:

# Process and visualize top items for each month
process_and_visualize_top_items(df_top_three_months, total_invoices_per_month, 'Month')

# In[9]:

# Calculate the total number of invoices for each biweek
total_invoices_per_biweek = df_top_three_months.groupby('Biweek')['InvoiceNo'].nunique()

# Process transactions and calculate metrics for each biweek
for i, biweek in enumerate(sorted_biweeks):
    biweek_df = df_top_three_months[df_top_three_months['Biweek'] == biweek]
    top_items_metrics = calculate_top_item_metrics(biweek_df, total_invoices_per_biweek, 'Biweek')
    
    print(f"Metrics for the top 20 items in Biweek {i+1} ({biweek}):")
    for item, total_receipts, total_sales_quantity, total_sales_value, support in top_items_metrics:
        print(f"Item: {item}")
        print(f"Total Receipts: {total_receipts}")
        print(f"Total Sales Quantity: {total_sales_quantity}")
        print(f"Total Sales Value: {total_sales_value:.2f}")
        print(f"Support: {support:.4f}")
        print()

# In[10]:

# Process transactions and calculate metrics for each biweek
for i, biweek in enumerate(sorted_biweeks):
    biweek_df = df_top_three_months[df_top_three_months['Biweek'] == biweek]
    top_items_metrics = calculate_top_item_metrics(biweek_df, total_invoices_per_biweek, 'Biweek')
    
    # Sort items by support value in descending order
    top_items_metrics.sort(key=lambda x: x[4], reverse=True)
    
    # Extract support values and item names
    item_names = [item for item, _, _, _, support in top_items_metrics]
    support_values_biweek = [support for _, _, _, _, support in top_items_metrics]
    
    # Visualization
    plt.figure(figsize=(10, 6))
    
    # Plotting in descending order
    plt.barh(item_names[::-1], support_values_biweek[::-1])  
    
    plt.xlabel("Support")
    plt.ylabel("Item Name")
    plt.title(f"Support of Top 20 Items in Biweek {i+1} ({biweek})")
    plt.tight_layout()
    plt.show()
    

# In[11]:

def perform_one_hot_encoding(transaction_list, remove_lower=False):
    
    # Convert transaction items to strings
    transaction_list_str = [[str(item) for item in sublist] for sublist in transaction_list]
    
    # Perform one-hot encoding
    encoder = TransactionEncoder()
    one_hot_encoded = encoder.fit_transform(transaction_list_str)
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.columns_)
    
    # Optionally remove columns containing lowercase characters
    if remove_lower:
        columns_to_remove = [col for col in one_hot_encoded_df.columns if any(c.islower() for c in col)]
        one_hot_encoded_df = one_hot_encoded_df.drop(columns=columns_to_remove)
    
    return one_hot_encoded_df

# Define the list of transaction sets
transaction_sets = [quarter1, quarter2, quarter3, quarter4, month1, month2, month3, biweek1, biweek2, biweek3, biweek4, biweek5, biweek6]

one_hot_encoded_dfs = []

# Perform one-hot encoding for each transaction set and store the resulting DataFrame
for i, transactions in enumerate(transaction_sets, start=1):
    one_hot_encoded_dfs.append(perform_one_hot_encoding(transactions, remove_lower=True))
    globals()[f'one_hot_encoded_df{i}'] = one_hot_encoded_dfs[-1]

# In[12]:

min_support = 0.02
min_threshold = 1

def generate_association_rules(one_hot_encoded_df, min_support=0.02, min_threshold=1):
    
    # Generate frequent itemsets
    frequent_itemsets = apriori(one_hot_encoded_df, min_support=min_support, use_colnames=True)
    
    # Check if frequent itemsets DataFrame is empty
    if frequent_itemsets.empty:
        raise ValueError("Frequent itemsets DataFrame is empty.")
        
    # Generate association rules using the frequent itemsets
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
    return rules

#%%

def process_transactions_and_save_results(transaction_sets, prefix, total_invoices, min_support, min_threshold):
    rules_list = []
    excel_filename = f"{prefix}_association_rules_results.xlsx"
    excel_writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')
    
    for idx, transactions in enumerate(transaction_sets, start=1):
        
        if prefix == "Biweek" and idx == 6:
            min_support = 0.0172
        
        # Perform one-hot encoding
        one_hot_encoded_df = perform_one_hot_encoding(transactions, remove_lower=True)
        
        # Generate association rules
        rules = generate_association_rules(one_hot_encoded_df, min_support, min_threshold)
        rules_list.append(rules)
        
        results = []
        for i, rule in rules.sort_values(["lift", "support", "confidence"], ascending=False).head(20).iterrows():
            antecedents = rule['antecedents']
            consequents = rule['consequents']
            support = rule['support']
            confidence = rule['confidence']
            lift = rule['lift']
            
            # Filter DataFrame based on period type
            if prefix == 'Quarter':
                filtered_df = df_top_four_quarters[((df_top_four_quarters['Description'].isin(antecedents)) | (df_top_four_quarters['Description'].isin(consequents))) & (df_top_four_quarters['Quarter'] == total_invoices.index[idx-1])]
                
            elif prefix == 'Month':
                filtered_df = df_top_three_months[((df_top_three_months['Description'].isin(antecedents)) | (df_top_three_months['Description'].isin(consequents))) & (df_top_three_months['Month'] == total_invoices.index[idx-1])]    
            
            else:
                filtered_df = df_top_three_months[((df_top_three_months['Description'].isin(antecedents)) | (df_top_three_months['Description'].isin(consequents))) & (df_top_three_months['Biweek'] == total_invoices.index[idx-1])]  
            
            total_receipts = len(filtered_df)
            total_sales_quantity = filtered_df['Quantity'].sum()
            total_sales_value = (filtered_df['Quantity'] * filtered_df['UnitPrice']).sum()
            
            results.append([f"Association Rule {i + 1}", antecedents, consequents, support, confidence, lift, total_receipts, total_sales_quantity, total_sales_value])
        
        result_df = pd.DataFrame(results, columns=["Association Rule", "Antecedents", "Consequents", "Support", "Confidence", "Lift", "Total Receipts", "Total Sales Quantity", "Total Sales Value"])
        sheet_name = f"{prefix}_{idx}_Results"
        result_df.to_excel(excel_writer, sheet_name, index=False)
        
        print(f"Top 20 Association Rules for {prefix} {idx}:")
        for i, rule in result_df.iterrows():
            print(f"Association Rule {i + 1}:")
            print(f"Antecedents: {rule['Antecedents']}")
            print(f"Consequents: {rule['Consequents']}")
            print(f"Support: {rule['Support']:.4f}")
            print(f"Confidence: {rule['Confidence']:.4f}")
            print(f"Lift: {rule['Lift']:.4f}")
            print(f"Total Receipts: {rule['Total Receipts']}")
            print(f"Total Sales Quantity: {rule['Total Sales Quantity']}")
            print(f"Total Sales Value: {rule['Total Sales Value']:.2f}")
            print()
    
    excel_writer.save()
    
    # Return list of dataframe containing the generated rules
    return rules_list


#%%

# Process and save results for quarters
transaction_sets_quarter = [quarter1, quarter2, quarter3, quarter4]
rules_list_quarter = process_transactions_and_save_results(transaction_sets_quarter, 'Quarter', total_invoices_per_quarter, min_support, min_threshold) 


#%%

# Get metrics for a specific association rule in a given period
def get_rule_metrics(rules, df, antecedents_of_interest, consequents_of_interest, period_idx, period_type):
    
    # Define mappings for periods
    period_mapping_quarter = {1: '2011Q1', 2: '2011Q2', 3: '2011Q3', 4: '2011Q4'}
    period_mapping_month = {1: '2011-10', 2: '2011-11', 3: '2011-12'}
    period_mapping_biweek = {1: '9/24/2011 - 10/7/2011', 2: '10/8/2011 - 10/21/2011', 
                             3: '10/22/2011 - 11/4/2011', 4: '11/5/2011 - 11/18/2011',
                             5: '11/19/2011 - 12/2/2011', 6: '12/3/2011 - 12/15/2011'}

    # Determine the period value based on the period type and index
    if period_type == "quarter":
        period_value = pd.Period(period_mapping_quarter[period_idx], freq='Q')
    elif period_type == "month":
        period_value = pd.Period(period_mapping_month[period_idx], freq='M')
    elif period_type == "biweek":
        period_value = period_mapping_biweek[period_idx]
    
    # Filter rules for the specified antecedents and consequents
    filtered_rules = rules[(rules['antecedents'] == antecedents_of_interest) & 
                           (rules['consequents'] == consequents_of_interest)]
    
    # Check if a valid rule exists and retrieve metrics
    if len(filtered_rules) == 1 and not filtered_rules[['support', 'confidence', 'lift']].isnull().values.any():
        support = round(filtered_rules.iloc[0]['support'], 4)
        confidence = round(filtered_rules.iloc[0]['confidence'], 4)
        lift = round(filtered_rules.iloc[0]['lift'], 4)
        
        # Filter transaction data for the specified period
        if period_type == "quarter":
            filtered_df = df_top_four_quarters[(df_top_four_quarters['Description'].isin(antecedents_of_interest) | df_top_four_quarters['Description'].isin(consequents_of_interest)) & (df_top_four_quarters['Quarter'] == period_value)]
        elif period_type == "month":
            filtered_df = df_top_three_months[(df_top_three_months['Description'].isin(antecedents_of_interest) | df_top_three_months['Description'].isin(consequents_of_interest)) & (df_top_three_months['Month'] == period_value)]
        elif period_type == "biweek":
            filtered_df = df_top_three_months[(df_top_three_months['Description'].isin(antecedents_of_interest) | df_top_three_months['Description'].isin(consequents_of_interest)) & (df_top_three_months['Biweek'] == period_value)]
        
        # Calculate total sales value for the filtered transactions
        total_sales_value = round((filtered_df['Quantity'] * filtered_df['UnitPrice']).sum(), 2)
        
        return support, confidence, lift, total_sales_value
    else:
        return None
        

#%%

# Quarterly
print("Metrics for rules in Quarter 3:")
print()

# Define the list of one-hot encoded DataFrames for each quarter
one_hot_encoded_dfs_quarter = [one_hot_encoded_df1, one_hot_encoded_df2, one_hot_encoded_df3, one_hot_encoded_df4]

# Process transactions and save results for each quarter
rules_list_quarter2 = process_transactions_and_save_results(transaction_sets_quarter, 'Quarter', total_invoices_per_quarter, 0.013, min_threshold) 

# Retrieve rules for Quarter 3
rules_quarter = rules_list_quarter2[3]
sorted_rules = rules_quarter.sort_values(["lift", "support", "confidence"], ascending=False)

# Initialize lists to store metrics for each quarter
supports_quarters = [[] for _ in range(4)]
lifts_quarters = [[] for _ in range(4)]
confidences_quarters = [[] for _ in range(4)]
sales_value_quarters = [[] for _ in range(4)]

i = 0
rule_count = 0
valid_rules_quarter = []

# Iterate through the sorted rules
while i < len(sorted_rules):
    antecedents = sorted_rules.iloc[i]['antecedents']
    consequents = sorted_rules.iloc[i]['consequents']
    
    support_list = []
    lift_list = []
    confidence_list = []
    sales_value_list = []
    
    valid_rule = True

    # Iterate through the quarters to calculate metrics for each rule
    for idx, rules in enumerate(rules_list_quarter2, start=1):
        
        # Get metrics for the current rule and quarter
        metrics = get_rule_metrics(rules, one_hot_encoded_dfs_quarter, antecedents, consequents, idx, 'quarter')
        
        if metrics is not None:
            support, confidence, lift, sales_value = metrics
            support_list.append(support)
            lift_list.append(lift)
            confidence_list.append(confidence)
            sales_value_list.append(sales_value)
            
        else:
            valid_rule = False  # Rule is not valid if any metric is missing
            break  # Break the loop if the rule is not valid

    if valid_rule:
        rule_count += 1
        if rule_count <= 20:
            print(f"Rule {i + 1}:")
            print("Antecedent:", antecedents)
            print("Consequent:", consequents)
            print("Support in four quarters:", ", ".join(str(x) for x in support_list))
            print("Lift in four quarters:", ", ".join(str(x) for x in lift_list))
            print("Confidence in four quarters:", ", ".join(str(x) for x in confidence_list))
            print("Sales value in four quarters:", ", ".join(str(x) for x in sales_value_list))
            print()
        
            # Append metrics to the corresponding quarter lists
            for idx in range(4):
                supports_quarters[idx].append(support_list[idx])
                lifts_quarters[idx].append(lift_list[idx])
                confidences_quarters[idx].append(confidence_list[idx])
                sales_value_quarters[idx].append(sales_value_list[idx])
            
        # Store rule information in a dictionary
        rule_info = {
            'antecedent': antecedents,
            'consequent': consequents,
            'support': support_list,
            'lift': lift_list,
            'confidence': confidence_list,
            'sales_value': sales_value_list
        }
        
        valid_rules_quarter.append(rule_info)
        print("Adding rules to the list...")
        print()
        
    i += 1
    
#%%

# Print the valid rules list
print(len(valid_rules_quarter))

#%%

def plot_rules(fig_title, supports, confidences, lifts, sales_values, period_names):
    plt.figure(figsize=(14, 16))

    # Plotting support for odd-numbered rules
    plt.subplot(4, 1, 1)
    for idx, period_name in enumerate(period_names, start=1):
        plt.plot(range(1, 21, 2), supports[idx - 1][::2], label=period_name)  # Adjusted range to include only odd numbers
    plt.title(f'Support of Top 20 Rules across {fig_title}')
    plt.xlabel('Rules')
    plt.xticks(range(1, 21, 2))
    plt.legend()

    # Plotting lift for odd-numbered rules
    plt.subplot(4, 1, 2)
    for idx, period_name in enumerate(period_names, start=1):
        plt.plot(range(1, 21), confidences[idx - 1], label=period_name)  # Adjusted range to include only odd numbers
    plt.title(f'Confidence of Top 20 Rules across {fig_title}')
    plt.xlabel('Rules')
    plt.xticks(range(1, 21))
    plt.legend()

    # Plotting confidence for odd-numbered rules
    plt.subplot(4, 1, 3)
    for idx, period_name in enumerate(period_names, start=1):
        plt.plot(range(1, 21, 2), lifts[idx - 1][::2], label=period_name)  # Adjusted range to include only odd numbers
    plt.title(f'Lift of Top 20 Rules across {fig_title}')
    plt.xlabel('Rules')
    plt.xticks(range(1, 21, 2))
    plt.legend()

    plt.subplot(4, 1, 4)
    for idx, period_name in enumerate(period_names, start=1):
        plt.plot(range(1, 21, 2), sales_values[idx - 1][::2], label=period_name)  # Adjusted range to include only odd numbers
    plt.title(f'Sales value of Top 20 Rules across {fig_title}')
    plt.xlabel('Rules')
    plt.xticks(range(1, 21, 2))
    plt.legend()

    plt.tight_layout()
    plt.show()

#%%

plot_rules('Four Quarters', supports_quarters, confidences_quarters, lifts_quarters, sales_value_quarters, ['Quarter 1', 'Quarter 2', 'Quarter 3', 'Quarter 4'])

#%%

# Process and save results for months
transaction_sets_month = [month1, month2, month3]
rules_list_month = process_transactions_and_save_results(transaction_sets_month, 'Month', total_invoices_per_month, min_support, min_threshold)

#%%

# Monthly
print("Metrics for rules in Month 1:")
print()

one_hot_encoded_dfs_month = [one_hot_encoded_df5, one_hot_encoded_df6, one_hot_encoded_df7]

rules_list_month2 = process_transactions_and_save_results(transaction_sets_month, 'Month', total_invoices_per_month, 0.014, min_threshold) 

rules_month = rules_list_month2[2]
sorted_rules = rules_month.sort_values(["lift", "support", "confidence"], ascending=False)

supports_months = [[] for _ in range(3)]
lifts_months = [[] for _ in range(3)]
confidences_months = [[] for _ in range(3)]
sales_value_months = [[] for _ in range(3)]

i = 0
rule_count = 0
valid_rules_month = []

cvc_list = []
cvl_list = []
ocvr_list = []

while i < len(sorted_rules):
    antecedents = sorted_rules.iloc[i]['antecedents']
    consequents = sorted_rules.iloc[i]['consequents']
    
    support_list = []
    lift_list = []
    confidence_list = []
    sales_value_list = []
    
    valid_rule = True

    for idx, rules in enumerate(rules_list_month2, start=1):
        
        metrics = get_rule_metrics(rules, one_hot_encoded_dfs_month, antecedents, consequents, idx, 'month')
        
        if metrics is not None:
            support, confidence, lift, sales_value = metrics
            support_list.append(support)
            lift_list.append(lift)
            confidence_list.append(confidence)
            sales_value_list.append(sales_value)
        else:
            valid_rule = False  # Rule is not valid if any metric is missing
            break  # Break the loop if the rule is not valid

    if valid_rule:
        rule_count += 1
        if rule_count <= 20:
            print(f"Rule {i + 1}:")
            print("Antecedent:", antecedents)
            print("Consequent:", consequents)
            print("Support in three months:", ", ".join(str(x) for x in support_list))
            print("Lift in three months:", ", ".join(str(x) for x in lift_list))
            print("Confidence in three months:", ", ".join(str(x) for x in confidence_list))
            print("Sales value in three months:", ", ".join(str(x) for x in sales_value_list))
            print()
             
            
            for idx in range(3):
                supports_months[idx].append(support_list[idx])
                lifts_months[idx].append(lift_list[idx])
                confidences_months[idx].append(confidence_list[idx])
                sales_value_months[idx].append(sales_value_list[idx])
            
        # Store rule information in a dictionary
        rule_info = {
            'antecedent': antecedents,
            'consequent': consequents,
            'support': support_list,
            'lift': lift_list,
            'confidence': confidence_list,
            'sales_value': sales_value_list
        }
        
        valid_rules_month.append(rule_info)
        
        print("Adding rules to the list...")
        print()
        
        # Calculate CVL
        mean_lift = sum(lift_list) / len(lift_list)
        std_lift = (sum((x - mean_lift) ** 2 for x in lift_list) / len(lift_list)) ** 0.5
        cvl = (std_lift / mean_lift) * 100

        # Calculate CVC
        mean_confidence = sum(confidence_list) / len(confidence_list)
        std_confidence = (sum((x - mean_confidence) ** 2 for x in confidence_list) / len(confidence_list)) ** 0.5
        cvc = (std_confidence / mean_confidence) * 100

        # Calculate OCVR
        ocvr = (cvl + cvc) / 2

        # Round the metrics to 4 decimal places
        cvl = round(cvl, 4)
        cvc = round(cvc, 4)
        ocvr = round(ocvr, 4)

        cvc_list.append(cvc)
        cvl_list.append(cvl)
        ocvr_list.append(ocvr)
            
    i += 1
    
    
#%%

print(len(valid_rules_month))    
    
#%%

plot_rules('Three Months', supports_months, confidences_months, lifts_months, sales_value_months, ['October', 'November', 'December'])

#%%

# Print CVL, CVC, OCVR, Antecedent, and Consequent for the first 20 valid rules
print("Metrics for first 20 valid rules:")

for i in range(20):
    if i < len(valid_rules_month):
        rule = valid_rules_month[i]
        antecedent = rule['antecedent']
        consequent = rule['consequent']
        cvl = cvl_list[i]
        cvc = cvc_list[i]
        ocvr = ocvr_list[i]
        print(f"Rule {i+1}: ")
        print(f"Antecedent: {antecedent}")
        print(f"Consequent: {consequent}")
        print(f"CVL:   {cvl:.2f}%   CVC:   {cvc:.2f}%   OCVR:   {ocvr:.2f}%")
    else:
        print(f"Rule {i + 1}:   No valid rule found")


#%%

print("Metrics for all valid rules:")

for i, (ocvr, cvl, cvc) in enumerate(zip(ocvr_list, cvl_list, cvc_list), start=1):
    print(f"Rule {i:3}:   CVL: {cvl:6.2f}%   CVC: {cvc:6.2f}%   OCVR: {ocvr:6.2f}%")
    
#%%


# Define the ocvr intervals
intervals = [(1, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
ocvr_intervals = [f"{start}-{end}%" for start, end in intervals]

# Categorize the rules into ocvr intervals
ocvr_counts = [0] * len(intervals)
for ocvr in ocvr_list:
    for i, (start, end) in enumerate(intervals):
        if start <= ocvr < end:
            ocvr_counts[i] += 1
            break

# Visualize the counts using a vertical bar graph
plt.figure(figsize=(10, 6))
plt.bar(ocvr_intervals, ocvr_counts, color='skyblue')
plt.xlabel('OCVR')
plt.ylabel('Number of Rules')
plt.title('Number of Rules in Each OCVR Interval')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
    


# In[22]:

custom_order = ['9/24/2011 - 10/7/2011', '10/8/2011 - 10/21/2011', '10/22/2011 - 11/4/2011',
                '11/5/2011 - 11/18/2011', '11/19/2011 - 12/2/2011', '12/3/2011 - 12/15/2011']

# Create a new series with the custom order
rearranged_total_invoices_per_biweek = total_invoices_per_biweek.reindex(custom_order)   

# Process and save results for months
transaction_sets_biweek = [biweek1, biweek2, biweek3, biweek4, biweek5, biweek6]
rules_list_biweek = process_transactions_and_save_results(transaction_sets_biweek, 'Biweek', rearranged_total_invoices_per_biweek, min_support, min_threshold)

#%%

# Monthly
print("Metrics for rules in Biweek 3:")
print()

one_hot_encoded_dfs_biweek = [one_hot_encoded_df8, one_hot_encoded_df9, one_hot_encoded_df10, one_hot_encoded_df11, one_hot_encoded_df12, one_hot_encoded_df13]

rules_list_biweek2 = process_transactions_and_save_results(transaction_sets_biweek, 'Biweek', rearranged_total_invoices_per_biweek, 0.0165, min_threshold)

rules_biweek = rules_list_biweek2[0]
sorted_rules = rules_biweek.sort_values(["lift", "support", "confidence"], ascending=False)

supports_biweeks = [[] for _ in range(6)]
lifts_biweeks = [[] for _ in range(6)]
confidences_biweeks = [[] for _ in range(6)]
sales_value_biweeks = [[] for _ in range(6)]

i = 0
rule_count = 0
valid_rules_biweek = []

while i < len(sorted_rules):
    antecedents = sorted_rules.iloc[i]['antecedents']
    consequents = sorted_rules.iloc[i]['consequents']
    
    support_list = []
    lift_list = []
    confidence_list = []
    sales_value_list = []
    
    valid_rule = True

    for idx, rules in enumerate(rules_list_biweek2, start=1):
        
        metrics = get_rule_metrics(rules, one_hot_encoded_dfs_biweek, antecedents, consequents, idx, 'biweek')
        
        if metrics is not None:
            support, confidence, lift, sales_value = metrics
            support_list.append(support)
            lift_list.append(lift)
            confidence_list.append(confidence)
            sales_value_list.append(sales_value)
        else:
            valid_rule = False  # Rule is not valid if any metric is missing
            print("Rule passed...")
            print()
            break  # Break the loop if the rule is not valid

    if valid_rule:
        rule_count += 1
        if rule_count <= 20:
            print(f"Rule {i + 1}:")
            print("Antecedent:", antecedents)
            print("Consequent:", consequents)
            print("Support in six biweeks:", ", ".join(str(x) for x in support_list))
            print("Lift in six biweeks:", ", ".join(str(x) for x in lift_list))
            print("Confidence in six biweeks:", ", ".join(str(x) for x in confidence_list))
            print("Sales value in six biweeks:", ", ".join(str(x) for x in sales_value_list))
            print()
            
            for idx in range(6):
                supports_biweeks[idx].append(support_list[idx])
                lifts_biweeks[idx].append(lift_list[idx])
                confidences_biweeks[idx].append(confidence_list[idx])
                sales_value_biweeks[idx].append(sales_value_list[idx])
            
        # Store rule information in a dictionary
        rule_info = {
            'antecedent': antecedents,
            'consequent': consequents,
            'support': support_list,
            'lift': lift_list,
            'confidence': confidence_list,
            'sales_value': sales_value_list
        }
        
        valid_rules_biweek.append(rule_info)
        print("Adding rules to the list...")
        print()
            
    i += 1

#%%

print(len(valid_rules_biweek))

#%%

plot_rules('Six Biweeks', supports_biweeks, confidences_biweeks, lifts_biweeks, sales_value_biweeks, ['Biweek 1', 'Biweek 2', 'Biweek 3', 'Biweek 4', 'Biweek 5', 'Biweek 6'])
    

#%%

# Initialize a dictionary to store matched rules
matched_rules = {}

# Initialize a set to keep track of saved rules
saved_rules = set()

# Iterate over the rules in the first dictionary
for rule1 in valid_rules_month:
    antecedent = rule1['antecedent']
    consequent = rule1['consequent']
    
    # Check if the same rule exists in the other two dictionaries
    for rule2 in valid_rules_biweek:
        if rule2['antecedent'] == antecedent and rule2['consequent'] == consequent:
            for rule3 in valid_rules_quarter:
                if rule3['antecedent'] == antecedent and rule3['consequent'] == consequent:
                    # Save the information of the matched rules
                    if (antecedent, consequent) not in saved_rules:
                        if antecedent not in matched_rules:
                            matched_rules[antecedent] = {}
                        if consequent not in matched_rules[antecedent]:
                            matched_rules[antecedent][consequent] = {
                                'monthly': {'support': [], 'confidence': [], 'lift': []},
                                'quarterly': {'support': [], 'confidence': [], 'lift': []},
                                'biweekly': {'support': [], 'confidence': [], 'lift': []}
                            }

                        matched_rules[antecedent][consequent]['monthly']['support'].extend(rule1['support'])
                        matched_rules[antecedent][consequent]['monthly']['confidence'].extend(rule1['confidence'])
                        matched_rules[antecedent][consequent]['monthly']['lift'].extend(rule1['lift'])

                        matched_rules[antecedent][consequent]['biweekly']['support'].extend(rule2['support'])
                        matched_rules[antecedent][consequent]['biweekly']['confidence'].extend(rule2['confidence'])
                        matched_rules[antecedent][consequent]['biweekly']['lift'].extend(rule2['lift'])

                        matched_rules[antecedent][consequent]['quarterly']['support'].extend(rule3['support'])
                        matched_rules[antecedent][consequent]['quarterly']['confidence'].extend(rule3['confidence'])
                        matched_rules[antecedent][consequent]['quarterly']['lift'].extend(rule3['lift'])

                        saved_rules.add((antecedent, consequent))

# Calculate CVL, CVC, and OCVR for each period type for each matched rule
# Print the matched rules
# Initialize lists to store data for plotting
rules = []
rule_labels = []
ocvr_values_quarterly = []
ocvr_values_monthly = []
ocvr_values_biweekly = []

# Calculate CVL, CVC, and OCVR for each period type for each matched rule
for idx, (antecedent, consequent_dict) in enumerate(matched_rules.items(), start=1):
    for consequent, metrics_dict in consequent_dict.items():
        cvc_values = []
        cvl_values = []
        ocvr_values = []

        for period, period_metrics in metrics_dict.items():
            if len(period_metrics['lift']) > 0:
                # Calculate CVL
                mean_lift = sum(period_metrics['lift']) / len(period_metrics['lift'])
                std_lift = (sum((x - mean_lift) ** 2 for x in period_metrics['lift']) / len(period_metrics['lift'])) ** 0.5
                cvl = (std_lift / mean_lift) * 100

                # Calculate CVC
                mean_confidence = sum(period_metrics['confidence']) / len(period_metrics['confidence'])
                std_confidence = (sum((x - mean_confidence) ** 2 for x in period_metrics['confidence']) / len(period_metrics['confidence'])) ** 0.5
                cvc = (std_confidence / mean_confidence) * 100

                # Calculate OCVR
                ocvr = (cvl + cvc) / 2

                # Round the metrics to 4 decimal places
                cvl = round(cvl, 4)
                cvc = round(cvc, 4)
                ocvr = round(ocvr, 4)

                cvc_values.append(cvc)
                cvl_values.append(cvl)
                ocvr_values.append(ocvr)
            else:
                cvc_values.append(0)  # Set to 0 for periods where metrics are not available
                cvl_values.append(0)  # Set to 0 for periods where metrics are not available
                ocvr_values.append(0)  # Set to 0 for periods where metrics are not available

        if len(ocvr_values) == 3:
            ocvr_values_quarterly.append(ocvr_values[0])
            ocvr_values_monthly.append(ocvr_values[1])
            ocvr_values_biweekly.append(ocvr_values[2])

            # Print rule details
            print(f"Rule {idx}:")
            print(f"Antecedent: {antecedent}")
            print(f"Consequent: {consequent}")
            print(f"CVC (Quarterly, Monthly, Biweekly): {cvc_values[0]}, {cvc_values[1]}, {cvc_values[2]}")
            print(f"CVL (Quarterly, Monthly, Biweekly): {cvl_values[0]}, {cvl_values[1]}, {cvl_values[2]}")
            print(f"OCVR (Quarterly, Monthly, Biweekly): {ocvr_values[0]}, {ocvr_values[1]}, {ocvr_values[2]}")
            print()

# Plotting
bar_width = 0.3
index = np.arange(len(ocvr_values_quarterly))
fig, ax = plt.subplots(figsize=(12, 6))

bar1 = ax.bar(index, ocvr_values_quarterly, bar_width, label='Quarterly')
bar2 = ax.bar(index + bar_width, ocvr_values_monthly, bar_width, label='Monthly')
bar3 = ax.bar(index + 2 * bar_width, ocvr_values_biweekly, bar_width, label='Biweekly')

ax.set_xlabel('Rule Number')
ax.set_ylabel('OCVR (%)')
ax.set_title('OCVR for Matched Rules')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(range(1, len(ocvr_values_quarterly) + 1))
ax.legend()

plt.tight_layout()
plt.show()








