import numpy as np
import pandas as pd

"""
Miguel Fernandez
2021-07-12

This script reads in the Zomato Bangalore Restaurant dataset
and cleans the data for modeling purposes. 
"""

# -------------------------------------------------------------
# CREATE HELPER FUNCTIONS
def create_rating(df):
    df['rate'] = df['rate'].fillna('0')

    rating = list(df['rate'])

    rating = [i.split('/')[0] for i in rating]

    total_length = len(rating)
    for i in range(total_length):
        if (rating[i] == 'NEW' or rating[i] == '-'):
            rating[i] = '0'
    rating = [float(x) for x in rating]

    df['final_rating'] = rating
    df = df[df['final_rating'] != 0]

# -------------------------------------------------------------
# READ IN DATA
path_to_folder = r'A:\Learning\UNCW\BAN-530'

df = pd.read_csv(path_to_folder + '\data\zomato.csv')

print('Finished reading in {:,} rows'.format(len(df)))

# -------------------------------------------------------------
# CREATE SUBSET OF DATA FOR ANALYSIS
df1 = df.copy()

# drop columns
cols_to_drop = ['url', 'phone', 'listed_in(type)', 'listed_in(city)', 'reviews_list', 'menu_item']
df1.drop(cols_to_drop, axis=1, inplace=True)

# create subset for restaurant types of interest
df1 = df1.loc[(df1['rest_type'] == 'Casual Dining') | (df1['rest_type'] == 'Quick Bites')]

# -------------------------------------------------------------
# FEATURE ENGINEERING
# create rating attribute
create_rating(df1)

# create binary variable for chain
# get count of restaurants by name
rest_by_name = df1['name'].value_counts().to_frame().reset_index()
# select those restaurants where there are at least 15 locations
chains = rest_by_name.loc[rest_by_name['name'] >= 15, 'index'].unique()
# create binary variable if restaurant is chain
df1['chain'] = np.where(df1['name'].isin(chains), 1, 0)

# create other binary variables
df1['online_order_bin'] = np.where(df1['online_order'] == 'Yes', 1, 0)
df1['book_table_bin'] = np.where(df1['book_table'] == 'Yes', 1, 0)
df1['casual_dining_bin'] = np.where(df1['rest_type'] == 'Casual Dining', 1, 0)
# drop columns
df1.drop(['online_order', 'book_table', 'rest_type', 'rate'], axis=1, inplace=True)

# drop rows with missing values in cuisines
df1.dropna(subset=['cuisines'], inplace=True)

# create neighborhood binary variables
# get count of neighborhoods by name
count_of_neighborhoods = df1['location'].value_counts().to_frame().reset_index()

# select those neighborhoods where there are at least 167 locations
# 167 is the median number of restaurants in a neighborhood
noi = count_of_neighborhoods.loc[count_of_neighborhoods['location'] >= 167, 'index'].unique()

# filter data set for those neighborhoods of interest
df1 = df1[df1['location'].isin(noi)]


# loop through neighborhoods of interest
# create binary neighborhood variables
suffix = '_bin'
for hood in noi:
    x = hood + suffix
    df1[x] = np.where(df1['location'] == hood, 1, 0)


# create cuisine binary variables
df1['cuisines_lst'] = df1['cuisines'].str.split(', ').values.tolist()

# create top 30 and not top 30 lists
top_30_cuisines = df1.explode('cuisines_lst')['cuisines_lst'].value_counts().index[0:30].tolist()
not_top_30_cuisines = df1.explode('cuisines_lst')['cuisines_lst'].value_counts().index[30:].tolist()

# create dummy variables
top_dummies = pd.get_dummies(df1['cuisines_lst'].explode()) \
    .reset_index().groupby('index')[top_30_cuisines].sum().add_suffix('_bin')

other_dummies = pd.get_dummies(df1['cuisines_lst'].explode()) \
    .reset_index().groupby('index')[not_top_30_cuisines].sum() \
    .any(axis=1).astype(int).rename('other_bin')

# merge the dummy dataframes to data set
df1 = pd.concat([df1, top_dummies, other_dummies], axis=1)

# clean up column names
df1.columns = df1.columns.str.lower().str.replace(' ', '_')

# -------------------------------------------------------------
# WRITE CSV FILE
df1.to_csv('processed_data.csv', index=False)
print('Successfully processed data...')
