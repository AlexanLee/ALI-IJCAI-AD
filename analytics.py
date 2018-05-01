import pandas as pd
import numpy as np

df = pd.read_csv('input/train_sample.txt', sep='\s+')

df.info()
df.describe()
df.head(1)
df.tail(1)
df.dtypes
df.columns
df.shape

df['user_age_level'].unique()

for fea in df.columns:
    df.loc[df[fea] == -1, fea] = np.nan

df.isnull().sum()

# 看每一行有多少缺省
df['n_null'] = df.isnull().sum(axis=1)

df['user_age_level'].value_counts()

df['item_brand_id'].value_counts()
df['item_city_id'].value_counts()
df['item_price_level'].value_counts()
# 7、6、8、5
df['item_sales_level'].value_counts()
# 11、10、12、9、8、13、7、14
df['item_collected_level'].value_counts()
# 12、13、11、10、14
df['item_pv_level'].value_counts()
# 17、18、16、19
df['user_gender_id'].value_counts()
# 0、1、-1、2
df['user_age_level'].value_counts()
# 1003、1004、1002、1005
df['user_occupation_id'].value_counts()
# 2005、2002、2004、2003

df['user_star_level'].value_counts()
# 3006、3003、3005、3007
df['context_page_id'].value_counts()

df['shop_review_num_level'].value_counts()
# 17、16


model_df = df.fillna(df.mode().iloc[0], inplace=True)
