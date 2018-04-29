import pandas as pd
import numpy as np

df = pd.read_csv('input/train_sample.txt', sep='\s+')

df.info()
df.describe()
df.head(1)
df.tail(1)
df.dtypes
df.columns

df['user_age_level'].unique()

for fea in df.columns:
    df.loc[df[fea] == -1, fea] = np.nan

df.isnull().sum()

# 看每一行有多少缺省
df['n_null'] = df.isnull().sum(axis=1)

df['user_age_level'].value_counts()

model_df = df.fillna(df.mode().iloc[0], inplace=True)


