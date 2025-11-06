import pandas as pd
import numpy as numpy
import matplotlib.pyplot as matplot
import seaborn as seaborn

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

df = pd.read_csv("data_preprocessing/data/raw.csv", header=1)
df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
df.rename(columns={'default_payment_next_month': 'default'}, inplace=True)

print(df.columns)
print(df.shape)
print(df.head())
print(df.info())


