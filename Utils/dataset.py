import pandas as pd

def read_csv(path):
    df = pd.read_csv(path)
    return df
#check the quality
df = pd.read_csv("./data/dataset.csv")
df.isnull().sum()
df.isna()
df_clean = df.dropna()

def load_clean_dataset(path):
    df=df_clean

    return df