import numpy as np
import pandas as pd


def add(a,b):
    return a+b
df = pd.read_csv('data/data1.csv')

etst = df.apply(lambda row: add(row['x'],row['y']),axis=1)

print('ok')

