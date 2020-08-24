import numpy as np
import pandas as pd
def getxy(df):
    x = df['x']
    x = np.expand_dims(x,axis=1)
    y = df['y']
    y = np.expand_dims(y,axis=1)

    xy = np.concatenate([x,y],axis=1)
    return  xy
def dataloader(path):
    df = pd.read_csv(path)
    return  df