import os #paths to file
import numpy as np # linear algebra
import pandas as pd # data processing
import warnings# warning filter
import scipy as sp #pivot egineering


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
rating_path = './rating.csv'
anime_path = './anime.csv'

rating_df = pd.read_csv(rating_path)
anime_df = pd.read_csv(anime_path)
print(anime_df.head())

