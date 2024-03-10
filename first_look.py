import os  # paths to file
import numpy as np  # linear algebra
import pandas as pd  # data processing
import warnings  # warning filter
import scipy as sp  # pivot egineering
from sklearn.metrics.pairwise import cosine_similarity

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

rating_path = './rating.csv'
anime_path = './anime.csv'

rating_df = pd.read_csv(rating_path)
anime_df = pd.read_csv(anime_path)

# print(f"anime set (row, col): {anime_df.shape}\n\nrating set (row, col): {rating_df.shape}")

# print(anime_df.info())
# print(anime_df.isnull().sum().sort_values(ascending=False))
anime_df = anime_df[~np.isnan(anime_df["rating"])]
# print(anime_df.isnull().sum().sort_values(ascending=False))

rating_df['rating'] = rating_df['rating'].apply(lambda x: np.nan if x == -1 else x)

# step 1
anime_df = anime_df[anime_df['type'] == 'TV']

# step 2
rated_anime = rating_df.merge(anime_df, left_on='anime_id', right_on='anime_id', suffixes=['_user', ''])
# laczy dwia data framy w jedno
# # step 3
rated_anime = rated_anime[['user_id', 'name', 'rating']]
#
# # step 4
rated_anime_7500 = rated_anime[rated_anime.user_id <= 7500]

pivot = rated_anime_7500.pivot_table(index=['user_id'], columns=['name'], values='rating')
pivot_n = pivot.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)), axis=1)

# step 2
pivot_n.fillna(0, inplace=True)

# step 3
pivot_n = pivot_n.T

# step 4
pivot_n = pivot_n.loc[:, (pivot_n != 0).any(axis=0)]

# step 5
piv_sparse = sp.sparse.csr_matrix(pivot_n.values)
# model based on anime similarity
anime_similarity = cosine_similarity(piv_sparse)

# Df of anime similarities
ani_sim_df = pd.DataFrame(anime_similarity, index=pivot_n.index, columns=pivot_n.index)


def anime_recommendation(ani_name):
    number = 1
    print('Recommended because you watched {}:\n'.format(ani_name))
    for anime in ani_sim_df.sort_values(by=ani_name, ascending=False).index[1:6]:
        print(f'#{number}: {anime}, {round(ani_sim_df[anime][ani_name] * 100, 2)}% match')
        number += 1

anime_recommendation('Naruto')
