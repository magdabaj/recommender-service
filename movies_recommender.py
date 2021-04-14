import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

EMBEDDING_SIZE = 50


def count_basic_data_information(df):
    user_ids = df["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    movie_ids = df["movieId"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
    df["user"] = df["userId"].map(user2user_encoded)
    df["movie"] = df["movieId"].map(movie2movie_encoded)

    num_users = len(user2user_encoded)
    num_movies = len(movie_encoded2movie)
    df["rating"] = df["value"].values.astype(np.float32)
    # min and max ratings will be used to normalize the ratings later
    min_rating = min(df["rating"])
    max_rating = max(df["rating"])

    print(
        "Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}".format(
            num_users, num_movies, min_rating, max_rating
        )
    )
    return num_users, num_movies, min_rating, max_rating


def split_df(df, min_rating, max_rating):
    df = df.sample(frac=1, random_state=42)
    print('df', df)
    x = df[["user", "movie"]].values
    print('x', x)
    # Normalize the targets between 0 and 1. Makes it easy to train.
    y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    print('y', y)
    # Assuming training on 90% of the data and validating on 10%.
    train_indices = int(0.9 * df.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:],
    )
    return x_train, x_val, y_train, y_val


class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        # Add all the components (including bias)
        x = dot_user_movie + user_bias + movie_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)


def get_movie2movie_encoded(df):
    movie_ids = df["movieId"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    return movie2movie_encoded


def get_user2user_encoded(df):
    user_ids = df["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    return user2user_encoded


def get_movie_encoded2movie(df):
    movie_ids = df["movieId"].unique().tolist()
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
    return movie_encoded2movie


def get_recommended_movies(df, movie_df, model, user_id):
    movie2movie_encoded = get_movie2movie_encoded(df)
    user2user_encoded = get_user2user_encoded(df)
    movie_encoded2movie = get_movie_encoded2movie(df)

    # user_id = df.userId.sample(1).iloc[0]
    movies_watched_by_user = df[df.userId == user_id]
    movies_not_watched = movie_df[
        ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
    ]["movieId"]
    movies_not_watched = list(
        set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
    )
    movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
    user_encoder = user2user_encoded.get(user_id)
    user_movie_array = np.hstack(
        ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
    )
    ratings = model.predict(user_movie_array).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_movie_ids = [
        movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
    ]

    print("Showing recommendations for user: {}".format(user_id))
    print("====" * 9)
    print("Movies with high ratings from user")
    print("----" * 8)
    top_movies_user = (
        movies_watched_by_user.sort_values(by="rating", ascending=False)
            .head(5)
            .movieId.values
    )
    movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]
    for row in movie_df_rows.itertuples():
        print(row.title, ":", row.genres)

    print("----" * 8)
    print("Top 10 movie recommendations")
    print("----" * 8)
    recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]

    return recommended_movies


def recommend_movies(movies_df, ratings_df, user_id):
    num_users, num_movies, min_rating, max_rating = count_basic_data_information(ratings_df)
    x_train, x_val, y_train, y_val = split_df(ratings_df, min_rating, max_rating)
    model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001)
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        epochs=5,
        verbose=1,
        validation_data=(x_val, y_val),
    )

    # plt.plot(history.history["loss"])
    # plt.plot(history.history["val_loss"])
    # plt.title("model loss")
    # plt.ylabel("loss")
    # plt.xlabel("epoch")
    # plt.legend(["train", "test"], loc="upper left")
    # plt.show()

    recommended_movies = get_recommended_movies(ratings_df, movies_df, model, user_id)

    return recommended_movies
# df = pd.read_csv("ratings.csv")


# user_ids = df["userId"].unique().tolist()
# user2user_encoded = {x: i for i, x in enumerate(user_ids)}
# userencoded2user = {i: x for i, x in enumerate(user_ids)}
# movie_ids = df["movieId"].unique().tolist()
# movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
# movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
# df["user"] = df["userId"].map(user2user_encoded)
# df["movie"] = df["movieId"].map(movie2movie_encoded)
#
# num_users = len(user2user_encoded)
# num_movies = len(movie_encoded2movie)
# df["rating"] = df["rating"].values.astype(np.float32)
# # min and max ratings will be used to normalize the ratings later
# min_rating = min(df["rating"])
# max_rating = max(df["rating"])
#
# print(
#     "Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}".format(
#         num_users, num_movies, min_rating, max_rating
#     )
# )

# df = df.sample(frac=1, random_state=42)
# print('df', df)
# x = df[["user", "movie"]].values
# print('x', x)
# # Normalize the targets between 0 and 1. Makes it easy to train.
# y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
# print('y', y)
# # Assuming training on 90% of the data and validating on 10%.
# train_indices = int(0.9 * df.shape[0])
# x_train, x_val, y_train, y_val = (
#     x[:train_indices],
#     x[train_indices:],
#     y[:train_indices],
#     y[train_indices:],
# )


# model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
# model.compile(
#     loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001)
# )
#
# history = model.fit(
#     x=x_train,
#     y=y_train,
#     batch_size=64,
#     epochs=5,
#     verbose=1,
#     validation_data=(x_val, y_val),
# )
#
# plt.plot(history.history["loss"])
# plt.plot(history.history["val_loss"])
# plt.title("model loss")
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.legend(["train", "test"], loc="upper left")
# plt.show()
# # movie_df = pd.read_csv(movielens_dir / "movies.csv")
# movie_df = pd.read_csv("movies.csv")


# Let us get a user and see the top recommendations.
# user_id = df.userId.sample(1).iloc[0]
# movies_watched_by_user = df[df.userId == user_id]
# movies_not_watched = movie_df[
#     ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
# ]["movieId"]
# movies_not_watched = list(
#     set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
# )
# movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
# user_encoder = user2user_encoded.get(user_id)
# user_movie_array = np.hstack(
#     ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
# )
# ratings = model.predict(user_movie_array).flatten()
# top_ratings_indices = ratings.argsort()[-10:][::-1]
# recommended_movie_ids = [
#     movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
# ]
#
# print("Showing recommendations for user: {}".format(user_id))
# print("====" * 9)
# print("Movies with high ratings from user")
# print("----" * 8)
# top_movies_user = (
#     movies_watched_by_user.sort_values(by="rating", ascending=False)
#     .head(5)
#     .movieId.values
# )
# movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]
# for row in movie_df_rows.itertuples():
#     print(row.title, ":", row.genres)
#
# print("----" * 8)
# print("Top 10 movie recommendations")
# print("----" * 8)
# recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
# for row in recommended_movies.itertuples():
#     print(row.title, ":", row.genres)

# recommended_movies.to_json(r'recommended_movies.json', orient='records')
