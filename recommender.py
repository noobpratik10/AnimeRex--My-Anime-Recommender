import pandas as pd
import datetime
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import joblib

def popular_based_recommendation(df,used_ids=[], m=200, K=20, rec_type='top'):
    """
    Calculate the Hybrid Weighted Score for anime recommendations.
    Parameters:
    - df: Original DataFrame (unaltered).
    - m: Minimum number of scored users required.
    - K: Number of top recommendations to return.
    Returns:
    - List of anime_id for the top K recommended anime.
    """

    #Filter the dataset
    cur_year = datetime.datetime.now().year
    if rec_type == 'classic':
        # Filter for anime aired more than 10 years ago
        df = df[df['Aired'].apply(lambda x: x.isdigit() and cur_year - int(x) > 20)]
    elif rec_type == 'recent':
        # Filter for anime aired less than 5 years ago
        df = df[df['Aired'].apply(lambda x: x.isdigit() and cur_year - int(x) < 3)]
    # Exclude rows with anime_id in used_ids
    df = df[~df['anime_id'].isin(used_ids)]

    # Define weights
    weights = {
        "score": 0.4,    # Weight for Base Weighted Score
        "members": 0.2,  # Weight for Members
        "favorites": 0.1,  # Weight for Favorites
        "rank": 0.1,     # Weight for Rank
        "popularity": 0.1,  # Weight for Popularity
        "time_delay": 0.1  # Weight for Time Delay
    }

    # Make a copy of the input DataFrame to ensure no alteration
    df_copy = df.copy()

    # Compute mean score across the entire database
    C = df_copy['Score'].mean()

    # Compute the base weighted score for each anime
    v = df_copy['Scored By']
    S = df_copy['Score']
    base_weighted_score = (
        (v / (v + m).replace(0, np.nan)) * S +
        (m / (v + m).replace(0, np.nan)) * C
    ).fillna(0)

    # Normalize relevant columns
    cols_to_normalize = ['Members', 'Favorites', 'Rank', 'Popularity']
    normalized_cols = {}
    for col in cols_to_normalize:
        max_val = df_copy[col].max()
        min_val = df_copy[col].min()
        # Avoid division by zero when max_val == min_val
        if max_val != min_val:
            normalized_cols[col] = (df_copy[col] - min_val) / (max_val - min_val)
        else:
            normalized_cols[col] = np.zeros_like(df_copy[col])

    # Linear time delay
    current_year = datetime.datetime.now().year
    a = 0.1
    year_df = pd.DataFrame(df_copy['Aired'])
    year_df['Time Delay Factor'] = df_copy['Aired'].apply(
        lambda x: a * (current_year - int(x)) if pd.notnull(x) and x != 'UNKNOWN' else 0
    )

    # Calculate the hybrid weighted score
    hybrid_weighted_score = (
        weights['score'] * base_weighted_score +
        weights['members'] * normalized_cols['Members'] +
        weights['favorites'] * normalized_cols['Favorites'] -
        weights['rank'] * normalized_cols['Rank'] -
        weights['popularity'] * normalized_cols['Popularity']
    )
    if type == 'classic':
        hybrid_weighted_score += weights['time_delay'] * year_df['Time Delay Factor']
    else:
        hybrid_weighted_score -= weights['time_delay'] * year_df['Time Delay Factor']

    # Add the hybrid weighted score to the DataFrame
    df_copy['Hybrid_Weighted_Score'] = hybrid_weighted_score

    # Sort by the hybrid weighted score and return top K anime IDs
    top_k_anime_ids = df_copy.sort_values(by='Hybrid_Weighted_Score', ascending=False).head(K)['anime_id'].tolist()

    return top_k_anime_ids

def content_based_recommendation(anime_df, anime_id, k=10):
    """
    Gives content based recommendations for anime.
    Parameters:
    - anime_df: Original DataFrame (unaltered).
    - anime_id: anime_id of selected anime.
    - k: Number of top recommendations to return.
    Returns:
    - List of anime_id for the top K recommended anime.
    """

    #load the requirements
    with open('models/content_based_nn.pkl',"rb") as nn_file:
        content_based_nn = pkl.load(nn_file)
    tfidf_tags = sp.load_npz('models/tfidf_tags.npz')
    anime_index = anime_df[anime_df['anime_id'] == anime_id].index[0]

    #get the recommendations
    distances, indices = content_based_nn.kneighbors(tfidf_tags[anime_index], n_neighbors=k+1)  # Including the input anime itself
    recommended_indices = indices[0][1:]
    top_k_anime_ids = anime_df.iloc[recommended_indices]['anime_id'].values.tolist()

    return top_k_anime_ids

def colab_based_recommendation(anime_df, anime_id, k=10):
    """
    Gives collaborative filtering based recommendations for anime.
    Parameters:
    - anime_df: Original DataFrame (unaltered).
    - anime_id: anime_id of selected anime.
    - k: Number of top recommendations to return.
    Returns:
    - List of anime_id for the top K recommended anime.
    """

    #load the requirements
    sparse_rating_matrix = sp.load_npz('models/sparse_rating_matrix.npz')
    sparse_colab_based_model = joblib.load('models/sparse_colab_based_nn_model.pkl')
    with open('models/anime_index_map.pkl', 'rb') as f:
        anime_index_map = pkl.load(f)

    #get the recommendations
    if anime_id not in anime_index_map:
        return []
    anime_index = anime_index_map[anime_id]
    anime_vector = sparse_rating_matrix[anime_index]
    distances, indices = sparse_colab_based_model.kneighbors(anime_vector, n_neighbors=k + 10)
    index_to_anime_id = {v: k for k, v in anime_index_map.items()}
    top_k_anime_ids = [index_to_anime_id[idx] for idx in indices.flatten() if index_to_anime_id[idx] != anime_id][:k+10]

    # filter recommendations
    valid_anime_ids = set(anime_df['anime_id'])
    top_k_anime_ids = [anime_id for anime_id in top_k_anime_ids if anime_id in valid_anime_ids]

    return top_k_anime_ids

def hybrid_recommendation(anime_df, anime_id, k=20, content_weight=0.5, colab_weight=0.5):
    """
    Gives hybrid recommendation based content based and collaborative filtering based recommendations
    Parameters:
    - anime_id: anime_id of selected anime.
    - content_weight: weight assigned to content based recommendations.
    - colab_weight: weight assigned to collaborative recommendations.
    Returns:
    - List of anime_id for the top K recommended anime.
    """
    # Get content and collaborative filtering based recommendations
    content_recommendations = content_based_recommendation(anime_df, anime_id, k)
    colab_recommendations = colab_based_recommendation(anime_df, anime_id, k)
    recommendation_scores = {}

    # Assign scores for content-based recommendations
    for rank, rec_id in enumerate(content_recommendations, start=1):
        score = (k - rank + 1) * content_weight  # Higher rank = higher score
        if rec_id in recommendation_scores:
            recommendation_scores[rec_id] += score
        else:
            recommendation_scores[rec_id] = score

    # Assign scores for collaborative-based recommendations
    for rank, rec_id in enumerate(colab_recommendations, start=1):
        score = (k - rank + 1) * colab_weight  # Higher rank = higher score
        if rec_id in recommendation_scores:
            recommendation_scores[rec_id] += score
        else:
            recommendation_scores[rec_id] = score

    sorted_recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
    top_k_recommendations = [anime_id for anime_id, _ in sorted_recommendations[:k]]

    return top_k_recommendations
