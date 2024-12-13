import streamlit as st
import pandas as pd
from recommender import popular_based_recommendation
from recommender import hybrid_recommendation

def display_rack(recommended_animes):
    with open('static/style.css', 'r') as file:
        st.markdown(f'<style>{file.read()}</style>',unsafe_allow_html=True)

    # Prepare the content for the rack (images + names)
    rack_content = "".join([
        f"""
        <div class="rack-item">
            <div class="rack-item-img">
                <img src="{image_url}" alt="{anime_name}" 
                onError="this.onerror=null; this.src='https://cdn.myanimelist.net/img/sp/icon/apple-touch-icon-256.png';" />    
            </div>
            <div class="rack-item-text">
                <p>{anime_name}</p>
            </div>
        </div>
        """
        for anime_name, image_url in zip(recommended_animes['Name'], recommended_animes['Image URL'])
    ])

    # Display the rack container with the anime items
    st.markdown(f"""
        <div class="rack-container">
            {rack_content}
        </div>
        """, unsafe_allow_html=True)

def display_searched_anime(selected_id, anime_df):
    with open('static/style.css', 'r') as file:
        st.markdown(f'<style>{file.read()}</style>',unsafe_allow_html=True)

    #display the searched anime
    anime_data = anime_df[anime_df['anime_id'] == selected_id].iloc[0]
    st.markdown(f"""
                <div class="anime-container">
                    <div class="anime-title">
                        <h2>{anime_data['Name']}</h2>
                    </div>
                    <div class="anime-english-title">
                        <h4>{anime_data.get('English name', 'N/A')}</h4>
                    </div>
                    <div class="anime-image-info">
                        <img src="{anime_data['Image URL']}" alt="{anime_data['Name']}" 
                            onError="this.onerror=null; this.src='https://cdn.myanimelist.net/img/sp/icon/apple-touch-icon-256.png';"  />
                        <div class="anime-details-info">
                            <p><strong>Score:</strong> {anime_data.get('Score', 'N/A')}</p>
                            <p><strong>Aired:</strong> {anime_data.get('Aired', 'N/A')}</p>
                            <p><strong>Type:</strong> {anime_data.get('Type', 'N/A')}</p>
                            <p><strong>Rating:</strong> {anime_data.get('Rating', 'N/A')}</p>
                            <p><strong>Studio:</strong> {anime_data.get('Studio', 'N/A')}</p>
                            <p><strong>Genres:</strong> {anime_data.get('Genres', 'N/A')}</p>
                        </div>
                    </div>
                    <div class="anime-synopsis">
                        <p><strong>Synopsis:</strong></p>
                        <p>{anime_data.get('Synopsis', 'N/A')}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)


def display_main_page():
    with open('static/style.css', 'r') as file:
        st.markdown(f'<style>{file.read()}</style>',unsafe_allow_html=True)

    # Load the data set
    anime_df = pd.read_csv('Animes2023.csv')

    #Display Top Recommendations
    st.title('Top Picks For You!')
    # filter the dataset
    genres_options = [
        "All", "Comedy", "Fantasy", "Action", "Adventure", "Sci-Fi", "Drama", "Romance",
        "Slice of Life", "Supernatural", "Mystery", "Avant Garde", "Ecchi", "Sports",
        "Horror", "Suspense", "Award Winning", "Boys Love", "Gourmet", "Girls Love"
    ]
    rating_options = [
        "All", "PG-13 - Teens 13 or older", "G - All Ages", "PG - Children",
        "R - 17+ (violence & profanity)", "R+ - Mild Nudity"
    ]
    type_options = [
        "All", "TV", "Movie", "OVA", "ONA", "Music", "Special"
    ]
    source_options = [
        "All", "Original", "Manga", "Novel", "Other"
    ]
    with st.container():
        st.markdown("""
            <style>
            .custom-selectbox-container .stSelectbox {
                width: 100% !important; 
            }
            </style>
        """, unsafe_allow_html=True)

        cols = st.columns([1, 1, 1, 1])

        with cols[0]:
            selected_genre = st.selectbox("Genres", options=genres_options, index=0)
        with cols[1]:
            selected_rating = st.selectbox("Rating", options=rating_options, index=0)
        with cols[2]:
            selected_type = st.selectbox("Type", options=type_options, index=0)
        with cols[3]:
            selected_source = st.selectbox("Source", options=source_options, index=0)

    filtered_df = anime_df
    if selected_rating != "All":
        filtered_df = filtered_df[filtered_df['Rating'] == selected_rating]
    if selected_type != "All":
        filtered_df = filtered_df[filtered_df['Type'] == selected_type]
    if selected_source != "All":
        filtered_df = filtered_df[filtered_df['Source'] == selected_source]
    if selected_genre != "All":
        filtered_df = filtered_df[filtered_df['Genres'].str.contains(selected_genre, case=False, na=False)]

    used_ids=[]
    top_anime_ids = popular_based_recommendation(used_ids=used_ids,df=filtered_df, K=min(filtered_df.shape[0],50), rec_type='top')
    used_ids.extend(top_anime_ids)
    recommended_animes = anime_df[anime_df['anime_id'].isin(top_anime_ids)][['Name', 'Image URL']]
    if len(recommended_animes) >= 20:
        display_rack(recommended_animes.iloc[:len(recommended_animes)//2])
        display_rack(recommended_animes.iloc[len(recommended_animes) // 2:])
    elif len(recommended_animes) > 0:
        display_rack(recommended_animes)
    else:
        st.write('No titles that matched your query were found.')

    #Display Recent Hits
    st.title('Recent Hits!')
    top_anime_ids_recent = popular_based_recommendation(used_ids=used_ids,df=anime_df, K=20, rec_type='recent')
    used_ids.extend(top_anime_ids_recent)
    recommended_animes_recent = anime_df[anime_df['anime_id'].isin(top_anime_ids_recent)][['Name', 'Image URL']]
    display_rack(recommended_animes_recent)

    #Display Cult Classics
    st.title('Cult Classics!')
    top_anime_ids_classic = popular_based_recommendation(used_ids=used_ids,df=anime_df, K=20, rec_type='classic')
    recommended_animes_classic = anime_df[anime_df['anime_id'].isin(top_anime_ids_classic)][['Name', 'Image URL']]
    used_ids.extend(top_anime_ids_classic)
    display_rack(recommended_animes_classic)




def display_search_page():
    with open('static/style.css', 'r') as file:
        st.markdown(f'<style>{file.read()}</style>',unsafe_allow_html=True)

    #Title
    st.markdown('<h1 class="responsive-title">Search Your Favourite Anime!</h1>', unsafe_allow_html=True)

    # Load the data set
    anime_df = pd.read_csv('Animes2023.csv')

    #Search the anime in select box
    options = anime_df.apply(
        lambda row: f"{row['Name']} ({row['English name']})" if row['English name'] != 'UNKNOWN' else row['Name'],
        axis=1
    ).tolist()
    searched_name = st.selectbox(
        index=None,
        label='Enter the name:',
        label_visibility='hidden',
        placeholder='Search Anime...',
        options=options
    )

    #get recommendations for searched name
    if searched_name :
        if '(' in searched_name:  # English name is included
            japanese_name = searched_name.split(' (')[0]
        else:
            japanese_name = searched_name
        selected_id = anime_df[anime_df['Name'] == japanese_name]['anime_id'].values[0]
        display_searched_anime(selected_id, anime_df)
        no_of_recommendation = st.select_slider(
            label='No. of Recommendations:',
            options=range(10, 41, 10),
            value=20
        )
        recommended_ids =hybrid_recommendation(anime_df,
                                               anime_id=selected_id,
                                               k=no_of_recommendation,
                                               content_weight=0.8,
                                               colab_weight=0.2,
                                               )

        # Display Recommendations
        st.title('Watch these next!')
        recommended_animes = anime_df[anime_df['anime_id'].isin(recommended_ids)][['Name', 'Image URL']]
        if len(recommended_animes) >= 30:
            display_rack(recommended_animes.iloc[:len(recommended_animes)//3])
            display_rack(recommended_animes.iloc[len(recommended_animes)//3:2*len(recommended_animes)//3])
            display_rack(recommended_animes.iloc[2*len(recommended_animes)//3:])
        elif len(recommended_animes) >= 20:
            display_rack(recommended_animes.iloc[:len(recommended_animes) // 2])
            display_rack(recommended_animes.iloc[len(recommended_animes) // 2:])
        elif len(recommended_animes) > 0:
            display_rack(recommended_animes)
        else:
            st.write('No titles that matched your query were found.')

