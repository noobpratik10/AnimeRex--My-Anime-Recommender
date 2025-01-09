# AnimeRex: My Anime Recommender

## Overview
AnimeRex is a user-friendly web application designed to help anime enthusiasts discover new shows and movies based on their preferences. Leveraging machine learning and a curated dataset, AnimeRex offers personalized recommendations to users.

## Demo
- You can try out the web app live on Streamlit Community Cloud: [AnimeRex: My Anime Recommender](https://animerex--my-anime-recommender-fj3mto2havaewsktffvtlv.streamlit.app/)

## Key Features
- *Personalized Recommendations:* Suggests anime based on user preferences such as genre, rating, and popularity.
- *Search Functionality:* Allows users to explore anime information directly.
- *Interactive Web App:* Built using Streamlit for a smooth and responsive user experience.
- *Extensive Dataset:* Data sourced from Kaggle's [MyAnimeList Dataset](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset).

## Working
The AnimeRex recommendation system combines three approaches to provide anime suggestions:
- Main Page: Weighted Score Formula
  - `Weighted Score=(v/(v+m))⋅S+(m/(v+m))⋅C`
      - `S: is the average score of the anime`
      - `v: is the number of users who rated the anime`
      - `m: is the minimum number of ratings required`
      - `C: is the average score across all anime`
  - Additional components like linear time decay factor, popularity, rank, and number of members are incorporated to refine the recommendations.
  - The time decay factor helps prioritize fresh hits, while it also positively boosts cult classics, ensuring a balanced recommendation strategy.

- Search Page: Hybrid Approach
  - *Content-Based Filtering:* Anime are tagged and vectorized using TF-IDF to find similar titles based on content.
  - *Item-Based Collaborative Filtering:* A rating matrix is used to find anime similar to those liked by users with similar preferences.
  - This hybrid approach combines both content similarity and user behavior to generate more accurate recommendations.

## Teck Stack
- *Framework:* Streamlit
- *Programming Language:* Python
- *Libraries:* Pandas, NumPy, Scikit-learn, Streamlit, NLTK, etc.
- *Frontend Enhancements:*
    *HTML:* Used via st.markdown for custom layouts.
    *CSS:* Integrated custom styles using an external stylesheet.
- *Model Training:* Built and trained models using Google Colab.

## Installation and Setup
1. Clone the repository to your local machine and navigate to the project directory
   ``` bash
     git clone https://github.com/noobpratik10/AnimeRex--My-Anime-Recommender.git
     cd animerex
   ```
2. Set up and activate Virtul Environment
   ```bash
     python -m venv env
     env\Scripts\activate
   ```
3. Install Dependecies
   ```bash
     pip install -r requirements.txt
   ```     
4. Run the WebApp
   ```bash
     streamlit run app.py
   ```



