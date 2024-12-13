from streamlit_option_menu import option_menu
from pages import *
import streamlit as st

st.set_page_config(
    page_title="AnimeRex",
    page_icon="static/logo.png",
)

# Streamlit UI
st.image('static/Banner Girl.jpg', use_container_width=True)

#horizontal menu
selected=option_menu(
    menu_title=None,
    options=['Home','Search'],
    icons=['house','search'],
    menu_icon='cast',
    default_index=0,
    orientation='horizontal',
)

if selected == 'Home':
    display_main_page()
elif selected == 'Search':
    display_search_page()












