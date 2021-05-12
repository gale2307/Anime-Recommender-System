import numpy as np
import pickle
import streamlit as st
import torch
import ncf
from ncf import NCF

@st.cache
def load_model(path):
    model = NCF(353405, 48493, 0, 0)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

@st.cache(allow_output_mutation=True)
def load_pickle(path):
    with open(path, 'rb') as handle:
        unpickled_file = pickle.load(handle)
    return unpickled_file

st.title('ARS (Anime Recommender System)')

#description
st.write('ARS is a recommender system for animes that is powered by a deep-learning neural network model that utilizes neural collaborative filtering. ' 
        + 'It is trained on reviews from the myanimelist.net website, and includes all animes that were aired during and before the Winter 2021 season (animes that are currently airing are not included and will not show up).')
st.write('To use the app, simply select all the animes you have watched below (the more the better), then press the \'Give Recommendations\' button to get your recommendations! '
        + 'The drop down list below contains around 17k animes - you can search through the list by typing an anime\'s title in the search bar. Note that the list uses Romaji titles (e.g. Shingeki no Kyojin, Kimetsu no Yaiba, etc)')

#load NCF model
model = load_model('recommender_model_3_3epochs.pt')

#load user_set_dict from pickle file
user_set_dict = load_pickle('user_set_dict.pickle')

#load anime_dict from pickle file
anime_dict = load_pickle('anime_dict.pickle')

#drop down multi-select tool for users to input animes they watched
anime_list = list(anime_dict.keys())
user_list = st.multiselect('What Animes have you watched?', anime_list, format_func = anime_dict.get)

#slider to choose how many anime recommendations the model will output
num_recommend = st.slider('Number of Recommendations', 10, 100, 10)

#button to run model
if st.button('Give Recommendations'):
    closest_user = ncf.find_closest_user(user_list, user_set_dict)
    top_recommendations = ncf.recommend_anime(model, closest_user[0], anime_list, anime_dict, user_list, num_recommend)
    st.write(top_recommendations)
    st.write('Thank you for using ARS!')

st.subheader('Links')
st.write('dataset: https://www.kaggle.com/hernan4444/anime-recommendation-database-2020')
st.write('github page: https://github.com/gale2307/Anime-Recommender-System')
st.write('model based on: https://towardsdatascience.com/deep-learning-based-recommender-systems-3d120201db7e and https://towardsdatascience.com/neural-collaborative-filtering-96cef1009401')