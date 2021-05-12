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

#load NCF model
model = load_model("recommender_model_3_3epochs.pt")

#load user_set_dict from pickle file
user_set_dict = load_pickle('user_set_dict.pickle')

#load anime_dict from pickle file
anime_dict = load_pickle('anime_dict.pickle')

#drop down multi-select tool for users to input animes they watched
anime_list = list(anime_dict.keys())
user_list = st.multiselect('What Animes have you watched?', anime_list, format_func = anime_dict.get)

'Your Anime list: ', user_list

#slider to choose how many anime recommendations the model will output
num_recommend = st.slider("Number of Anime Recommendations", 10, 100, 10)

#button to run model
if st.button("Give Anime Recommendation"):
    closest_user = ncf.find_closest_user(user_list, user_set_dict)
    top_recommendations = ncf.recommend_anime(model, closest_user[0], anime_list, anime_dict, user_list, num_recommend)
    st.write(top_recommendations)