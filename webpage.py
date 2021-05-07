import numpy as np
import pandas as pd
import pickle
import streamlit as st
import torch
import ncf
from ncf import NCF

st.title('ARS (Anime Recommender System)')

anime_df = pd.read_csv('anime.csv')

model = NCF(353405, 48493, 0, 0)
model.load_state_dict(torch.load("recommender_model_2.pt"))
model.eval()


#load user_set_dict from pickle file
with open('user_set_dict.pickle', 'rb') as handle:
    user_set_dict = pickle.load(handle)

#load anime_dict from pickle file
with open('anime_dict.pickle', 'rb') as handle:
    anime_dict = pickle.load(handle)

anime_list = list(anime_dict.keys())

user_list = st.multiselect('What Animes have you watched?', anime_list, format_func = anime_dict.get)

'Your Anime list: ', user_list

num_recommend = st.slider("Number of Anime Recommendations", 10, 100, 10)

#add button to run model
if st.button("Give Anime Recommendation"):
    closest_user = ncf.find_closest_user(user_list, user_set_dict)
    top_recommendations = ncf.recommend_anime(model, closest_user[0], anime_list, anime_df, user_list, num_recommend)
    st.write(top_recommendations)