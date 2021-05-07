import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class NCF(nn.Module):
    """ Neural Collaborative Filtering (NCF)
    
        Args:
            num_users (int): Number of unique users
            num_items (int): Number of unique items
            ratings (pd.DataFrame): Dataframe containing the anime ratings for training
            all_animeIds (list): List containing all animeIds (train + test)
    """
    
    def __init__(self, num_users, num_items, ratings, all_animeIds):
        super().__init__()
        #embedding of all users, each embedding having 8 dimensions
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        #embedding of all animes, each embedding having 8 dimensions
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=16, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.ratings = ratings
        self.all_animeIds = all_animeIds
        
    def forward(self, user_input, item_input):
        
        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # Concat the two embedding layers
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        # Pass through dense layer
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        # Output layer
        pred = nn.Sigmoid()(self.output(vector))

        return pred
    
    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(AnimeLensTrainDataset(self.ratings, self.all_animeIds),
                          batch_size=512)

def find_closest_user(user_anime, user_set_dict):
    #user_anime: list of anime Ids active user interacted with
    #user_set_dict: dict of userID to their set of animes watched

    active_user_set = set(user_anime)
    
    closest_user = 0
    closest_distance = float('inf')
    for user in user_set_dict.keys():
        distance = len(active_user_set.symmetric_difference(user_set_dict[user]))
        if distance < closest_distance:
            closest_user = user
            closest_distance = distance
    return closest_user, closest_distance

def print_anime(anime_list, anime_df):
    #
    anime_name_list = list()
    for anime in anime_list:
        anime_name_list.append(anime_df.loc[anime_df['MAL_ID'] == anime]['Name'].item())
    print(anime_name_list)
    return(anime_name_list)

def recommend_anime(model, user_id, all_animeIds, anime_df, user_watched = set(), num_recommend = 10):
    #model: NCF model used for recommendation
    #user_id: integer value representing a user
    #all_animeIds: set of all anime Ids
    #user_watched: set of all anime watched by active user
    #num_recommend: number of animes returned by the function
    non_interacted = set(all_animeIds)-set(user_watched)
    non_interacted = list(non_interacted)
    predicted_labels = np.squeeze(model(torch.tensor([user_id]*len(non_interacted)), 
                                        torch.tensor(non_interacted)).detach().numpy())
    recommended_anime = [non_interacted[x] for x in np.argsort(predicted_labels)[::-1][0:num_recommend].tolist()]
    recommended_anime = print_anime(recommended_anime, anime_df)
    return recommended_anime