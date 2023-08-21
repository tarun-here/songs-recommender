from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

with open("models/Songs.pkl", "rb") as songs_file:
    Songs_df1 = pickle.load(songs_file)

with open("models/selected_features_.pkl", "rb") as features_file:
    selected_features = pickle.load(features_file)
nn_model = NearestNeighbors(n_neighbors=11, metric='cosine')
nn_model.fit(Songs_df1[selected_features])

@app.route('/', methods=['GET', 'POST'])
def index():
    recommended_songs = []
    input_song = None

    if request.method == 'POST':
        input_song = request.form['song_name']
        
        if input_song:
            song_indices = Songs_df1[Songs_df1['name'] == input_song].index
            if len(song_indices) > 0:
                song_index = song_indices[0]
                input_song_features = Songs_df1[selected_features].iloc[song_index].values

                distances, indices = nn_model.kneighbors([input_song_features])

                recommended_song_indices = [idx for idx in indices[0] if idx != song_index]

                recommended_songs_data = Songs_df1.iloc[recommended_song_indices]

                for _, song in recommended_songs_data.iterrows():
                    recommended_songs.append({'name': song['name'], 'artist': song['artist'], 'img': song['img']})
    
    return render_template('index.html', recommended_songs=recommended_songs, input_song=input_song)

if __name__ == '__main__':
    app.run(debug=True)