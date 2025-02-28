import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

# Authenticate with Spotify API
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET
))

def get_top_songs(artist_name, country="PH"):
    results = sp.search(q=artist_name, type="artist", limit=1)
    if not results['artists']['items']:
        return f"No artist found for {artist_name}"

    artist_id = results['artists']['items'][0]['id']
    top_tracks = sp.artist_top_tracks(artist_id, country=country)
    
    return {
        "image": results['artists']['items'][0]['images'][0]['url'],
        "artist": results['artists']['items'][0]['name'],
		"songs": [track['name'] for track in top_tracks['tracks']],
  		"genres": results['artists']['items'][0]['genres'],
		"followers": results['artists']['items'][0]['followers']['total'],
  		"spotify_link": results['artists']['items'][0]['external_urls']['spotify']
	}


