# Entertainment Recommendation Engine

A command-line application that helps users discover movies and music based on their preferences. It uses the OMDB and Last.fm APIs to search for content, generate recommendations, and find users with similar tastes.

---

## Features

- Search movies and music via OMDB and Last.fm APIs
- Save favourites to a local JSON database
- Get movie recommendations based on your favourite genres
- Get music recommendations based on artists you like
- Find other users with similar genre preferences
- View your profile with stats and achievements

---

## Requirements

- Python 3.10+
- An OMDB API key — get one at [omdbapi.com](https://www.omdbapi.com/apikey.aspx)
- A Last.fm API key — get one at [last.fm/api](https://www.last.fm/api/account/create)

Install dependencies:

```bash
pip install requests
```

---

## Setup

1. Clone or download this repository.

2. Open `app.py` and replace the placeholder API keys:

```python
# In moviesearch() and getMovieRecommendation()
apiKey = 'YOUR_OMDB_API_KEY'

# In musicsearch() and getMusicRecommendation()
apiKey = "YOUR_LASTFM_API_KEY"
```

3. Make sure `userDatabase.json` is in the same directory as `app.py`. If it does not exist, one will be created automatically on first run.

---

## Usage

```bash
python app.py
```

On startup, choose whether you are a new or existing user. New users are assigned a generated username that is saved to the database.

---

## Menu Options

| Option | Description |
|---|---|
| 1. Search Movies | Search by title, view details, add to favourites |
| 2. Search Music | Search by artist or song, add to favourites |
| 3. Manage Favorites | View or remove saved movies and music |
| 4. Get Movie Recommendations | Suggests movies based on your favourite genres |
| 5. Get Music Recommendations | Suggests tracks from artists similar to ones you like |
| 6. View Profile | See your stats, genre breakdown, and achievements |
| 7. Find Similar Users | Find users with overlapping genre preferences |
| 8. Exit | Save and quit |

---

## Project Structure

```
entertainment-engine/
│
├── app.py                # Main application
├── userDatabase.json     # Local user data storage
├── test.py               # Scratch/test file
└── README.md
```

---

## Data Storage

All user data is stored locally in `userDatabase.json`. Each user profile contains their name, join date, favourite movies, favourite music, preferred genres, recommendations history, and similar users.
