from urllib.parse import quote
import json
import requests
from re import sub
import random
from datetime import date
from collections import Counter


allUser = {}


def ProfileNames(x):
    snakeName = '_'.join(
        sub('([A-Z][a-z]+)', r' \1',
            sub('([A-Z]+)', r' \1',
                x.replace('-', ' '))).split()).lower()
    numberUser = str(random.randint(1000, 9999))
    UserName = snakeName+"_"+numberUser
    return UserName


def save_to_json(allUser, filename="userDatabase.json"):
    with open(filename, "w") as final:
        json.dump(allUser, final, indent=4)


def load_from_json(filename="userDatabase.json"):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}


def moviesearch(profile):
    userDict = load_from_json()
    favourite_movies = userDict[profile]['favourite_movies']
    addedMovies = []
    print("\n=== MOVIE SEARCH ===")
    apiKey = '58f78525'
    title = input("Enter movie title: ")
    url = f"http://www.omdbapi.com/?s={title}&apikey={apiKey}&type=movie"
    response = requests.get(url)
    data = response.json()
    movies = data.get("Search", [])[:3]
    print("\n🎬 Search Results:")
    movies_list = []
    for i, movie in enumerate(movies, 1):
        detail_url = f"http://www.omdbapi.com/?i={movie['imdbID']}&apikey={apiKey}&plot=short"
        detail_response = requests.get(detail_url)
        details = detail_response.json()
        title = details.get("Title", "N/A")
        year = details.get("Year", "N/A")
        genre = details.get("Genre", "N/A")
        genre_list = genre.split(",")
        rating = details.get("imdbRating", "N/A")
        runtime = details.get("Runtime", "N/A")
        plot = details.get("Plot", "N/A")[:50] + "..."
        imdbid = details.get("imdbID", "N/A")
        print(f"{i}. {title} ({year})")
        print(f"   Genre: {genre}")
        print(f"   Rating: {rating}/10 | Runtime: {runtime}")
        print(f"   Plot: {plot}\n")
        movies_list.append({
            "title": title,
            "year": year,
            "runtime": runtime,
            "genre": genre_list,
            "rating": rating,
            "imdb_id": imdbid
        })
    favMovies = input(
        "\nSelect movie to add to favorites (e.g.: 1 2 for first 2 movies) or 0 to go back: ").strip().split()

    if favMovies == ['0']:
        return

    for i in favMovies:
        selection = int(i)-1
        favourite_movies.append(movies_list[selection])
        addedMovies.append(movies_list[selection])

    for x in addedMovies:
        print(f"\n✅ '{x['title']} ({x['year']})' added to your favorites !")

    save_to_json(userDict)
    mainMenu(profile)


def musicsearch(profile):
    userDict = load_from_json()
    favourite_tracks = userDict[profile]['favourite_music']
    addedMusics = []
    print("\n=== MUSIC SEARCH ===")
    apiKey = "b3ef61aacfc1d02a1ed7770af6c8e9a5"
    query = input("Enter artist or song: ")
    url = f"http://ws.audioscrobbler.com/2.0/?method=track.search&track={query}&api_key={apiKey}&format=json&limit=3"
    response = requests.get(url)
    data = response.json()
    tracks = data["results"]["trackmatches"]["track"]
    print("\n🎵 Search Results:")
    music_list = []
    for i, track in enumerate(tracks, 1):
        detail_url = f"http://ws.audioscrobbler.com/2.0/?method=track.getInfo&artist={track['artist']}&track={track['name']}&api_key={apiKey}&format=json"
        detail_response = requests.get(detail_url)
        details = detail_response.json()["track"]
        artist = track["artist"]
        title = details.get("name", track["name"])
        album = details.get("album", {}).get("title", "N/A")
        duration = details.get("duration", "N/A")
        durationSecondTotal = int(duration)//1000
        durationMinute = durationSecondTotal//60
        durationSecond = durationSecondTotal % 60
        fDuration = f"{durationMinute}:{durationSecond}"
        listeners = details.get("listeners", "N/A")
        flisteners = f"{int(listeners):,}"
        print(f"\n{i}. {artist} - {title}")
        print(f"   Duration: {fDuration} mins")
        print(f"   Album: {album}")
        print(f"   Listeners: {flisteners}")

        music_list.append({
            "artist": artist,
            "title": title,
            "duration": fDuration,
            "album": album,
            "listeners": flisteners
        })

    favMusics = input(
        "\nSelect song to add to favorites (e.g.: 1 2 for first 2 musics) or 0 to go back:").strip().split()

    if favMusics == ['0']:
        return

    for i in favMusics:
        selection = int(i)-1
        favourite_tracks.append(music_list[selection])
        addedMusics.append(music_list[selection])

    for x in addedMusics:
        print(f"\n✅ '{x['title']}' by {x['artist']} added to your favourites!")

    save_to_json(userDict)
    mainMenu(profile)


def manageFavorites(profile):
    userDict = load_from_json()
    print("\n=== MANAGE FAVORITES ===")
    print("\nCurrent Favorites:")
    favMovies = userDict[profile]['favourite_movies']
    favMusics = userDict[profile]['favourite_music']
    FavMoviesNum = len(favMovies)
    FavMusicsNum = len(favMusics)
    print(f"Movies: {FavMoviesNum} items")
    print(f"Musics: {FavMusicsNum} items")

    print("""\n1. View Movie Favorites
2. View Music Favorites
3. Remove from Favorites
4. Back to Main Menu""")
    manageFavs = int(input("\nChoose an Option: "))
    match manageFavs:
        case 1:
            print("\n🎬 Your Favorite Movies:\n")
            for i, x in enumerate(favMovies, 1):
                genre = ','.join(x['genre'])
                print(f"{i}. {x['title']} ({x['year']}) - {genre}")

            manageFavorites(profile)

        case 2:
            print("\n🎵 Your Favorite Musics:\n")
            for j, y in enumerate(favMusics, 1):
                print(f"{j}. {y['title']} - {y['album']} - {y['artist']}")

            manageFavorites(profile)

        case 3:
            print("\nRemove favorites from:")
            print("1. Movies")
            print("2. Music")
            removeItem = int(input("\nChhose an option: "))

            match removeItem:
                case 1:
                    print("\n🎬 Your Favorite Movies:\n")
                    for i, x in enumerate(favMovies, 1):
                        genre = ','.join(x['genre'])
                        print(f"{i}. {x['title']} ({x['year']}) - {genre}")

                    removeMovie = int(input(
                        "\nEnter the number of the movie to remove: ")) - 1
                    if 0 <= removeMovie < FavMoviesNum:
                        removedMovie = favMovies.pop(removeMovie)
                        print(
                            f"\n✅ '{removedMovie['title']} ({removedMovie['year']})' removed from favorites!")
                    else:
                        print("\nInvalid selection!")
                    save_to_json(userDict)

                case 2:
                    print("🎵 Your Favorite Musics:")
                    for j, y in enumerate(favMusics, 1):
                        print(
                            f"{j}. {y['title']} - {y['album']} - {y['artist']}")

                    removeMusic = int(
                        input("\nEnter the number of the music to remove: "))-1

                    if 0 <= removeMusic < FavMusicsNum:
                        removedMusic = favMusics.pop(removeMusic)
                        print(
                            f"\n✅ '{removedMusic['title']}' by {removedMusic['artist']} removed from favorites!")
                    else:
                        print("\nInvalid selection!")

                    save_to_json(userDict)

            manageFavorites(profile)

        case 4:
            mainMenu(profile)


def getMovieRecommendation(profile):
    userDict = load_from_json()
    favourite_movies = userDict[profile]['favourite_movies']
    usedMovieRecommends = userDict[profile]['recommends']['movies']
    addedMovies = []
    print("\n=== MOVIE RECOMMENDATION FOR YOU ===")
    print("\nBased on your favorites")
    print("\n🎬 Top Recommendations:")
    fiftyMovies = []
    apiKey = '58f78525'
    user = userDict.get(profile)

    favorite_genres = []
    allimdbid = []
    for movie in user["favourite_movies"]:
        favorite_genres.extend(movie["genre"])
        allimdbid.append(movie["imdb_id"])
    allfavGenre = list(set(favorite_genres))

    for i in range(5):
        url = f"http://www.omdbapi.com/?s=movie&apikey={apiKey}&page={i+1}"
        response = requests.get(url)
        data = response.json()
        movies = data.get("Search", [])
        for i, movie in enumerate(movies, 1):
            detail_url = f"http://www.omdbapi.com/?i={movie['imdbID']}&apikey={apiKey}&plot=short"
            detail_response = requests.get(detail_url)
            details = detail_response.json()
            title = details.get("Title", "N/A")
            year = details.get("Year", "N/A")
            genre = details.get("Genre", "N/A")
            genre_list = genre.split(",")
            rating = details.get("imdbRating", "N/A")
            runtime = details.get("Runtime", "N/A")
            plot = details.get("Plot", "N/A")[:50] + "..."
            imdbid = details.get("imdbID", "N/A")

            if imdbid in allimdbid:
                continue

            if any(genre in allfavGenre for genre in genre_list):
                if (rating == "N/A"):
                    continue
                fiftyMovies.append({
                    "title": title,
                    "year": year,
                    "runtime": runtime,
                    "genre": genre_list,
                    "rating": rating,
                    "imdb_id": imdbid
                })

    sortedList = sorted(fiftyMovies, key=lambda i: i['rating'], reverse=True)

    recommendedMovies = sortedList[:5]

    for j, x in enumerate(recommendedMovies, 1):
        genre = ','.join(x['genre'])

        def pretty_list(words):
            if not words:
                return ""
            elif len(words) == 1:
                return words[0]
            elif len(words) == 2:
                return f"{words[0]} and {words[1]}"
            else:
                return ", ".join(words[:-1]) + f", and {words[-1]}"

        print(f"\n{j}. {x['title']} ({x['year']})")
        print(f"  Rating: {x['rating']} | Runtime: {x['runtime']}")
        print(f"  Why? : Because it has {pretty_list(x['genre'])} ")

    favMovies = input(
        "\nSelect movie to add to favorites (e.g.: 1 2 for first 2 movies) or 0 to go back: ").strip().split()

    if favMovies == ['0']:
        return

    for i in favMovies:
        selection = int(i)-1
        favourite_movies.append(recommendedMovies[selection])
        addedMovies.append(recommendedMovies[selection])
        usedMovieRecommends.append(recommendedMovies[selection])

    for x in addedMovies:
        print(
            f"\n✅ '{x['title']} ({x['year']})' added to your favorites !")

    save_to_json(userDict)

    mainMenu(profile)


def getMusicRecommendation(profile):
    userDict = load_from_json()
    usedMusicRecommends = userDict[profile]['recommends']['musics']
    print("\n=== MUSIC RECOMMENDATIONS FOR YOU ===")
    print("\nBased on your listening history...")
    print("\n🎵 Top Recommendations:")
    favourite_tracks = userDict[profile]['favourite_music']
    apiKey = "b3ef61aacfc1d02a1ed7770af6c8e9a5"
    artists = set(song["artist"] for song in favourite_tracks)
    recommendMusic = []
    addedMusics = []
    for x in artists:
        url = f"http://ws.audioscrobbler.com/2.0/?method=artist.getsimilar&artist={quote(x)}&api_key={apiKey}&format=json&limit=2"
        response = requests.get(url)
        data = response.json()
        tracks = data["similarartists"]["artist"]
        for x in tracks:
            artistName = x["name"]
            topSongurl = f"http://ws.audioscrobbler.com/2.0/?method=artist.gettoptracks&artist={quote(artistName)}&api_key={apiKey}&format=json&limit=5"
            topSong_response = requests.get(topSongurl)
            details = topSong_response.json()["toptracks"]["track"]
            for y in details:
                title = y["name"]
                detail_url = f"http://ws.audioscrobbler.com/2.0/?method=track.getinfo&artist={quote(artistName)}&track={quote(title)}&api_key={apiKey}&format=json"
                detail_response = requests.get(detail_url)
                details = detail_response.json()["track"]
                title = details["name"]
                album = details["album"]["title"] if "album" in details else "N/A"
                duration = details["duration"]
                durationSecondTotal = int(duration) // 1000
                durationMinute = durationSecondTotal // 60
                durationSecond = durationSecondTotal % 60
                fDuration = f"{durationMinute}:{durationSecond:02d}"
                listeners = details["listeners"]
                flisteners = f"{int(listeners):,}"
                recommendMusic.append({
                    "artist": artistName,
                    "title": title,
                    "duration": fDuration,
                    "album": album,
                    "listeners": flisteners
                })

    sorted_songs = sorted(
        recommendMusic,
        key=lambda x: int(x['listeners'].replace(',', '')),
        reverse=True
    )[:5]

    for i, x in enumerate(sorted_songs, 1):
        artist = x["artist"]
        title = x["title"]
        duration = x["duration"]
        album = x["album"]
        listeners = x["listeners"]
        print(f"\n{i}. {artist} - {title}")
        print(f"   Duration: {duration} mins")
        print(f"   Album: {album}")
        print(f"   Listeners: {listeners}")

    favMusics = input(
        "\nSelect song to add to favorites (e.g.: 1 2 for first 2 musics) or 0 to go back:").strip().split()

    if favMusics == ['0']:
        return

    for i in favMusics:
        selection = int(i) - 1
        favourite_tracks.append(sorted_songs[selection])
        addedMusics.append(sorted_songs[selection])
        usedMusicRecommends.append(sorted_songs[selection])

    for x in addedMusics:
        print(f"\n✅ '{x['title']}' by {x['artist']} added to your favourites!")

    save_to_json(userDict)

    mainMenu(profile)


def getSimilarUser(profile):
    userDict = load_from_json()
    print("\n=== FIMD SIMILAR USER ===")
    print("\nAnalysing your preferences...")
    print("\n🤝 Users with Similar Tastes:")
    setPrefferedGenre(profile)
    genre_data = {}
    for x in userDict:
        userName = x
        allUserGenre = userDict[x]["preffered_genres"]
        genre_data[userName] = allUserGenre

    my_genres = set(genre_data[profile])

    similarities = []
    for other_profile in genre_data:
        if other_profile == profile:
            continue
        other_genres = set(genre_data[other_profile])
        intersection = len(my_genres & other_genres)
        union = len(my_genres | other_genres)
        similarity = (intersection / union) * 100 if union > 0 else 0
        common_genres = list(my_genres & other_genres)
        unique_genres = list(other_genres - my_genres)
        similarities.append({
            "profile": other_profile,
            "similarity": similarity,
            "common_genres": common_genres,
            "unique_genres": unique_genres if unique_genres else ["None"]
        })

    sorted_similarities = sorted(
        similarities, key=lambda x: x["similarity"], reverse=True)[:3]

    userDict[profile]["similar_users"] = sorted_similarities
    save_to_json(userDict)

    for i, user in enumerate(sorted_similarities, 1):
        print(
            f"\n{i}. {user['profile']} ({user['similarity']:.0f}% similarity)")
        print(f"   Common genres: {', '.join(user['common_genres'])}")
        print(f"   Their unique favorites: {', '.join(user['unique_genres'])}")

    getUserRecommendation = int(
        input("\nView recommendations from similar user? (Enter number): "))-1

    selectedUser = sorted_similarities[getUserRecommendation]
    similarUser = selectedUser["profile"]
    userReccomendation = userDict[similarUser]['favourite_movies']
    currentUserMovies = userDict[profile]['favourite_movies']
    list2_ids = set(movie['imdb_id'] for movie in currentUserMovies)
    unique_movies = [
        movie for movie in userReccomendation if movie['imdb_id'] not in list2_ids][:3]
    print(f"\n🎬 {similarUser}'s Recommendations for You: \n")
    for i, movie in enumerate(unique_movies, 1):
        print(
            f"{i}. {movie['title']} ({movie['year']}) - {', '.join(movie['genre'])}")

    favMovies = input(
        "\nSelect movie to add to favorites (e.g.: 1 2 for first 2 movies) or 0 to go back: ").strip().split()

    if favMovies == ['0']:
        mainMenu(profile)
    addedMovies = []
    for i in favMovies:
        selection = int(i)-1
        currentUserMovies.append(unique_movies[selection])
        addedMovies.append(unique_movies[selection])

    for x in addedMovies:
        print(f"\n✅ '{x['title']} ({x['year']})' added to your favorites !")

    save_to_json(userDict)
    mainMenu(profile)


def setPrefferedGenre(profile):
    userDict = load_from_json()
    favMovies = userDict[profile]['favourite_movies']
    genre = []
    for movie in favMovies:
        genre.extend(genre.strip()
                     for genre in movie.get("genre", []))
    cleanGenre = list(set(genre))
    userDict[profile]["preffered_genres"] = cleanGenre
    save_to_json(userDict)


def viewProfile(profile):
    userDict = load_from_json()
    name = userDict[profile]['name']
    joinedDate = userDict[profile]['date_joined']
    usedMusicRecommends = userDict[profile]['recommends']['musics']
    usedMovieRecommends = userDict[profile]['recommends']['movies']
    similarUser = userDict[profile]['similar_users']
    recommendationUsed = len(usedMusicRecommends)+len(usedMovieRecommends)

    print("\n=== YOUR PROFILE ===")
    print(f"\n👤 User: {name}")
    print(f"📅 Member since: {joinedDate}")
    favMovies = userDict[profile]['favourite_movies']
    favMusics = userDict[profile]['favourite_music']
    genres = []
    for movie in favMovies:
        genres.extend(genre.strip() for genre in movie.get("genre", []))
    FavMoviesNum = len(favMovies)
    FavMusicsNum = len(favMusics)
    print("\n📊 Statistics:")
    print(f"Movies Favorited: {FavMoviesNum}")
    print(f"Musics Favorited: {FavMusicsNum}")
    print(f"Recommendation Used: {recommendationUsed}")
    genre_counts = Counter(genres)
    genre_dist = [
        {"genre": genre, "percentage": (count / FavMoviesNum) * 100}
        for genre, count in genre_counts.most_common(5)
    ]
    genre_dist.sort(key=lambda x: x["percentage"], reverse=True)
    print("\n🎭 Favorite Genres:")
    genre_str = ", ".join(
        f"{genre['genre']} ({genre['percentage']:.0f}%)"
        for genre in genre_dist
    )
    print(f"Movies: {genre_str}")

    print("\n🏆 Achievements:")
    if FavMoviesNum >= 10:
        print("✅ Movie Buff - 10+ favorite movies")
    else:
        print(f"🔄 Movie Buff - Add {10-FavMoviesNum} more movies in favorites")

    if FavMusicsNum >= 15:
        print("✅ Music Lover - 15+ favorite musics")
    else:
        print(f"🔄 Music Lover - Add {15-FavMusicsNum} more music in favorites")

    if recommendationUsed >= 5:
        print("✅ Explorer - Tried 5+ new recommends")
    else:
        print(
            f"🔄 Explorer - Explore {5-recommendationUsed} more recommendation")

    print("\n🤝 Similar Users:")

    for i, user in enumerate(similarUser, 1):
        print(
            f" {i}. {user['profile']} ({user['similarity']:.0f}% similarity)")

    mainMenu(profile)


def mainMenu(profile):
    print("\n=== ENTERTAINMENT RECOMMENDATION ENGINE ===")
    print("""\n1. Search Movies
2. Search Music  
3. Manage Favorites
4. Get Movie Recommendations
5. Get Music Recommendations
6. View Profile
7. Find Similar Users
8. Exit""")

    mainMenuOption = int(input("\nChoose an option:"))

    match mainMenuOption:
        case 1:
            moviesearch(profile)
        case 2:
            musicsearch(profile)
        case 3:
            manageFavorites(profile)
        case 4:
            getMovieRecommendation(profile)
        case 5:
            getMusicRecommendation(profile)
        case 6:
            viewProfile(profile)
        case 7:
            getSimilarUser(profile)
        case 8:
            print('''\nSaving your preferences...
✅ Data saved to userDatabse.json

Thanks for using the Entertainment Recommendation Engine!
Keep discovering amazing movies and music! 🎬🎵

Program terminated.
''')


def main():
    userDict = load_from_json()
    print("\n=== WELCOME ===")
    print("1. I am a new user ")
    print("2. I am an existing user ")
    welcomeUser = int(input("\nWhat are you ? : "))
    match welcomeUser:
        case 1:
            name = input("\nEnter Your Name: ")
            ProfileName = ProfileNames(name)
            print(f"Your username is : {ProfileName} ")
            today = date.today()
            ftoday = today.strftime("%d-%m-%Y")

            userDict[ProfileName] = {
                'name': name,
                'date_joined': ftoday,
                'favourite_movies': [],
                'favourite_music': [],
                'preffered_genres': [],
                "recommends": {
                    "movies": [],
                    "musics": []
                },
                "similar_users": []
            }

            save_to_json(userDict)
            print("\nUsername saved !!")

        case 2:
            ProfileName = input("\nEnter your username : ")
            if ProfileName in userDict:
                print(f"\nWelcome Back, {userDict[ProfileName]['name']} !")
                mainMenu(ProfileName)
            else:
                print("\nUser not found !")

        case _:
            print("\nPlease enter a valid choice ")


main()
