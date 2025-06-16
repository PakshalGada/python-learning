import json
import requests


class movies:

    def __init__(self, apiKey):
        self.apiKey = apiKey

    def searchedMovie(self, title):
        try:
            url = f"http://www.omdbapi.com/?t{title}apiKey={apiKey}"
            response = requests.get(url)
            print(response)
        except Exception as e:
            print(e)


apiKey = '58f78525'
movie = movies(apiKey)
title = input("Enter Movie Title :")
movie.searchedMovie(title)
