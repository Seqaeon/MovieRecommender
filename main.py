import streamlit as st
import random
import numpy as np
import pandas as pd
import shutil
import gc
import ast
import time
import openai
from tqdm.notebook import tqdm
import re
import os
import requests
import streamlit_analytics2
import urllib.request

from config import openai_api_key #, hf_token
from huggingface_hub import hf_hub_download

import ao_core as ao
from arch__Recommender import arch
import json
from typing import List
from bs4 import BeautifulSoup
from datasets import Dataset

import struct

st.set_page_config(
    page_title="Recommender Demo by AO Labs",
    page_icon="misc/ao_favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://discord.gg/Zg9bHPYss5",
        "Report a bug": "mailto:eng@aolabs.ai",
        "About": "AO Labs builds next-gen AI models that learn after training; learn more at docs.aolabs.ai/docs/mnist-benchmark",
    },
)

# Initialize global variables
if "videos_in_list" not in st.session_state:
    st.session_state.videos_in_list = []

if "recommendation_result" not in st.session_state:
    st.session_state.recommendation_result = []
if "current_binary_input" not in st.session_state:
    st.session_state.current_binary_input = []
if "training_history" not in st.session_state:
    st.session_state.training_history = (np.zeros([1000, 7], dtype="O"))
    st.session_state.numberVideos = 0
if "mood" not in st.session_state:
    st.session_state.mood = ["Random"]
if "type" not in st.session_state:
    st.session_state.type = ["Both"]
if "language" not in st.session_state:
    st.session_state.language = ["Any"]
if "display_video" not in st.session_state:
    st.session_state.display_video = False
if "start_year" not in st.session_state:
    st.session_state.start_year = 0
if "date_range" not in st.session_state:
    st.session_state.date_range = []
if "end_year" not in st.session_state:
    st.session_state.end_year = 0



if "start_year_options" not in st.session_state:
    st.session_state.start_year_options = []

if "end_year_options" not in st.session_state:
    st.session_state.end_year_options = []

if "natural_language_input" not in st.session_state:
    st.session_state.natural_language_input = None
if "recommened" not in st.session_state:
    st.session_state.recommended = False
if "number_videos_not_recommended" not in st.session_state:
    st.session_state.number_videos_not_recommended = 0
    st.session_state.threshold = 0


# init agent
if "agent" not in st.session_state:
    print("-------creating agent-------")
    st.session_state.agent = ao.Agent(arch, notes="Default Agent")

# intially train on random inputs
    for i in range(4):
        st.session_state.agent.reset_state()
        #st.session_state.agent.reset_state(training=True)


# Predefined list of random search terms
# random_search_terms = ['funny', 'gaming', 'science', 'technology', 'news', 'random', 'adventure', "programming", "history", "podcast", "romance", "animation", "current events"]
# random_genres = random.sample(all_genres, random.randint(1, 5))


def reduce_mem_usage(df, float16_as32=True):
    # memory_usage()
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type) != 'category':
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    if float16_as32:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))
    gc.collect()

    return df


# df = pd.read_csv("df.csv")

# df = reduce_mem_usage(df)

with open('genres_dict.json', 'r') as file:
    data_genre = json.load(file)
# all_genres = []
# genre_list = df['genres'].values
# for i in tqdm(genre_list):
#     all_genres += ast.literal_eval(i)
# all_genres = list(set(all_genres))
# Step 1: Download the CSV file from the Hugging Face Hub


@st.cache_data
def load_and_optimize_df():
    # df = pd.read_csv("df.csv")
    file_path = hf_hub_download(
        repo_id="ArchaeonSeq/MovieImdb",  # Repository ID
        filename="df.csv",                # Name of the CSV file in the repository
        repo_type="dataset",              # Type of repository (dataset)
    )

    # Step 2: Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(file_path)
    return reduce_mem_usage(df)


@st.cache_data
def process_genres(df):
    all_genres = []
    genre_list = df['genres'].values
    for i in genre_list:
        all_genres += ast.literal_eval(i)
    return list(set(all_genres))


# Use the cached functions
df = load_and_optimize_df()
all_genres = process_genres(df)
st.session_state.start_year = min(df[df['startYear'] > 0]['startYear'].values)
st.session_state.end_year =  time.localtime().tm_year


def get_plot(filter_genres, df, all_genres, type, movie_id=None, released_only=True, retries=10, openai_api_key=openai_api_key, dimensions=2, subgenres=data_genre['subgenres']):
    if retries == 0:
        raise ValueError("No released movies found after maximum retries.")
    # Set your OpenAI API key
    openai.api_key = openai_api_key
    # Generate embeddings

    def get_embeddings(texts, model="text-embedding-3-small", dimensions=dimensions):
        response = openai.embeddings.create(
            input=texts,
            model=model,
            dimensions=dimensions
        )
        # Extract embeddings from the response
        embeddings = [item.embedding for item in response.data]
        return embeddings

    def float_to_binary(f, precision="32-bit"):

        if precision == "64-bit":
            # Pack the float into 8 bytes (64-bit) using IEEE 754 format
            packed = struct.pack('!d', f)
        elif precision == "32-bit":
            packed = struct.pack('!f', f)
        else:
            print(
                f"Incorrect value for precision. Precision takes values: '64-bit' and '32-bit', but you provided {precision}")
        # Convert the packed bytes to a binary string
        binary_string = ''.join(f'{byte:08b}' for byte in packed)
        bin_array = np.array([int(i) for i in binary_string], dtype=np.int8)
        return bin_array
    # df = df[['tconst', 'genres']]
    df = df[(df['startYear'] >= st.session_state.start_year) & (df['startYear'] <=st.session_state.end_year)]
    df['averageRating']  = df['averageRating'].astype(float)
    if type == "Movies":
        filtered_list = ['movie', 'tvMovie', 'video']
    elif type == "TV Shows":
        filtered_list = ['tvSeries', 'tvMiniSeries']
    else:
        filtered_list = ['movie', 'tvMovie', 'tvSeries', 'tvMiniSeries', 'video']
    df = df[df['titleType'].isin(filtered_list)]
    if movie_id is None:
        def optimize_genre_filter_v3(df, filter_genres):
            # Create a regex pattern for matching genres
            pattern = '|'.join(map(re.escape, filter_genres))

            # Direct string matching without parsing
            return df[df['genres'].str.contains(pattern)]

        df = optimize_genre_filter_v3(df, filter_genres)
        if not filter_genres:
            choice = random.sample(all_genres, random.randint(1,5))
            df =optimize_genre_filter_v3(df, choice)

            ids =df[['tconst','averageRating']]


        else:
            df = optimize_genre_filter_v3(df, filter_genres)
            ids = df[['tconst','averageRating']]
        # print(filter_genres)
        # print(ids)
        
        probabilities = ids['averageRating'] / sum(ids['averageRating'])
        # probabilities = probabilities / np.sum(probabilities) 
        # print(ids.columns)
        # print(sum(probabilities))
        imdb_id = np.random.choice(ids['tconst'], p=probabilities)
    else:
        imdb_id = movie_id

    # def get_genre(genre: List[str]):

    #     genre_bin = np.zeros(len(subgenres))
    #     if genre:
    #         genre = [i for i in genre if i in subgenres]

    #         for i in genre:
    #             genre_bin[subgenres.index(i)] = 1

    #     return genre_bin
    def get_genre(genre: List[str]):

        genre_bin = []
        #if genre:
        genre = [i for i in genre if i in subgenres]

        for i in genre[:10]:
            index = subgenres.index(i)
            bin_array = np.array([int(i) for i in list(np.binary_repr(int(index), 8))])
            genre_bin.append(bin_array)
        if len(genre_bin) < 10:
            for i in range(10-len(genre_bin)):
                bin_array = np.array([int(i) for i in list(np.binary_repr(int(0), 8))])
                genre_bin.append(bin_array)


        return genre_bin

    def get_boxoffice(budget: int, width: int = 32):
        if budget < 2**width:
            budget_bin = np.array(
                [int(i) for i in np.binary_repr(budget, width=width)])
        else:
            budget_bin = np.array(
                [int(i) for i in np.binary_repr(2**width-1, width=width)])
        return budget_bin

    c = ['TV Movie', 'TV Series', 'Short', 'TV Mini Series', 'Movie' 'Video']
    d = ['h', 'm']
    url = f"https://www.imdb.com/title/{imdb_id}/"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

    # for i in range(5):

    for i in range(5):
        # Send a GET request to the URL with headers
        response = requests.get(url, headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.content, 'html.parser')

            # {"type":"application/ld+json", "id":"__NEXT_DATA__"})
            original_title = soup.find("script", id="__NEXT_DATA__")
            a = json.loads(original_title.text)
            try:
                status = a['props']['pageProps']['mainColumnData']['productionStatus']['currentProductionStage']['text']
                if not status:
                    status = " "
            except:
                status = ' '
            if status != 'Released':
                return get_plot(filter_genres, df, all_genres, movie_id=None, released_only=released_only, retries=retries-1)

            try:
                budget = int(a['props']['pageProps']['mainColumnData']
                             ['productionBudget']['budget']['amount'])
                if not budget:
                    budget = 0
            except:
                budget = 0  # Assign a default value or handle the error
            budget = int(np.round(budget/1000000))
            budget = get_boxoffice(budget=budget, width=10)

            try:
                openingWeekendGross = int(
                    a['props']['pageProps']['mainColumnData']['openingWeekendGross']['gross']['total']['amount'])
                if not openingWeekendGross:
                    openingWeekendGross = 0
            except:
                openingWeekendGross = 0
            openingWeekendGross = int(np.round(openingWeekendGross/1000000))
            openingWeekendGross = get_boxoffice(budget=openingWeekendGross, width=10)
            try:
                revenue = int(a['props']['pageProps']['mainColumnData']
                              ['worldwideGross']['total']['amount'])
                if not revenue:
                    revenue = 0
            except:
                revenue = 0
            revenue = int(np.round(revenue/1000000))
            revenue = get_boxoffice(budget=revenue, width=12)

            try:
                trivia = " ".join([i['node']['text']['plaidHtml']
                                  for i in a['props']['pageProps']['mainColumnData']['trivia']['edges']])
                if not trivia:
                    trivia = " "
            except:
                trivia = ' '

            try:
                wins = int(a['props']['pageProps']
                           ['mainColumnData']['wins']['total'])
                if not wins:
                    wins = 0
            except:
                wins = 0
            wins = get_boxoffice(budget=wins, width=10)

            try:
                nominationsExcludeWins = int(
                    a['props']['pageProps']['mainColumnData']['nominationsExcludeWins']['total'])
                if not nominationsExcludeWins:
                    nominationsExcludeWins = 0
            except:
                nominationsExcludeWins = 0
            nominationsExcludeWins = get_boxoffice(
                budget=nominationsExcludeWins, width=10)

            try:
                goofs = " ".join([i['node']['text']['plaidHtml']
                                 for i in a['props']['pageProps']['mainColumnData']['goofs']['edges']])
                if not goofs:
                    goofs = " "
            except:
                goofs = ' '

            try:
                origin = [i['text'] for i in a['props']['pageProps']
                          ['mainColumnData']['countriesOfOrigin']['countries']]
                origin = " ".join(origin)
                if not origin:
                    origin = " "
            except:
                origin = []
                origin = " "

            try:
                trailers = [i['node']['playbackURLs'][0]['url'] for i in a['props']
                            ['pageProps']['aboveTheFoldData']['primaryVideos']['edges']]
            except:
                trailers = []

            try:
                keywords = [i['node']['text'] for i in a['props']
                            ['pageProps']['aboveTheFoldData']['keywords']['edges']]
                keywords = " ".join(keywords)
                if not keywords:
                    keywords = " "
            except:
                # keywords = []
                keywords = " "

            try:
                genres = [i['node']['primaryText']['text'] for i in a['props']
                          ['pageProps']['aboveTheFoldData']['interests']['edges']]
            except:
                genres = []
            genres = " ".join(genres)
            genres = get_genre(genres)

            try:
                languages = [i['text'] for i in a['props']['pageProps']
                             ['mainColumnData']['spokenLanguages']['spokenLanguages']]
                languages = " ".join(languages)
                if not languages:
                    languages = " "

            except:
                languages = " "

            try:
                # a = soup.find("ul", {"class": "ipc-inline-list ipc-inline-list--show-dividers sc-ec65ba05-2 joVhBE baseAlt","role": "presentation"}).find_all("li")

                # a = [i.text for i in a]
                # [i for i in a if i not in c and i[-1] not in d and not i[:4].isdigit() ][0]
                movie_rating = a['props']['pageProps']['aboveTheFoldData']['certificate']['rating']
                if not movie_rating:
                    movie_rating = " "
            except:
                movie_rating = ' '

            try:
                image = a['props']['pageProps']['aboveTheFoldData']['primaryImage']['url']
                # soup.find("div", class_ ="ipc-media ipc-media--poster-27x40 ipc-image-media-ratio--poster-27x40 ipc-media--media-radius ipc-media--baseAlt ipc-media--poster-l ipc-poster__poster-image ipc-media__img").find('img')['src']
                if not image:
                    image = " "

            except:
                image = ' '
            try:
                quotes = a['props']['pageProps']['mainColumnData']['quotes']['edges'][0]['node']['lines'][0]['text']
                if not quotes:
                    quotes = " "
            except:
                quotes = " "
            break

        elif response.status_code >= 500:
            budget, openingWeekendGross, revenue, status, trivia, wins, nominationsExcludeWins, goofs, origin, trailers, keywords, genres, languages, movie_rating, image, quotes = get_boxoffice(budget=0, width=10), get_boxoffice(
                budget=0, width=10), get_boxoffice(budget=0, width=12), " ", " ", get_boxoffice(budget=0, width=10), get_boxoffice(budget=0, width=10), " ", " ", [], " ", get_genre([]), " ", " ", " ", " "

            continue

        else:
            budget, openingWeekendGross, revenue, status, trivia, wins, nominationsExcludeWins, goofs, origin, trailers, keywords, genres, languages, movie_rating, image, quotes = get_boxoffice(budget=0, width=10), get_boxoffice(
                budget=0, width=10), get_boxoffice(budget=0, width=12), " ", " ", get_boxoffice(budget=0, width=10), get_boxoffice(budget=0, width=10), " ", " ", [], " ", get_genre([]), " ", " ", " ", " "

            break

    def extract_text(div):
        try:
            span = div.find("span")
            result = div.text.replace(span.text, "").strip()
        except:
            result = div.text.strip()
        return result

    for i in range(5):

        response2 = requests.get(f"{url}plotsummary/", headers=headers)

        if response2.status_code == 200:
            soup2 = BeautifulSoup(response2.content, 'html.parser')
            # {"type":"application/ld+json", "id":"__NEXT_DATA__"})
            original_title2 = soup2.find("script", id="__NEXT_DATA__")
            b = json.loads(original_title2.text)

            try:
                summary = " ".join([i['node']['plotText']['plaidHtml'] for i in b['props']
                                   ['pageProps']['contentData']['data']['title']['plotSummaries']['edges']])
                if not summary:
                    summary = " "
                # " ".join([extract_text(i) for i in soup2.find("div", {"class": "sc-f65f65be-0 dQVJPm", "data-testid":"sub-section-summaries"}).find("ul").find("li").find("div", class_="ipc-html-content-inner-div")])
            except:
                summary = " "
            try:
                summary_text = b['props']['pageProps']['contentData']['data']['title'][
                    'plotSummaries']['edges'][0]['node']['plotText']['plaidHtml']
                if not summary_text:
                    summary_text = " "
                # " ".join([extract_text(i) for i in soup2.find("div", {"class": "sc-f65f65be-0 dQVJPm", "data-testid":"sub-section-summaries"}).find("ul").find("li").find("div", class_="ipc-html-content-inner-div")])
            except:
                summary_text = " "
            try:
                synopsis = " ".join([i['node']['plotText']['plaidHtml'] for i in b['props']
                                    ['pageProps']['contentData']['data']['title']['plotSynopsis']['edges']])
                if not synopsis:
                    synopsis = " "
                # "".join([i.find("div", class_="ipc-html-content-inner-div").text for i in soup2.find("div", {"class": "sc-f65f65be-0 dQVJPm", "data-testid":"sub-section-synopsis"}).find("ul").find_all('li')])
            except:
                synopsis = " "
            try:
                synopsis_text = b['props']['pageProps']['contentData']['data']['title'][
                    'plotSynopsis']['edges'][0]['node']['plotText']['plaidHtml']
                if not synopsis_text:
                    synopsis_text = " "
                # "".join([i.find("div", class_="ipc-html-content-inner-div").text for i in soup2.find("div", {"class": "sc-f65f65be-0 dQVJPm", "data-testid":"sub-section-synopsis"}).find("ul").find_all('li')])
            except:
                synopsis_text = " "

            break
        else:
            summary, synopsis = " ", " "

    primaryTitle, originalTitle, genres_i, actors, crew, directors, writers = df[df['tconst'] == imdb_id]['primaryTitle'].values[0], df[df['tconst'] == imdb_id]['originalTitle'].values[0], df[df['tconst'] == imdb_id][
        'genres'].values[0], df[df['tconst'] == imdb_id]['actors'].values[0], df[df['tconst'] == imdb_id]['crew'].values[0], df[df['tconst'] == imdb_id]['directors'].values[0], df[df['tconst'] == imdb_id]['writers'].values[0]
    numVotes, runtimeMinutes, averageRating = int(df[df['tconst'] == imdb_id]['numVotes'].values[0]), int(df[df['tconst'] == imdb_id]['runtimeMinutes'].values[0]),  get_boxoffice(budget=int(np.round(df[df['tconst'] == imdb_id]['averageRating'].values[0])), width=4)
    #float_to_binary(df[df['tconst'] == imdb_id]['averageRating'].values[0], precision="32-bit")
    numVotes, runtimeMinutes = get_boxoffice(budget=int(np.round(numVotes/100000)), width=10), get_boxoffice(
        # , get_boxoffice(budget= averageRating, width= 4)
        budget=int(np.round(runtimeMinutes/60)), width=8)


    # Define the text you want to embed
    texts = [
        primaryTitle, originalTitle, actors, crew, directors, writers, trivia,
        goofs, origin,
        keywords, languages, movie_rating, quotes, summary, synopsis
    ]

    # print(imdb_id)
    # Get embeddings for the texts
    embeddings = get_embeddings(texts)
    embeddings = np.array([np.array(i) for i in embeddings])
    embeddings = np.array([np.array([float_to_binary(i)
                          for i in j]).reshape(-1) for j in embeddings])

    primaryTitle, originalTitle, actors, crew, directors, writers, trivia, goofs, origin, keywords, languages, movie_rating, quotes, summary, synopsis = embeddings

    startYear, endYear = df[df['tconst'] ==
                            imdb_id]['startYear'].values[0], df[df['tconst'] == imdb_id]['endYear'].values[0]
    startYear, endYear = get_boxoffice(
        budget=startYear, width=11), get_boxoffice(budget=endYear, width=11)
    return imdb_id, primaryTitle, originalTitle, genres_i, actors, crew, directors, writers, budget, openingWeekendGross, revenue, status, trivia, wins, nominationsExcludeWins, goofs, origin, trailers, keywords, genres, languages, movie_rating, image, quotes, summary, synopsis, numVotes, runtimeMinutes, averageRating, startYear, endYear, summary_text, synopsis_text


def sort_agent_response(agent_response):
    # st.write("Agent response in binary: ", agent_response)
    count = 0
    for element in agent_response:
        if element == 1:
            count += 1
    percentage = (count / len(agent_response)) * 100
    return percentage


def prepare_for_next_video(user_feedback):  # Only run once per video
    print("running pfnv")

    # Update the training history for the current video, based on user feedback
    if st.session_state.natural_language_input:
        st.session_state.training_history[st.session_state.numberVideos,
                                          :] = st.session_state.natural_language_input
        print("Added", st.session_state.natural_language_input, "to agent history")

    st.session_state.numberVideos += 1

    if len(st.session_state.videos_in_list) > 1:
        # Remove the first video from the list
        st.session_state.videos_in_list.pop(0)
        st.session_state.display_video = True
        # Instead of always setting it to "User Disliked," track the actual response
        # Store feedback in history
        st.session_state.training_history[st.session_state.numberVideos -
                                          1, -1] = user_feedback


# function return closest genre and binary encoding of next video and displays it
def next_video(df, all_genres):

    mood = st.session_state.mood
    type = st.session_state.type
    if "Random" in mood:
        filter_genres = random.sample(all_genres, random.randint(1, 5))

    else:
        filter_genres = mood
    print("Starting Next Video Processing")
    (imdb_id, primaryTitle, originalTitle, genres_i, actors, crew, directors, writers, budget, openingWeekendGross, revenue, status, trivia, wins,
     nominationsExcludeWins, goofs, origin, trailers,
     keywords, genres, languages, movie_rating, image, quotes, summary, synopsis, numVotes, runtimeMinutes, averageRating, startYear, endYear, summary_text, synopsis_text) = get_plot(filter_genres, df, all_genres,type)
    print("Done processing Next Video")
    data = imdb_id
    if data not in st.session_state.videos_in_list:
        st.session_state.videos_in_list.append(data)
    st.session_state.display_video = True
    combined_array = np.concatenate([
        primaryTitle, originalTitle, actors, crew, directors, writers,
        budget, openingWeekendGross, revenue, trivia, wins,
        nominationsExcludeWins, goofs, origin, keywords, *genres, languages,
        movie_rating, quotes, summary, synopsis, numVotes, runtimeMinutes, averageRating, startYear, endYear
    ])

    # binary_input_to_agent = genre_binary_encoding+ length_binary + fnf_binary +mood_binary
    binary_input_to_agent = combined_array
    # st.write("binary input:", binary_input_to_agent)++
    # storing the current binary input to reduce redundant calls
    st.session_state.current_binary_input = binary_input_to_agent
    st.session_state.recommendation_result = agent_response(
        binary_input_to_agent)
    percentage_response = sort_agent_response(
        st.session_state.recommendation_result)
    recommended = (str(percentage_response) + "%")

    title = df[df['tconst'] == imdb_id]['primaryTitle'].values[0]
    closest_genre = df[df['tconst'] == imdb_id]['genres'].values[0]
    try:
        closest_genre = ast.literal_eval(closest_genre)

    except:
        try:
            closest_genre = closest_genre.replace(",","|").replace("[","").replace("]","").replace("'","")

        except:
            closest_genre = closest_genre

    length = int(df[df['tconst'] == imdb_id]['runtimeMinutes'].values[0])
    fnf = summary_text if summary_text != " " else synopsis_text
    st.session_state.natural_language_input = [
        title, closest_genre, length, fnf, mood, recommended, "User's Training"]

    if percentage_response >= st.session_state.threshold:
        if st.session_state.threshold < 50:
            # bring the threshold up once videos are being recommended
            st.session_state.threshold += 10


        st.markdown("     Genre: "+str("  ||  ".join(closest_genre)),
                    help="Extracted by an LLM")
        # st.markdown("     Length: "+str(length), help="in minutes; extracted via pytube")
        # st.markdown("     Fiction/Non-fiction: "+str(fnf), help="Extracted by an LLM")
        st.markdown("     User's Mood: "+str(mood[0]),  help="Inputted by user")
        st.markdown("")
        # Centered large text using HTML and CSS
        st.markdown("""
        <style>
        .centered-large {
            text-align: center;
            font-size: 50px;
            font-weight: bold;
            color: #3366FF;  /* Custom color */
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="centered-large">{title}</div>', unsafe_allow_html=True)
        #st.markdown(f"## {title}")
        st.write("**Agent's Recommendation:**  ", recommended)
        if st.session_state.number_videos_not_recommended > 0:
            t0 = (str(st.session_state.number_videos_not_recommended) +
                  " videos were skipped to get to the recommendation below")
            t0_help = "The recommender skips videos that are below 50 percent recommendation, in effect learning on the fly to filter for you. If it's being too selective and skipping too many videos, try clicking the **Stop Recommending** to change things up. (If it gets stuck skipping videos, we temporarily reduce the 50 percent threshold.)"
            st.markdown(t0, help=t0_help)
        st.session_state.number_videos_not_recommended = 0

        if trailers:
            for i in range(len(trailers)):
                try:
                    st.video(trailers[i])
                    break
                except:
                    continue
        elif image != " ":
            st.image(image)
        else:
            st.image("placeholder1.svg")

        if summary_text != " ":
            st.markdown("**Plot:**  " + summary_text)
        else:
            st.write("No Plot Summary Available")
        if synopsis_text != " ":
            st.markdown("**Synopsis:**  " + synopsis_text)
        else:
            st.write("No Synopsis Available")

        # Custom CSS for the button
        # st.markdown("""
        # <style>
        # .custom-button {
        #     background-color: #f63366;
        #     color: white;
        #     padding: 10px 20px;
        #     border: none;
        #     border-radius: 5px;
        #     cursor: pointer;
        # }
        # .custom-button:hover {
        #     background-color: #ff4c4c;
        # }
        # </style>
        # """, unsafe_allow_html=True)
        imdb_url = f"https://www.imdb.com/title/{imdb_id}/"
        # Create a button with custom styling
        # Create a button with custom styling
        # if st.button("Check it out on IMDB", key="custom_button"):
        #     # Use JavaScript to redirect to the IMDb URL
        #     st.markdown(
        #         f'<script>window.open("{imdb_url}", "_blank");</script>',
        #         unsafe_allow_html=True
        #     )
        st.link_button("Check it out on IMDB", imdb_url)

    else:
        if st.session_state.number_videos_not_recommended > 5:
            # If the recommended is going in a non recommending spiral then we counter that by bring the recommend threshold down
            st.session_state.threshold -= 10
            print("Bought threshold down to ", st.session_state.threshold)
        st.session_state.number_videos_not_recommended += 1
        # if st.button("Next"):
        prepare_for_next_video(user_feedback="Video not recommended")
        genre = next_video(df, all_genres)
    return closest_genre  # , genre_binary_encoding


def train_agent(user_response):
    st.session_state.agent.reset_state()
    binary_input = st.session_state.current_binary_input
    if user_response == "RECOMMEND MORE":
        Cpos = True
        Cneg = False
        label = np.ones(
            st.session_state.agent.arch.Z__flat.shape, dtype=np.int8)
        size = 5
    elif user_response == "STOP RECOMMENDING":
        Cneg = True
        Cpos = False
        label = np.zeros(
            st.session_state.agent.arch.Z__flat.shape, dtype=np.int8)
        size = 10

    # st.session_state.agent.next_state(INPUT=binary_input, Cpos=Cpos, Cneg=Cneg, print_result=False)
    for i in range(size):
        st.session_state.agent.reset_state()
        st.session_state.agent.next_state(
            INPUT=binary_input, LABEL=label, print_result=False, unsequenced=True)


def agent_response(binary_input):  # function to get agent response on next video
    # input = get_agent_input()
    st.session_state.agent.reset_state()
    last_response = 0
    n = 5
    for i in range(n):
        response = st.session_state.agent.next_state(
            INPUT=binary_input, print_result=False)
        print("response:", response)
        if i == n-1:
            last_response = response
            print("Last response: ", last_response)
    return last_response



streamlit_analytics2.start_tracking()
############################################################################
with st.sidebar:

    import os

    # def reset_interrupt():
    #     st.session_state.interrupt = False

    # def set_interrupt():
    #     st.session_state.interrupt = True

    st.write("## Current Active Agent:")
    st.write(st.session_state.agent.notes)

    # start_button = st.button(
    #     "Re-Enable Training & Testing",
    #     on_click=reset_interrupt,
    #     help="If you stopped a process\n click to re-enable Testing/Training agents.",
    # )
    # stop_button = st.button(
    #     "Stop Testing",
    #     on_click=set_interrupt,
    #     help="Click to stop a current Test if it is taking too long.",
    # )

    st.write("---")
    st.write("## Load Agent:")

    def load_pickle_files(directory):
        pickle_files = [
            f[:-10] for f in os.listdir(directory) if f.endswith(".ao.pickle")
        ]  # [:-10] is to remove the "ao.pickle" file extension
        return pickle_files

    # directory_option = st.radio(
    #     "Choose directory to retrieve Agents:",
    #     ("App working directory", "Custom directory"),
    #     label_visibility="collapsed"
    # )
    # if directory_option == "App working directory":
    directory = os.path.dirname(os.path.abspath(__file__))
    # else:
    #     directory = st.text_input("Enter a custom directory path:")

    if directory:
        pickle_files = load_pickle_files(directory)

        if pickle_files:
            selected_file = st.selectbox(
                "Choose from saved Agents:", options=pickle_files
            )

            if st.button(f"Load {selected_file}"):
                file_path = os.path.join(directory, selected_file)
                st.session_state.agent = ao.Agent.unpickle(
                    file=file_path, custom_name=selected_file
                )
                st.session_state.agent._update_neuron_data()
                st.write("Agent loaded")
        else:
            st.warning("No Agents saved yet-- be the first!")

    st.write("---")
    st.write("## Save Agent:")

    agent_name = st.text_input(
        "## *Optional* Rename active Agent:", value=st.session_state.agent.notes
    )
    st.session_state.agent.notes = agent_name

    @st.dialog("Save successful!")
    def save_modal_dialog():
        st.write(
            "Agent saved to your local disk (in the same directory as this app).")

    agent_name = agent_name.split("\\")[-1].split(".")[0]
    if st.button("Save " + agent_name):
        st.session_state.agent.pickle(agent_name)
        save_modal_dialog()

    st.write("---")
    st.write("## Download/Upload Agents:")

    @st.dialog("Upload successful!")
    def upload_modal_dialog():
        st.write(
            "Agent uploaded and ready as *Newly Uploaded Agent*, which you can rename during saving."
        )

    uploaded_file = st.file_uploader(
        "Upload .ao.pickle files here", label_visibility="collapsed"
    )
    if uploaded_file is not None:
        if st.button("Confirm Agent Upload"):
            st.session_state.agent = ao.Agent.unpickle(
                uploaded_file, custom_name="Newly Uploaded Agent", upload=True
            )
            st.session_state.agent._update_neuron_data()
            upload_modal_dialog()

    @st.dialog("Download ready")
    def download_modal_dialog(agent_pickle):
        st.write(
            "The Agent's .ao.pickle file will be saved to your default Downloads folder."
        )

        # Create a download button
        st.download_button(
            label="Download Agent: " + st.session_state.agent.notes,
            data=agent_pickle,
            file_name=st.session_state.agent.notes,
            mime="application/octet-stream",
        )

    if st.button("Prepare Active Agent for Download"):
        agent_pickle = st.session_state.agent.pickle(download=True)
        download_modal_dialog(agent_pickle)
############################################################################

# Title of the app
st.title("LLM + WNNs - a Real-Time Personal Movie and TV Show Recommender")
st.write("### *a preview by [aolabs.ai](https://www.aolabs.ai/)*")

# big_left, big_right = st.columns([0.3, 0.7], gap="large")

# with big_left:

# url = st.text_input("Enter link to a YouTube video: ", value=None, placeholder="Optional", help="This app automatically loads YouTube videos, and you can also add a specific YouTube link here.")
# if url !=None:
#     if st.button("Add Link"):
#         try:
#             if url not in st.session_state.videos_in_list:
#                 st.session_state.videos_in_list.insert(0, url)
#                 print(st.session_state.videos_in_list)
#             else:
#                 st.write("Unable to add link as it has already been used; please try another")
#         except Exception as e:
#             st.write("Error: URL not recognised; please try another")
#         st.session_state.display_video = True

# data = get_random_youtube_link()
# while not data:  # Retry until a valid link is retrieved
#     data = get_random_youtube_link()

with st.expander("How this app works:", expanded=True, icon=":material/question_mark:"):
    explain_txt = '''
    Movie and TV Shows recommendations are often impersonal and hard to control-- the ominous *"Algorithm."*

    This app is a preview of a new concept-- a personal recommender that's continuously (re)trained only on your data as you use it; the idea of a recommender as a remote control instead of a pre-trained model trying to get your views.\n
    
    ***How it works:*** an embedding model classifies the video (from its title) as  a specific genre and you as the user can set your mood. There's an AO Agent (see its [architecture here](https://github.com/aolabsai/Recommender/blob/main/arch__Recommender.py)) that learns to associate the genre and mood with your "Recommend More" or "Stop Recommending" button clicks to learn to filter new videos according to your accumulated preferences.  
    
    Below the buttons, you can see the genre, the mood you set, and the percentage of recommendation of the Agent for that particular video. You can then train your Agent by clicking the recommend more or stop buttons.  

    ***Things to try:***
    * see if your AO Agent can learn to recommend a specific genre for you, like News or Podcasts, when you're in a Serious mood.  
    * try unlearning an Agent's recommendations by clicking "Stop Recommending."  
    * if you like an Agent's recommendations, you can save (or even download) it for future sessions using the sidebar to the left.  

    ***Note:*** To make testing easier while in preview mode, the app is fixed on a few genres: Comedy, Music, Documentary, News, Podcast  
    
    Our lightweight systems can easily be extended with more inputs (content-specific inputs like fiction/non-fiction or duration and user-specific inputs like viewing device or day of week) to learn to serve more complicated, nuanced recommendations (eg. maybe you like the News only when you're in a Serious mood on your iPad, and Comedy when in a Random mood on your TV). If you're building a recommender, get in touch to explore the possibilities continuous, per-user training can unlock for your build! [Take a look at the code here.](https://github.com/aolabsai/recommender)
    '''
    st.markdown(explain_txt)
st.session_state.mood = st.multiselect(
    "Set your mood (Preferred Genre(s)):",
    ["Random"] + all_genres,
    default=["Random"]  # Pre-select "Random"
)










st.session_state.date_range = list(range(st.session_state.start_year, st.session_state.end_year+1))
col1, col2, col3 = st.columns(3)

with col1:
    st.session_state.type = st.selectbox(
        "Choose Preferred Title Type:",
        ['Both', 'Movies', 'TV Shows'],
        index=0  # Pre-select "Both"
    )


#
# Callback functions to update the values
#def update_start_year():
#    # Ensure start_year is not greater than end_year
#    if st.session_state.start_year > st.session_state.end_year:
#        st.session_state.start_year = st.session_state.end_year
#
#def update_end_year():
#    # Ensure end_year is not less than start_year
#    if st.session_state.end_year < st.session_state.start_year:
#        st.session_state.start_year = st.session_state.end_year
#
#with col2:
#    st.selectbox(
#        "Start Year",
#        [year for year in st.session_state.date_range if year <= st.session_state.end_year][::-1],
#        key="start_year",
#        on_change=update_start_year,
#        #index=len([year for year in st.session_state.date_range if year <= st.session_state.end_year])-1
#    )
#
#with col3:
#    st.selectbox(
#        "End Year",
#        st.session_state.date_range[st.session_state.date_range.index(st.session_state.start_year):][::-1],
#        key="end_year",
#        on_change=update_end_year,
#        #index=0
#    )
#

with col2:
    st.session_state.start_year_options = [year for year in st.session_state.date_range if year <= st.session_state.end_year][::-1]
    st.session_state.start_year = st.selectbox(
            "Start Year",
            st.session_state.start_year_options,
            
            index=len(st.session_state.date_range)-1,
            #key="start_year"    # Pre-select first item
            )

with col3:
    st.session_state.end_year_options = st.session_state.date_range[st.session_state.date_range.index(st.session_state.start_year):][::-1]
    st.session_state.end_year = st.selectbox(
        "End Year",
        st.session_state.end_year_options,
        index=0, 
        #key="end_year"# Pre-select last item
     )   





#st.session_state.start_year = st.selectbox(
#    "Start Year",
#    st.session_state.date_range,
#    index=0  # Pre-select "Both"
#)
#
#st.session_state.end_year = st.selectbox(
#    "End Year",
#    st.session_state.date_range,
#    index=len(st.session_state.date_range)-1 # Pre-select "Both"
#) 


st.write("Video number: ", str(st.session_state.numberVideos))
small_right, middle, small_left = st.columns(3)
if small_right.button(":green[RECOMMEND MORE]", type="primary", icon=":material/thumb_up:"):
    # Train agent positively as user like recommendation
    train_agent(user_response="RECOMMEND MORE")
    user_feedback = "More"
    prepare_for_next_video(user_feedback)

if small_left.button(":red[STOP RECOMMENDING]", icon=":material/thumb_down:"):
    # train agent negatively as user dislike recommendation
    train_agent(user_response="STOP RECOMMENDING")
    user_feedback = "Less"
    prepare_for_next_video(user_feedback)
if middle.button(":blue[SKIP]", icon=":material/skip_next:"):
    # train agent negatively as user dislike recommendation

    user_feedback = "Skip"
    prepare_for_next_video(user_feedback)

genre = next_video(df, all_genres)
# if st.session_state.display_video == True:


# if data not in st.session_state.videos_in_list:
#     st.session_state.videos_in_list.append(data)

with st.expander("### Agent's Training History"):
    history_titles = ["Title", "Closest Genre", "Duration", "Type",
                      "User's Mood", "Agent's Recommendation", "User's Training"]
    df = pd.DataFrame(
        st.session_state.training_history[0:st.session_state.numberVideos,
                                          :], columns=history_titles
    )
    st.dataframe(df)

st.write("---")
footer_md = """
[View & fork the code behind this application here.](https://github.com/aolabsai/Recommender) \n
To learn more about Weightless Neural Networks and the new generation of AI we're developing at AO Labs, [visit our docs.aolabs.ai.](https://docs.aolabs.ai/)\n
\n
We eagerly welcome contributors and hackers at all levels! [Say hi on our discord.](https://discord.gg/Zg9bHPYss5)
"""
st.markdown(footer_md)
st.image("misc/aolabs-logo-horizontal-full-color-white-text.png", width=300)

streamlit_analytics2.stop_tracking()
