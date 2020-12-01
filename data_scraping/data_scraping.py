import os
import sys
import tweepy as tw
import pandas as pd
import numpy as np
import json
import csv

# The consumer_key, consumer_secret, access_token, access_token_secret are provided by applying a Twitter Developer Account.
# Define the search_words and the date_since and date_until date as variables.
# We scrape the data using search_words: #DonaldTrump, #JoeBiden, #2020Election, #Vote, #Debates2020 respectively
# The saved_file_name is the file that you want to store your data.
# The number_of_tweet is the maximum number of tweets you expect to collect.


def twitter_data_scraping(consumer_key, consumer_secret, access_token, access_token_secret
                        , saved_file_name, search_words, date_since, date_until, number_of_tweet):
    
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)

    csvFile = open(saved_file_name, 'a',encoding='utf-8')
    # Use csv writer.
    csvWriter = csv.writer(csvFile)
    for tweet in tw.Cursor(api.search,
                           q = search_words,
                           since = date_since,
                           until = date_until,
                           lang = "en").items(number_of_tweet):

        # Write a row to the CSV file. We use encode UTF-8.
        csvWriter.writerow([tweet.id, tweet.created_at,tweet.entities,tweet.text.encode('utf-8')])
        # print(tweet.id, tweet.created_at,tweet.entities,tweet.text)
    csvFile.close() 