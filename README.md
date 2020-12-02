# 2020-Election-Hashtags-Classification-Sentiment-Analysis
Twitter, one of the most influential online social media, has 145 million daily active users and 500 million tweets per day. Thousands of tweets occur on Twitter expressing their emotions, opinions, and promoting cultural exchanges. Hashtags, a metadata tag that is prefaced by the hash symbol, play a crucial role on microblogging and photo-sharing services such as Twitter. People widely use hashtags to express what they want to convey.\
In this project, we first perform a good hashtag classifier with high accuracy can promote the sentiment analysis. Then we look at the whats the sentiment on twitter that are related to the U.S election 2020. Given the sentiment we analyzed, the result could be connected with the 2020 presidential election result of having Joe Biden as a projected winner.
# Data Scraping and Cleanning
In this section we perform data collecting and cleaning process.\
data_scrap.py - used tweetpy to access Twitter API to scrap tweets from October 21st to November 8th. The output will create a csv file.\
data_cleanning.py - input the output .csv file from data_scrap.py.The output csv file from data_cleaning.py will be the input of our modeling sections.\
combine_hashtags.py - input the output .csv file from data_cleanning.py. This step is to combine all the similar hashtags to a specific hashtag. The output csv file from combine_hashtags.py will be the input of the sentiment analysis.

# Model
For Classic models.py file, it contains four models which are Multinomial Naive Bayes model with Tf-idf vector or Word Count vector, Random Forest model with Tf-idf vector or Word count vector. Data is the 'combined_tags_final.csv'. Outputs are the metrics tables for models.\
For LSTM.py file, it is used to train the LSTM model with 'combined_tags_final.csv'.


# Sentiment Analysis
We want to understand how people feel about the top four hashtags we collected on Twitter by sentiment analysis. The hashtags are #Joe Biden, #Donald Trump, #2020election, and #vote.\
sentiment_analysis.ipynb - open with jyupter notebook to see the process of our analysis using VADER



