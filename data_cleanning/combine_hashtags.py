import pandas as pd
import numpy as np
import spacy

df = pd.read_csv('cleaned_sentiment_data.csv')

temp = df.hashtags.to_list()
hashtags = []
for i in temp:
    hashtags.append(i.lower())

df['hashtags'] = hashtags

for i in range(len(df)):
    for j in range(len(df.hashtags[i])):
        try: 
            if df.hashtags[i][j] == 'b' and df.hashtags[i][j+1] =='i' and df.hashtags[i][j+2] =='d':
                df.hashtags[i] = 'joebiden'
        except: continue
    
for i in range(len(df)):
    for j in range(len(df.hashtags[i])):
        try: 
            if df.hashtags[i][j] == 't' and df.hashtags[i][j+1] =='r' and df.hashtags[i][j+2] =='u'and df.hashtags[i][j+3] =='m':
                df.hashtags[i] = 'donaldtrump'
        except: continue

for i in range(len(df)):
    for j in range(len(df.hashtags[i])):
        try: 
            if df.hashtags[i][j] == 'e' and df.hashtags[i][j+1] =='l' and df.hashtags[i][j+2] =='e'and df.hashtags[i][j+3] =='c':
                df.hashtags[i] = '2020election'
        except: continue
    
for i in range(len(df)):
    for j in range(len(df.hashtags[i])):
        try: 
            if df.hashtags[i][j] == 'v' and df.hashtags[i][j+1] =='o' and df.hashtags[i][j+2] =='t':
                df.hashtags[i] = 'vote'
        except: continue

df.to_csv('combined_tags_final_2.csv', index = False)