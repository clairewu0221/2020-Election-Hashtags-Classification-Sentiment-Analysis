#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import json
import re


pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)



df = pd.read_csv('2020election.csv', names = ["id", "date", "hashtages", "text"])
df1 = pd.read_csv('biden&vote&trump_before_elec.csv', names = ["id", "date", "hashtages", "text"])
df2 = pd.read_csv('DonaldTrump_1107.csv', names = ["id", "date", "hashtages", "text"])
df3 = pd.read_csv('DonaldTrump_1108_1107half.csv', names = ["id", "date", "hashtages", "text"])
df4 = pd.read_csv('JoBiden8_9.csv', names = ["id", "date", "hashtages", "text"])
df1 = df1.drop(df.index[0])
df2 = df2.drop(df.index[0])
df3 = df3.drop(df.index[0])
df4 = df4.drop(df.index[0])

df_final = pd.concat([df,df1,df2,df3,df4])

df_final=df_final.drop_duplicates()


def cleaning_data(df):
#     df = df.drop(df.index[0])
    temp = df.date.tolist()
    date = []
    time = []
    for i in temp:
        date.append(i.split(' ')[0])
#         time.append(i.split(' ')[1])
    temp = df.hashtages.tolist()
    res_ht = []
    for i in range(len(temp)):
        ht = []
        temp1 = temp[i]
        text = temp1.replace("\'", "\"")
        try:
            json_form = json.loads(text)
        except:
            res_ht.append([])
            continue
        ht_temp = json_form['hashtags']
        for j in ht_temp:
            ht.append(j['text'])
        res_ht.append(ht)
    temp = df.text.tolist()
    tweets = []
    for t in temp:
        t1 = t[2:-1]
        res = ''
        i=0
        while i <len(t1):
            if t1[i] == '\\':
                i+=4
            else:
                res = res+ t1[i]
                i+=1
        cleaned_Text = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)|(\w+:\/\/\S+)"," ",res).split())
        try:
            if cleaned_Text[0] == 'R' and cleaned_Text[1] == 'T':
                cleaned_Text = cleaned_Text[3::]
                tweets.append(cleaned_Text)
            else:
                tweets.append(cleaned_Text)
        except:
            tweets.append(cleaned_Text)
    df1 = pd.DataFrame()
    df1['id'] = df.id
    df1['date'] = date
    df1['hashtags'] = res_ht
    df1['tweet'] = tweets
    
    # remove the rows with no hashtags
    df2 = df1[df1.astype(str)['hashtags'] != '[]']

    # save the rows with no hashtags for later use
    no_hashtag_rows = df1[df1.astype(str)['hashtags'] == '[]']

    df3 = df2.explode('hashtags').reset_index(drop=True)
    # df3.to_csv('cleaned_2020election.csv')
    return df3

cleaned_df = cleaning_data(df_final)

cleaned_df.to_csv('cleaned_sentiment_data.csv',index = False)



