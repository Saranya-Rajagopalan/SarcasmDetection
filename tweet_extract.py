import csv
import io
import re
import time
import logging
import tweepy
import twython
import os

# Get the Twitter API authentication tokens from the developer.twitter.com website by creating a new application
consumer_key = 'xxxx'
consumer_secret = 'yyy'
access_token = 'access_token'
access_token_secret = 'access_token_secret'

# Authenticate Tweepy api
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
twitter = twython.Twython(consumer_key, consumer_secret, access_token, access_token_secret)

# Headers for the csv file
headers = ['INDEX', 'LABEL', 'TWEET']
no_of_entries = 0

# file names used
normal_file_name = os.curdir+'\\data\\normal.txt', 'r'
sarcastic_file_name = os.curdir+'\\data\\sarcastic.txt', 'r'
data_set_file_name = 'dataset.csv'

NORMAL_LABEL = 0
SARCASM_LABEL = 1

def save_tweets_with_hashtag(filename, hashtag_list=[]):
    tweetfile = io.open(filename, 'ab')
    for data in hashtag_list:
        tweets = tweepy.Cursor(api.search, q=data, tweet_mode='extended').items(20000)
        for tweet in tweets:
            if 'retweeted_status' in dir(tweet):
                result = re.sub(r"http\S+", "", tweet.retweeted_status.full_text)
            else:
                result = re.sub(r"http\S+", "", tweet.full_text)

            tweetfile.write(result.encode('utf8'))
            tweetfile.write(bytes('\n', 'utf-8'))
            print(tweet.full_text)
    tweetfile.close()


def save_tweets_with_id(data_set_file, id_of_tweet, label):
    global no_of_entries
    with open(data_set_file, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        for ids in id_of_tweet:
            try:
                tweet = twitter.show_status(id=ids)
            except twython.exceptions.TwythonRateLimitError:
                logging.error("Upto " + str(ids) + "printed. RateLimit reached returning!")
                return
            except twython.exceptions.TwythonError:
                continue
            result = re.sub(r"http\S+", "", tweet['text'])
            no_of_entries = no_of_entries + 1
            row = {'INDEX': no_of_entries, 'LABEL': label, 'TWEET': result.encode('utf8')}
            writer.writerow(row)
            # print(row)


##########################################################################################
#  Extract tweets from the tweet ids of normal tweets
##########################################################################################
with open(data_set_file_name, 'a') as csvfile:
    csv.DictWriter(csvfile, fieldnames=headers).writeheader()

id_list = []

tweet_id = 0
for id in normal_file_name.readlines():
    if id != '\n':
        id_list.append(int(id))

for i in range(len(id_list)/400):
    save_tweets_with_id(data_set_file_name, id_list[400 * i:(i + 1) * 400], NORMAL_LABEL)
    print(i)
    # Wait for 15 minutes before proceeding because Twitter API has request rate limit per
    # user for a time frame of 15 minutes
    print("Waiting for 750 seconds...")
    time.sleep(750)


###########################################################################################
#  Extract tweets from the tweet ids of Sarcastic tweets
###########################################################################################

with open(sarcastic_file_name, 'a') as csvfile:
    csv.DictWriter(csvfile, fieldnames=headers).writeheader()

id_list = []

tweet_id = 0
for id in normal_file_name.readlines():
    if id != '\n':
        id_list.append(int(id))

for i in range(len(id_list)/400):
    save_tweets_with_id(data_set_file_name, id_list[400 * i:(i + 1) * 400], SARCASM_LABEL)
    print(i)
    # Wait for 15 minutes before proceeding because Twitter API has request rate limit per
    # user for a time frame of 15 minutes
    print("Waiting for 750 seconds...")
    time.sleep(750)


