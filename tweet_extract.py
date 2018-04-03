import io
import sys
from textblob import TextBlob
import tweepy
import re

consumer_key = 'dau9wcio88oFoSo8LJVALYqK2'
consumer_secret = 'LNxd2HkmqJaUIMdVLJy5Si5YRJh9dh8jRmWU72lWRJ1KMaPZ0i'

access_token = '833000616922275840-WO1yS6C7PLXMleC4DfbYjeNJLhQQCTI'
access_token_secret = '70sHg54pAETIYwmTXncViHwyjRWNynmItUdZgVtwaFdRk'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Step2 - Call the API
api = tweepy.API(auth)
output = ''
outputfile = io.open('tweetcontent.txt', 'w', encoding='utf8')
outputfile.write(output)
outputfile.close()