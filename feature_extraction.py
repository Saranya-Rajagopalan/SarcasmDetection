# Code to extract features from a given dataset(.csv file) and generate a feature list(.csv file)

# Import required libraries
import csv
import os
import re
from emoji import UNICODE_EMOJI
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
emoji_sentiment = {':100:':1,
 ':blue_heart:':1,
 ':blush:':1,
 ':broken_heart:':-1,
 ':clap:':1,
 ':confused:':-1,
 ':cry:':-1,
 ':disappointed:':-1,
 ':expressionless:':-1,
 ':eyes:':1,
 ':facepunch:':-1,
 ':flushed:':-1,
 ':grin:':1,
 ':hand:':1,
 ':heart:':1,
 ':heart_eyes:':1,
 ':hearts:':1,
 ':heavy_check_mark:':1,
 ':imp:':-1,
 ':information_desk_person:':1,
 ':joy:':1,
 ':kiss:':1,
 ':kissing_heart:':1,
 ':neutral_face:':-1,
 ':notes:':1,
 ':ok_hand:':1,
 ':pensive:':-1,
 ':pray:':1,
 ':purple_heart:':1,
 ':rage:':-1,
 ':raised_hands:':1,
 ':relaxed:':1,
 ':relieved:':-1,
 ':scream:':-1,
 ':see_no_evil:':1,
 ':sleeping:':-1,
 ':sleepy:':-1,
 ':smile:':1,
 ':smirk:':-1,
 ':sob:':-1,
 ':speak_no_evil:':1,
 ':stuck-out_tongue:':1,
 ':stuck-out_tongue_closed_eyes:':1,
 ':sunglasses:':1,
 ':sweat_smile:':1,
 ':thumbsup:':1,
 ':tired_face:':-1,
 ':triumph:':-1,
 ':two_hearts:':1,
 ':unamused:':-1,
 ':v:':1,
 ':wave:':1,
 ':weary:':-1,
 ':wink:':1,
 ':yum:':1}


# Variable Initiailization
non_sarcastic_list = []
sarcastic_list = []
positive = False
negative = False
inversions = []

exclamation_count = []
question_mark_count = []

user_mention_count = []

interjection_count = []
label = []
negative_count = []
positive_count = []
features = []
upperCase = []
accuracy = []
emoji_positive = []
emoji_negative = []
repeat_letter_words = []
emoji_count = []
sentence_polarity = []
# Reference Lists

interjections = ['wow', 'haha', 'lol', 'rofl', 'lmao', 'kidding', 'wtf', 'duh']
exclude = ['I', 'U.S']
emojis = [':)', ';)', '🤔', '🙈', 'así', '😌', 'ds', '💕','👭', ':-)',':p', '(y)']

# emojis = []
emoji_dict = {}

j = -1

sid = SentimentIntensityAnalyzer()
ps = PorterStemmer()
lemm = WordNetLemmatizer()

FEATURE_LIST_CSV_FILE_PATH = os.curdir + "\\data\\feature_list.csv"


def user_mentions(tweet):
    return re.findall("@([a-zA-Z0-9]{1,15})", tweet)


# Reads every tweet in the dataset.csv word by word and extracts features
with open( os.curdir + '\\data\\dataset.csv', 'rU', encoding='utf8') as fp:
    nsreader = csv.reader(fp, delimiter=',')
    matcher = re.compile(r'(.)\1*')
    for i, line in enumerate(nsreader):
        j += 1
        if j > 2:
            break
        emoji_positive.append(0)
        emoji_negative.append(0)
        upperCase.append(0)
        inversions.append(0)
        interjection_count.append(0)
        exclamation_count.append(0)
        question_mark_count.append(0)
        user_mention_count.append(0)
        non_sarcastic_list.append(line)
        label.append(0)
        negative_count.append(0)
        positive_count.append(0)
        repeat_letter_words.append(0)
        emoji_count.append(0)

        # Generate a separate list of labels
        label[j] = int(line[1])
        tweet = line[2]

        # user counts
        user_mention_count[j] = user_mention_count[j] + len(user_mentions(tweet))

        # Feature - Punctuation [Includes punctuation which influence most sarcastic comments ('!' and '?')]
        exclamation_count[j] = exclamation_count[j] + tweet.count('!')
        question_mark_count[j] = question_mark_count[j] + tweet.count('?')
        sentence_polarity.append(sid.polarity_scores(tweet))

        for words in tweet.split(' '):
            repeat_letters = [match.group() for match in matcher.finditer(words)]
            for segments in repeat_letters:
                if len(segments) >= 3 and str(segments).isalpha():
                    repeat_letter_words[j] += 1
                    break

            # Feature - UpperCase word [which is not an interjection]
            if words.isupper() and words not in exclude and words not in interjections:
                upperCase[j] += 1

            # Feature - Emoji [Compared with a list of Unicodes and common emoticons]
            for e in list(UNICODE_EMOJI.keys()):
                if UNICODE_EMOJI[e] in emoji_sentiment.keys():
                    emoji_count[j] += words.count(e)
                    if words.count(e)>0:
                        if emoji_sentiment[UNICODE_EMOJI[e]] > 0:
                            emoji_positive[j] += 1
                        elif words.count(UNICODE_EMOJI[e]) > 0:
                            emoji_negative[j] += 1

            # Feature - Interjection ['Word' converted to lower case and compared with the list of common interjections]
            for interj in interjections:
                if words.lower().count(interj):
                    interjection_count[j] += 1


            # sentiment.append((words, sid.polarity_scores(words)))
            ss = sid.polarity_scores(words)
            if ss["neg"] == 1.0:
                negative = True
                negative_count[j] += 1
                if positive:
                    inversions[j] += 1 
                    positive = False
            elif ss["pos"] == 1.0:
                positive = True
                positive_count[j] += 1
                if negative:
                    inversions[j] += 1
                    negative = False



# Create a single list of lists with label and all the extracted features
feature_label = list(zip(label, positive_count, negative_count, inversions, question_mark_count, exclamation_count,
                         upperCase, interjection_count))

# Headers for the new feature list
headers = ["label", "positive_count", "negative_count", "inversions", "punctuations", "upperCase", "interjections",
           "emoji"]

# Writing headers to the new .csv file
with open(FEATURE_LIST_CSV_FILE_PATH, "w") as header:
    header = csv.writer(header)
    header.writerow(headers)
    
# Append the feature list to the file
with open(FEATURE_LIST_CSV_FILE_PATH, "a") as feature_csv:
    writer = csv.writer(feature_csv)
    for line in feature_label:
        writer.writerow(line)
