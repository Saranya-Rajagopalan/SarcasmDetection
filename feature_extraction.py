import csv
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import constants
import numpy as np
import pandas as pd
from ekphrasis.classes.segmenter import Segmenter

sid = SentimentIntensityAnalyzer()
ps = PorterStemmer()
lemm = WordNetLemmatizer()

FEATURE_LIST_CSV_FILE_PATH = os.curdir + "\\data\\feature_list.csv"
DATASET_FILE_PATH = os.curdir + "\\data\\dataset.csv"
stopwords = stopwords.words('english')


def read_data(filename):
    data = pd.read_csv(filename, header=None, encoding="utf-8", names=["Index", "Label", "Tweet"])
    # data = data[data["Index"] > 76750]
    return data


def clean_data(tweet, lemmatize=True, remove_punctuations=True, remove_stop_words=False):
    stopwords = nltk.corpus.stopwords.words('english')
    lemm = nltk.stem.wordnet.WordNetLemmatizer()
    tokens = nltk.word_tokenize(tweet)
    if remove_punctuations:
        tokens = [word for word in tokens if word not in string.punctuation]
    if remove_stop_words:
        tokens = [word for word in tokens if word.lower() not in stopwords]
    if lemmatize:
        tokens = [lemm.lemmatize(word) for word in tokens]
    return tokens


def user_mentions(tweet):
    return len(re.findall("@([a-zA-Z0-9]{1,15})", tweet))


def punctuations_counter(tweet, punctuation_list):
    punctuation_count = {}
    for p in punctuation_list:
        punctuation_count.update({p: tweet.count(p)})
    return punctuation_count


# def hashtag_sentiment(tweet):
#     hash_tag = (re.findall("#([a-zA-Z0-9]{1,25})", tweet))
#     seg = Segmenter(corpus="english")
#     hashtag_polarity = []
#     for hashtag in hash_tag:
#         tokens = (seg.segment(hashtag))
#         ss = sid.polarity_scores(tokens)
#         if 'not' not in tokens.split(' '):
#             hashtag_polarity.append(ss['compound'])
#         else:
#             hashtag_polarity.append(- ss['compound'])
#     sentiment = 0
#     if len(hashtag_polarity) > 0:
#         sentiment = round(float(sum(hashtag_polarity) / float(len(hashtag_polarity))), 2)
#     return sentiment


def getEmojiSentiment(tweet):
    # Feature - Emoji [Compared with a list of Unicodes and common emoticons]
    emoji_sentiment = 0
    count = 0
    # flipped = 0
    for e in constants.emoji_sentiment.keys():
        if e in tweet:
            count += 0
            emoji_sentiment += constants.emoji_sentiment[e]
    if count > 0:
        emoji_sentiment = (float(emoji_sentiment) / float(count))
    return emoji_sentiment


def interjections_counter(tweet):
    interjection_count = 0
    for interj in constants.interjections:
        interjection_count += tweet.lower().count(interj)
    return interjection_count


def captitalWords_counter(tokens):
    upperCase = 0
    for words in tokens:
        if words.isupper() and words not in constants.exclude:
            upperCase += 1
    return upperCase


def repeatLetterWords_counter(tweet):
    repeat_letter_words = 0
    matcher = re.compile(r'(.)\1*')
    repeat_letters = [match.group() for match in matcher.finditer(tweet)]
    for segments in repeat_letters:
        if len(segments) >= 3 and str(segments).isalpha():
            repeat_letter_words += 1
    return repeat_letter_words


def getSentimentScore(tweet):
    return round(sid.polarity_scores(tweet)['compound'], 2)


def polarityFlip_counter(tokens):
    positive = False
    negative = False
    positive_word_count, negative_word_count, flip_count = 0, 0, 0
    for words in tokens:
        ss = sid.polarity_scores(words)
        if ss["neg"] == 1.0:
            negative = True
            negative_word_count += 1
            if positive:
                flip_count += 1
                positive = False
        elif ss["pos"] == 1.0:
            positive = True
            positive_word_count += 1
            if negative:
                flip_count += 1
                negative = False
    return positive_word_count, negative_word_count, flip_count


def POS_count(tokens):
    # tokens = clean_data(tweet, lemmatize= False)
    Tagged = nltk.pos_tag(tokens)
    nouns = ['NN', 'NNS', 'NNP', 'NNPS']
    verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    noun_count, verb_count = 0, 0
    no_words = len(tokens)
    for i in range(0, len(Tagged)):
        if Tagged[i][1] in nouns:
            noun_count += 1
        if Tagged[i][1] in verbs:
            verb_count += 1
    return round(float(noun_count)73 / float(no_words),2), round(float(verb_count) / float(no_words),2)


def intensifier_counter(tokens):
    # tweet = clean_data(tweet, lemmatize= False)
    posC, negC = 0, 0
    for index in range(len(tokens)):
        if tokens[index] in constants.intensifier_list:
            if (index < len(tokens) - 1):
                ss_in = sid.polarity_scores(tokens[index + 1])
                if (ss_in["neg"] == 1.0):
                    negC += 1
                if (ss_in["pos"] == 1.0):
                    posC += 1
    return posC, negC


def skip_grams(tokens, n, k):
    skip_gram_value = 0
    # tokens = clean_data(tweet, lemmatize= False)
    a = [x for x in nltk.skipgrams(tokens, n, k)]
    for j in range(len(a)):
        for k in range(n):
            ss = sid.polarity_scores(a[j][k])
            if (ss["pos"] == 1):
                skip_gram_value += 1
            if (ss["neg"] == 1):
                skip_gram_value -= 1
    return skip_gram_value


def find_common_unigrams(data_set):
    unigram_sarcastic_dict = {}
    unigram_non_sarcastic_dict = {}
    sarcastic_unigram_list = []
    non_sarcastic_unigram_list = []
    tweets = data_set['Tweet'].values
    labels = data_set['Label'].values
    for i, tweet in enumerate(tweets):
        tokens = clean_data(tweet, lemmatize=True, remove_punctuations=True, remove_stop_words=True)
        for words in tokens:
            if words in unigram_sarcastic_dict.keys() and int(labels[i]) == 1:
                unigram_sarcastic_dict[words] += 1
            else:
                unigram_sarcastic_dict.update({words: 1})
            if words in unigram_non_sarcastic_dict.keys() and int(labels[i]) == 0:
                unigram_non_sarcastic_dict[words] += 1
            else:
                unigram_non_sarcastic_dict.update({words: 1})

    # Creat list of high frequency unigrams
    # change value > 'x' where x is the frequency threshold

    for key, value in unigram_sarcastic_dict.items():
        if value > 1000 and key not in stopwords:
            sarcastic_unigram_list.append(key)
    for key, value in unigram_non_sarcastic_dict.items():
        if value > 1000 and key not in stopwords:
            non_sarcastic_unigram_list.append(key)
    return sarcastic_unigram_list, non_sarcastic_unigram_list


def passive_aggressive_counter(tweet):
    sentence_count = 0
    new_sentence = []
    i = 0
    for j in range(len(tweet)):
        if '.' != tweet[j]:
            i += 1
            new_sentence.append(tweet[j])
        else:
            sentence_count += 1
            makeitastring = ''.join(map(str, new_sentence))
            tokens = (nltk.word_tokenize(makeitastring))
            if (len(tokens) > 3):
                return 0
            if (i == len(tweet)):
                return 0
            new_sentence = []
    return sentence_count


# Returns # most common unigrams from non sarcastic tweets which are also present in current tweet
def unigrams_counter(tokens, common_unigrams):
    common_unigrams_count = {}
    for word in tokens:
        if word in common_unigrams:
            if word in common_unigrams_count.keys():
                common_unigrams_count[word] += 1
            else:
                common_unigrams_count.update({word: 1})
    return common_unigrams_count


def normalize( array):
    max = np.max(array)
    min = np.min(array)


    def normalize(x):
        return round(((x-min) / (max-min)),2)
    if max != 0:
        array = [x for x in map(normalize, array)]
    return array


def main():
    data_set = read_data(DATASET_FILE_PATH)
    label = list(data_set['Label'].values)
    tweets = list(data_set['Tweet'].values)
    user_mention_count = []
    exclamation_count = []
    questionmark_count = []
    ellipsis_count = []
    emoji_sentiment = []
    interjection_count = []
    uppercase_count = []
    repeatLetter_counts = []
    sentimentscore = []
    positive_word_count = []
    negative_word_count = []
    polarityFlip_count = []
    noun_count = []
    verb_count = []
    positive_intensifier_count = []
    negative_intensifier_count = []
    skip_bigrams_sentiment = []
    skip_trigrams_sentiment = []
    skip_grams_sentiment = []
    unigrams_count = []
    passive_aggressive_count = []
    emoji_tweet_flip = []
    hashtag_sentiment_score = []

    COMMON_UNIGRAMS = find_common_unigrams(data_set)
    print(COMMON_UNIGRAMS)

    for t in tweets:
        hashtag_sentiment_score.append(0)
        tokens = clean_data(t)
        user_mention_count.append(user_mentions(t))
        p = punctuations_counter(t, ['!', '?', '...'])
        exclamation_count.append(p['!'])
        questionmark_count.append(p['?'])
        ellipsis_count.append(p['...'])
        interjection_count.append(interjections_counter(t))
        uppercase_count.append(captitalWords_counter(tokens))
        repeatLetter_counts.append(repeatLetterWords_counter(t))
        sentimentscore.append(getSentimentScore(t))
        x = polarityFlip_counter(tokens)
        positive_word_count.append(x[0])
        negative_word_count.append(x[1])
        polarityFlip_count.append(x[-1])
        x = POS_count(tokens)
        noun_count.append(x[0])
        verb_count.append(x[1])
        x = intensifier_counter(tokens)
        emoji_sentiment.append(getEmojiSentiment(t))
        positive_intensifier_count.append(x[0])
        negative_intensifier_count.append(x[1])
        skip_bigrams_sentiment.append(skip_grams(tokens, 2, 0))
        skip_trigrams_sentiment.append(skip_grams(tokens, 3, 0))
        skip_grams_sentiment.append(skip_grams(tokens, 2, 2))
        unigrams_count.append(unigrams_counter(tokens, COMMON_UNIGRAMS))
        passive_aggressive_count.append(passive_aggressive_counter(t))
        if (sentimentscore[-1] < 0 and emoji_sentiment[-1] > 0) or (sentimentscore[-1] > 0 and emoji_sentiment[-1] < 0):
            emoji_tweet_flip.append(1)
        else:
            emoji_tweet_flip.append(0)

    feature_label = zip(label, normalize(user_mention_count), normalize(exclamation_count),
                        normalize(questionmark_count), normalize(ellipsis_count), normalize(interjection_count),
                        normalize(uppercase_count), normalize(repeatLetter_counts), sentimentscore,
                        normalize(positive_word_count), normalize(negative_word_count), normalize(polarityFlip_count),
                        noun_count, verb_count, normalize(positive_intensifier_count),
                        normalize(negative_intensifier_count), skip_bigrams_sentiment, skip_trigrams_sentiment,
                        skip_grams_sentiment, emoji_sentiment, normalize(passive_aggressive_count), emoji_tweet_flip,
                        hashtag_sentiment_score)

    # Headers for the new feature list
    headers = ["label", "User mention", "Exclamation", "Question mark", "Ellipsis" "Interjection", "UpperCase",
               "RepeatLetters", "SentimentScore", "positive word count", "negative word count", "polarity flip",
               "Nouns", "Verbs", "PositiveIntensifier", "NegativeIntensifier", "Bigrams", "Trigram", "Skipgrams",
               "Emoji Sentiment", "Passive aggressive count", "Emoji_tweet_polarity flip", "hashtag_polarity"]

    # Writing headers to the new .csv file
    with open(FEATURE_LIST_CSV_FILE_PATH, "w") as header:
        header = csv.writer(header)
        header.writerow(headers)

    # Append the feature list to the file
    with open(FEATURE_LIST_CSV_FILE_PATH, "a") as feature_csv:
        writer = csv.writer(feature_csv)
        for line in feature_label:
            writer.writerow(line)


if __name__ == "__main__":
    main()
