from emoji import UNICODE_EMOJI


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

words_list = [emoji for emoji in UNICODE_EMOJI.keys() if UNICODE_EMOJI[emoji] in emoji_sentiment.keys()]

emoji_positive = [0]
emoji_negative = [0]

for words in words_list:
    for e in list(UNICODE_EMOJI.keys()):
        if e in emoji_sentiment.keys():
            if UNICODE_EMOJI[e] in emoji_sentiment.keys() and words.count(UNICODE_EMOJI[e]) > 0:
                if emoji_sentiment[UNICODE_EMOJI[e]] == 1:
                    emoji_positive[0] += 1
                else:
                    emoji_negative[0] += 1


print(emoji_positive, emoji_negative)