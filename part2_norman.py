import re 
import pandas as pd 
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
import nltk 
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import random



data_dev = pd.read_csv("Gold/dev.txt", sep='\t', header=None, index_col=False,
                   names=['id', 'target', 'tweet'], encoding='utf-8')
data_dev.drop(['id'], axis=1, inplace=True)

data_train = pd.read_csv("Gold/train.txt", sep='\t', header=None, index_col=False,
                   names=['id', 'target', 'tweet'], encoding='utf-8')
data_train.drop(['id'], axis=1, inplace=True)

data_devtest = pd.read_csv("Gold/devtest.txt", sep='\t', header=None, index_col=False,
                           names=['id', 'target', 'tweet'], encoding='utf-8')
data_devtest.drop(['id'], axis=1, inplace=True)

data_test = pd.read_csv("Gold/test.txt", sep='\t', header=None, index_col=False,
                           names=['id', 'target', 'tweet'], encoding='utf-8')
data_test.drop(['id'], axis=1, inplace=True)

kaggle_data = a = pd.read_csv("Gold/kaggle_data.csv", encoding='iso-8859-1',
                              names=["target", "ids", "date", "flag", "user", "tweet"])
kaggle_data.drop(['ids','flag','date','user'],axis=1,inplace=True)
kaggle_data.loc[kaggle_data.loc[:, 'target'] == 0, 'target'] = 'negative'
kaggle_data.loc[kaggle_data.loc[:, 'target'] == 4, 'target'] = 'positive'
kaggle_data = kaggle_data.sample(frac=1).reset_index(drop=True).loc[0:50000,:]

data_dev = data_dev.append(kaggle_data, ignore_index=True)



regexes=(
# Keep usernames together (any token starting with @, followed by A-Z, a-z, 0-9)        
r"(?:@[\w_]+)",

# Keep hashtags together (any token starting with #, followed by A-Z, a-z, 0-9, _, or -)
r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",

# abbreviations, e.g. U.S.A.
r'(?:[A-Z]\.)+',
r'[A-Za-z]\.(?:[A-Za-z0-9]\.)+',
r'[A-Z][bcdfghj-np-tvxz]+\.',

# URL, e.g. https://google.com Ryans's url pattern.
r'https?:\/\/w{0,3}\.?[a-zA-Z0-9-]+\.[a-zA-Z0-9]{1,6}\/?[a-zA-Z0-9\/=]*\s*',
r'w{3}\.[a-zA-Z0-9]+\.[a-zA-Z0-9\/=]+\s*',
r'w{0:3}\.?[a-zA-Z0-9]+\.[a-zA-Z0-9\/=]+\/+[a-zA-Z0-9\/=]*\s*',
r'[a-zA-Z0-9]+\.com\/?[a-zA-Z0-9\/=]*\s*',
r'[a-zA-Z0-9]+\.edu\/?[a-zA-Z0-9\/=]*\s*',
r'[a-zA-Z0-9]+\.gov\/?[a-zA-Z0-9\/=]*\s*',
r'[a-zA-Z0-9]+\.org\/?[a-zA-Z0-9\/=]*\s*',

# currency and percentages, e.g. $12.40, 82%
r'\$?\d+(?:\.\d+)?%?',

# Numbers i.e. 123,56.34
r'(?:[0-9]+[,]?)+(?:[.][0-9]+)?',

# Keep words with apostrophes, hyphens and underscores together
r"(?:[a-z][a-z’'\-_]+[a-z])",

# Keep all other sequences of A-Z, a-z, 0-9, _ together
r"(?:[\w_]+)",

# Match words at the end of a sentence.  e.g. tree. or tree!
r'(?:[a-z]+(?=[.!\?]))',

# Everything else that's not whitespace
# It seems like this captures punctuations and emojis and emoticons.  
#r"(?:\S)"
)
big_regex = "|".join(regexes)
my_extensible_tokenizer = re.compile(big_regex, re.VERBOSE | re.I | re.UNICODE)


trash_pattern = (
# URL, e.g. https://google.com
# This pattern will match any url.  
r'(https?:\/\/(?:www\.',
r'(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}',
r'www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}',
r'https?:\/\/(?:www\.',
r'(?!www))[a-zA-Z0-9]+\.[^\s]{2,}',
r'www\.[a-zA-Z0-9]+\.[^\s]{2,})',

# Numbers i.e. 123,56.34
r'(?:[0-9]+[,]?)+(?:[.][0-9]+)?',

# Keep usernames together (any token starting with @, followed by A-Z, a-z, 0-9)        
r"(?:@[\w_]+)"

)
big_trash_pattern = "|".join(trash_pattern)
trash_tokenizer = re.compile(big_trash_pattern, re.VERBOSE | re.I | re.UNICODE)

hashtag_pattern = (
r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"        
)
hashtag_tokenizer = re.compile(hashtag_pattern, re.VERBOSE | re.I | re.UNICODE)


url_pattern = (
# URL, e.g. https://google.com
# This pattern will match any url.   Ryan's URl pattern
r'https?:\/\/w{0,3}\.?[a-zA-Z0-9-]+\.[a-zA-Z0-9]{1,6}\/?[a-zA-Z0-9\/=]*\s*',
r'w{3}\.[a-zA-Z0-9]+\.[a-zA-Z0-9\/=]+\s*',
r'w{0:3}\.?[a-zA-Z0-9]+\.[a-zA-Z0-9\/=]+\/+[a-zA-Z0-9\/=]*\s*',
r'[a-zA-Z0-9]+\.com\/?[a-zA-Z0-9\/=]*\s*',
r'[a-zA-Z0-9]+\.edu\/?[a-zA-Z0-9\/=]*\s*',
r'[a-zA-Z0-9]+\.gov\/?[a-zA-Z0-9\/=]*\s*',
r'[a-zA-Z0-9]+\.org\/?[a-zA-Z0-9\/=]*\s*',
)
big_url_pattern = "|".join(url_pattern)
url_tokenizer = re.compile(big_url_pattern, re.VERBOSE | re.I | re.UNICODE)


#Setting stop words
stops = list(stopwords.words('english')) 
stop_words = []
for word in stops:
    if word[-3:] != "n't":
        if word[-1] != 'n':
            stop_words.append(word)
        
negation_words = ['not', 
                  'no', 
                  'never',
                  'none',
                  'no one',
                  'nobody',
                  'nothing',
                  'neither',
                  'never',
                  'hardly',
                  'scarcely',
                  'barely',
                  "doesn't",
                  "isn't",
                  "wasn't",
                  "shouldn't",
                  "wouldn't",
                  "couldn't",
                  "won't",
                  "can't",
                  "don't"
                  ]

def randomOversampler(DataFrame):
    # Assumption is that positive class always majority.  
    # This is true for current datasets.  
    neutral_tweets = DataFrame.loc[DataFrame['target'] == 'neutral', :]
    negative_tweets = DataFrame.loc[DataFrame['target'] == 'negative', :]
    neu_rows = neutral_tweets.shape[0]
    neg_rows = negative_tweets.shape[0]
    pos_class_count = DataFrame.groupby(by='target').count().loc['positive',:][0]
    neg_class_count = DataFrame.groupby(by='target').count().loc['negative',:][0]
    neu_class_coutn = DataFrame.groupby(by='target').count().loc['neutral',:][0]
    neu_nums = random.choices(range(neu_rows), k=pos_class_count-neu_class_coutn)
    neg_nums = random.choices(range(neg_rows), k=pos_class_count-neg_class_count)
    DataFrame = DataFrame.append(neutral_tweets.iloc[neu_nums,:], ignore_index=True)
    DataFrame = DataFrame.append(negative_tweets.iloc[neg_nums,:], ignore_index=True)
    return DataFrame


def tweettag(tweet):
    tags = [
            'JJ',
            'NN',
            'NNS',
            'IN',
            'JJR',
            'JJS',
            'POS',
            'RB',
            'RBR',
            'RBS',
            'VB',
            'VBD',
            'BVG',
            'VBN',
            'VBP',
            'VBZ',
            'WRB'
            ]
    twt = []    
    for tag in tags:
        for word in pos_tag(my_extensible_tokenizer.findall(tweet), tagset='universial'):
            if word[1] in tag:
                twt.append((tag,True))
    return(twt)


def url_processsing(text, list):
    url_matches = url_tokenizer.findall(text)
    if url_matches:
        list.append(('url_flag', True))
#        list.append(('url_count', len(url_matches)))
#        list.append(('url_count_2', len(url_matches)**3))
#    elif not url_matches:
#        list.append(('url_count', 0))
#        list.append(('url_count_2', 0))
    return list

def hashtag_processing(text, list):
    hashtag_matches = hashtag_tokenizer.findall(text)
    if hashtag_matches:
        list.append(('hashtag_flag', True))
#        list.append(('hashtag_flag', len(hashtag_matches)))
#        list.append(('hashtag_flag_2', len(hashtag_matches)**2))
#    elif not hashtag_matches:
#        list.append(('hashtag_flag', 0))
#        list.append(('hashtag_flag_2', 0))
    return list

def textPolarity(str, list, sia):
    pol = sia.polarity_scores(str)
    if pol['neg'] > .8:
        list.append((str + '(neg)', True))
    if pol['pos'] > .8:
        list.append((str + '(pos)', True))
    if pol['neu'] > .8:
        list.append((str + '(neu)', True))
    return list

def preprocessing(data):
    tokens = [] # stores all features for dataset.  
    sia = nltk.sentiment.vader.SentimentIntensityAnalyzer()
    for text in data.values: # for each tweet.  
        temp = []  # stores the modified matches for a single tweet
        NOT = 0 # setting variable to 0 that will trigger the variable "negate" to append "_not" to all words following "n't" or "not" in a tweet
        negate = 0 # set variable to 0 that, when equal to 1, will append "_not" to all words following an instance of "n't" or "not"
        for matches in my_extensible_tokenizer.findall(text[1]):
            matches = matches.lower()
            # determine if matches is a url.
            # Set NOT to 1, where it will remain for remainder of tweet until new row resets NOT to 0
            if (matches[-3:] == "n't" or matches in negation_words):
                NOT = 1
            trash_matches = trash_tokenizer.findall(matches)
            temp = url_processsing(matches, temp)
            temp = hashtag_processing(matches, temp)
            temp = textPolarity(matches, temp, sia) 
            # if the match is unwanted, then won't add to temp.  
            if not trash_matches and matches !='':
                #whatisthis = nltk.tag.pos_tag([matches])[0][1]
                if matches not in stop_words:
                    # If the NOT trigger was previously activated in this tweet, add "_not" to this (and all subsequent) words
                    if negate == 1:
                        matches = matches + '_not'
                    temp.append(('contains(' + matches + ')', True))
            # Activate _not trigger if this word ended with "n't" or was "not"
            if NOT == 1:
                negate = 1
        temp = temp + tweettag(text[1])
        tokens.append((dict(temp), text[0]))
    return tokens

data_dev = randomOversampler(data_dev)
data_train = randomOversampler(data_train)
data_devtest = randomOversampler(data_devtest)


train_tokens = preprocessing(data_train)
dev_tokens = preprocessing(data_dev)
devtest_tokens = preprocessing(data_devtest)
test_tokens = preprocessing(data_test)

training_features = train_tokens + dev_tokens + devtest_tokens
test_final = test_tokens # + devtest_tokens

sentiment_analyzer = SentimentAnalyzer()
trainer = NaiveBayesClassifier.train
classifier = sentiment_analyzer.train(trainer=trainer, training_set=training_features)
# Evaluating model on training data.
#sentiment_analyzer.evaluate(training_features, classifier)


sentiment_analyzer.evaluate(test_final, classifier)












            