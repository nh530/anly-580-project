#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:27:39 2019

@author: ryanpatrickcallahan
"""
import re 
import pandas as pd 
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from statistics import stdev
import numpy as np


data = pd.read_csv("Dev/INPUT.txt", sep='\t', header=None, 
                   names=['id', 'target', 'tweet'], encoding='utf-8')
data.drop(['id', 'target'], axis=1, inplace=True)


# 1. Total number of tweets.
# 11906
print("Number of tweets: " + str(data.shape[0]))

# Construct dictionaries that will contain distinct character n-grams
n_2 = {}
n_3 = {}
n_4 = {}
n_5 = {}
n_6 = {}
n_7 = {}

# 2. Total number of characters.
# 1354375
num = 0
# Moving through all tweets in the data
for tweet in data.values:
    # Add to total character count
    num += len(tweet[0])
    
    # Moving through each letter of the tweet...
    # Creating distinct character n-grams
    for i in range(1,len(tweet[0])):
        # Form two-character combos (i.e. n-2 grams)
        string2 = tweet[0][i-1:i+1]
        # Check whether n-2 gram is already in corpus
        if string2 in n_2:
            # If already in corpus, add to count
            n_2[string2] += 1
        # If n-2 gram has not already been seen, add to list
        else:
            n_2[string2] = 1
        # For situations where we are at least 3 characters away from the end-character   
        if len(tweet[0]) - i >= 2:
            # Form three-character combo
            string3 = tweet[0][i-1:i+2]
            # Check whether n-3 gram is already in corpus
            if string3 in n_3:
                # If already in corpus, add to count
                n_3[string3] += 1
            # If n-3 has not already been seen, add to list
            else:
                n_3[string3] = 1
                
        ## Continue equivalently for n-4, n-5, n-6, n-7: check whether we're far
        ## enough away from end of tweet to form forward-looking string of that size,
        ## save the string, and either save new dict entry or add to dict counter
        if len(tweet[0]) - i >= 3:
            string4 = tweet[0][i-1:i+3]

            if string4 in n_4:
                n_4[string4] += 1
            # If n-4 gram has not already been seen, add to list
            else:
                n_4[string4] = 1
                
        if len(tweet[0]) - i >= 4:
            string5 = tweet[0][i-1:i+4]

            if string5 in n_5:
                n_5[string5] += 1
            # If n-5 gram has not already been seen, add to list
            else:
                n_5[string5] = 1
            
        if len(tweet[0]) - i >= 5:
            string6 = tweet[0][i-1:i+5]

            if string6 in n_6:
                n_6[string6] += 1
            # If n-6 gram has not already been seen, add to list
            else:
                n_6[string6] = 1
                
        if len(tweet[0]) - i >= 6:
            string7 = tweet[0][i-1:i+6]

            if string7 in n_7:
                n_7[string7] += 1
            # If n-7 gram has not already been seen, add to list
            else:
                n_7[string7] = 1               
            
    
print("Total number of characters, including spaces: " + str(num))




# 3. Total number of distinct words (vocabulary)
# 41760

# NLTK has a tweet tokenizer
# corpus includes hashtags, handles, punctuations (:, ..., ., :), and urls.
# twtker = TweetTokenizer()
# temp = []
# for text in data.values:    
#     for words in twtker.tokenize(text[0]):
#         temp.append(words)

# NLTK word_tokenize api
# Doesnt remove punctuation, or urls.  
# temp = []
# for text in data.values:
#     for words in word_tokenize(text[0]):
#         temp.append(words)



# Does not ignore emojis
# A somewhat better way to write a tokenizer with multiple
# regular expressions is in this snippet below (shorter version of 
# http://sentiment.christopherpotts.net/code-data/happyfuntokenizing.py)
# The order is important (match from first to last)
#### Figure out how to not match urls and remove the # from hashtags. 
regexes=(
# Keep usernames together (any token starting with @, followed by A-Z, a-z, 0-9)        
r"(?:@[\w_]+)",
            
# Keep hashtags together (any token starting with #, followed by A-Z, a-z, 0-9, _, or -)
r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",

# abbreviations, e.g. U.S.A.
r'(?:[A-Z]\.)+',
r'[A-Za-z]\.(?:[A-Za-z0-9]\.)+',
r'[A-Z][bcdfghj-np-tvxz]+\.',

# URL, e.g. https://google.com
r'https?:\/\/(?:www\.',
r'(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}',
r'www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}',
r'https?:\/\/(?:www\.',
r'(?!www))[a-zA-Z0-9]+\.[^\s]{2,}',
r'www\.[a-zA-Z0-9]+\.[^\s]{2,}',

# currency and percentages, e.g. $12.40, 82%
r'\$?\d+(?:\.\d+)?%?',

# Numbers i.e. 123,56.34
r'(?:[0-9]+[,]?)+(?:[.][0-9]+)?',

# Keep words with apostrophes, hyphens, and underscores together
r"(?:[a-z][a-z’'\-_]+[a-z])",

# Keep all other sequences of A-Z, a-z, 0-9, _ together
r"(?:[\w_]+)",

# Match words at the end of a sentence.  e.g. tree. or tree!
r'(?:[a-z]+(?=[.!\?]))',

# Everything else that's not whitespace
# It seems like this captures punctuations and emojis and emoticons.  
#r"(?:\S)"
)

big_regex="|".join(regexes)

# Note re.I for performing Perform case-insensitive matching; 
# expressions like [A-Z] will match lowercase letters, too. 
my_extensible_tokenizer = re.compile(big_regex, re.VERBOSE | re.I | re.UNICODE)

# Define list that will hold all tokens
temp = []
# Define dictionary that will hold counts of all distinct tokens
corpus = {}

# Define dictionaries that will store word 2-, 3-, 4-, and 5- grams
corp_2gram = {}
corp_3gram = {}
corp_4gram = {}
corp_5gram = {}


# Iterate through each tweet
for text in data.values:
    # Define counter that will identify the index for the word in the current tweet
    count2 = 0
    # Define string that will store current matches in a given tweet.  
    string = []
    # Each time the REGEX matches, add the matched string to the running list of all words
    for matches in my_extensible_tokenizer.findall(text[0]):
        # Add matched string to list containing elements of current tweet
        string.append(matches)
        # If we are at least 2 words into this tweet...
        if count2 >= 1:
            # Store backward-looking 2-gram in variable with space concatenated in middle
            gram2 = string[count2 - 1] + ' ' + string[count2]
            # Check whether 2-gram is in dictionary
            if gram2 in corp_2gram:
                # If in dictionary, add to the entry's count
                corp_2gram[gram2] += 1
            # If not in entry, create new entry and set count to 1
            else:
                corp_2gram[gram2] = 1
        ## Proceed equivalently for 3-, 4-, and 5-gram dictionaries: check
        ## whether we are sufficiently far into the tweet to form a backward-looking
        ## gram of that size, save the gram, and either add to that gram's counter
        ## in the dictionary or create new entry for that gram.
        if count2 >= 2:
            gram3 = string[count2 - 2] + ' ' + gram2
            if gram3 in corp_3gram:
                corp_3gram[gram3] += 1
            else:
                corp_3gram[gram3] = 1
                
        if count2 >= 3:
            gram4 = string[count2 - 3] + ' ' + gram3
            if gram4 in corp_4gram:
                corp_4gram[gram4] += 1
            else:
                corp_4gram[gram4] = 1
        
        if count2 >= 4:
            gram5 = string[count2 - 4] + ' ' + gram4
            if gram5 in corp_5gram:
                corp_5gram[gram5] += 1
            else:
                corp_5gram[gram5] = 1
                
        count2 += 1
        
         #just in case get empty matches
        if matches != '':
            # Add string to master list
            temp.append(matches)
            
         
# Define variable that will hold counts of each word length
lengths = []


# Looking through entire list of (repeated) words, count instances of distinct words
for word in temp:
    # Add character count (non-whitespace) to counting list
    lengths.append(len(word))
    # If word has already been seen, add one to its count
    if word in corpus:
        corpus[word] += 1
    # If word has not already been seen, add word to list
    else:
        corpus[word] = 1
    
# Print number of distinct words
# 41760
## temp contains urls, which are included in the count.  
## Are urls really words?
print("Number of distinct words: " + str(len(corpus)))


# 4. Average number of words and characters per tweet.
# Avg. Words/Tweet: 16.066
# Avg. Characters/Tweet: 113.756

# Add columns that will store number of characters and words per tweet
data["characters"] = 0

# Write character count of each tweet to the character count column
data["characters"] = data["tweet"].str.len()
## Average number of characters and words per tweet
print("Average number of characters per tweet:" + str(data["characters"].mean()))


# Average number of words per tweet is just total words/total tweets
print("Avg. number of words/tweet is " + str(len(temp)/data.shape[0]))

# 5. Average number and standard deviation of characters/token
# Mean: 5.979
# Standard Dev.: 4.706

print("Average number of characters per token: " + str(sum(lengths)/len(temp)))
print("Standard deviation of characters per token: " + str(stdev(lengths)))

# 6. Total tokens corresponding to 10 most frequent words
# 23268
# Save dictionary of words and counts to dataframe to enable sorting by count
corpus_df = pd.DataFrame(list(corpus.items()), columns = ['word', 'count'])
# Sort by count, in descending fashion
corpus_df = corpus_df.sort_values(by=['count'],ascending=False)
# Sum counts from first 10 rows in sorted dataframe
Top10 = sum(corpus_df['count'][0:9])
print("Total number of times that 10 most popular tokens appear: " + str(Top10))


# 7. Token/type ratio???????




# 8. Number of distinct word n-grams for n=2,3,4,5
# n-2: 130116; n-3: 157675; n-4: 153193; n-5: 142858

print('\nDistinct word 2-grams:' + str(len(corp_2gram)))
print('\nDistinct word 3-grams:' + str(len(corp_3gram)))
print('\nDistinct word 4-grams:' + str(len(corp_4gram)))
print('\nDistinct word 5-grams:' + str(len(corp_5gram)))

# 9. Number of distinct character n-grams for n=2,3,4,5,6,7
# n-2: 8069; n-3: 98192; n-4: 212264; n-5: 361987; n-6: 520197; n-7: 662468

print('\nDistinct char 2-grams:' + str(len(n_2)))
print('\nDistinct char 3-grams:' + str(len(n_3)))
print('\nDistinct char 4-grams:' + str(len(n_4)))
print('\nDistinct char 5-grams:' + str(len(n_5)))
print('\nDistinct char 6-grams:' + str(len(n_6)))
print('\nDistinct char 7-grams:' + str(len(n_7)))


# 10. Plot of token log frequency????????????






## Load in and run tokenizer on all four gold datasets - train, dev, devtest, and test 

dev = pd.read_csv("Gold/dev.txt", sep='\t', header=None, 
                   names=['id', 'class', 'tweet'], encoding='utf-8')
dev.drop(['id'], axis=1, inplace=True)

train = pd.read_csv("Gold/train.txt", sep='\t', header=None, 
                   names=['id', 'class', 'tweet'], encoding='utf-8')
train.drop(['id'], axis=1, inplace=True)

devtest = pd.read_csv("Gold/devtest.txt", sep='\t', header=None, 
                   names=['id', 'class', 'tweet'], encoding='utf-8')
devtest.drop(['id'], axis=1, inplace=True)

test = pd.read_csv("Gold/test.txt", sep='\t', header=None, 
                   names=['id', 'class', 'tweet'], encoding='utf-8')
test.drop(['id'], axis=1, inplace=True)


# Define list that will hold all token types for Gold dev data
temp_gold_dev = []

for text in dev['tweet'].values:
    # Each time the REGEX matches, add the matched string to the running list of all words
    for matches in my_extensible_tokenizer.findall(text):
         #just in case get empty matches
        if matches != '':
            # Add string to master list
            temp_gold_dev.append(matches)
            

# Define list that will hold all token types for Gold training data
temp_gold_train = []

for text in train['tweet'].values:
    # Each time the REGEX matches, add the matched string to the running list of all words
    for matches in my_extensible_tokenizer.findall(text):
         #just in case get empty matches
        if matches != '':
            # Add string to master list
            temp_gold_train.append(matches)
         
            
            
# Define list that will hold all token types for Gold DEVTEST data
temp_gold_devtest = []

for text in devtest['tweet'].values:
    # Each time the REGEX matches, add the matched string to the running list of all words
    for matches in my_extensible_tokenizer.findall(text):
         #just in case get empty matches
        if matches != '':
            # Add string to master list
            temp_gold_devtest.append(matches)
     
        

# Define list that will hold all token types for Gold test data
temp_gold_test = []

for text in test['tweet'].values:
    # Each time the REGEX matches, add the matched string to the running list of all words
    for matches in my_extensible_tokenizer.findall(text):
         #just in case get empty matches
        if matches != '':
            # Add string to master list
            temp_gold_test.append(matches)


# Create null dictionaries that will store types and counts for each dataset
corpus_gold_dev = {}
corpus_gold_train = {}
corpus_gold_devtest = {}
corpus_gold_test = {}
# Create null dictionary that will store types and counts for all gold datasets combined
corpus_gold = {}


# Looking through entire list of (repeated) words, count instances of distinct words
for word in temp_gold_dev:
    # If word has already been seen, add one to its count
    if word in corpus_gold_dev:
        corpus_gold_dev[word] += 1
    # If word has not already been seen, add word to list
    else:
        corpus_gold_dev[word] = 1
    if word in corpus_gold:
        corpus_gold[word] += 1
    else:
        corpus_gold[word] = 1


# Looking through entire list of (repeated) words, count instances of distinct words
for word in temp_gold_train:
    # If word has already been seen, add one to its count
    if word in corpus_gold_train:
        corpus_gold_train[word] += 1
    # If word has not already been seen, add word to list
    else:
        corpus_gold_train[word] = 1
        
    if word in corpus_gold:
        corpus_gold[word] += 1
    else:
        corpus_gold[word] = 1

# Looking through entire list of (repeated) words, count instances of distinct words
for word in temp_gold_devtest:
    # If word has already been seen, add one to its count
    if word in corpus_gold_devtest:
        corpus_gold_devtest[word] += 1
    # If word has not already been seen, add word to list
    else:
        corpus_gold_devtest[word] = 1

    if word in corpus_gold:
        corpus_gold[word] += 1
    else:
        corpus_gold[word] = 1       

# Looking through entire list of (repeated) words, count instances of distinct words
for word in temp_gold_test:
    # If word has already been seen, add one to its count
    if word in corpus_gold_test:
        corpus_gold_test[word] += 1
    # If word has not already been seen, add word to list
    else:
        corpus_gold_test[word] = 1

    if word in corpus_gold:
        corpus_gold[word] += 1
    else:
        corpus_gold[word] = 1     



# 11. Number of types that appear in dev data but not training data
# 5062   
    
# Define variable that will store values found in exclusively the dev data
justdev = corpus_gold_dev.keys() - corpus_gold_train.keys()

print('Number of types found in dev data but not in training data: ' + str(len(justdev)))



# 12. Compare vocab size of combined gold datasets versus input dataset.


# Number of unique words in dev set
unique1 = len(corpus_gold)

# Number of unique words in training set
unique2 = len(corpus_gold)

# Number of unique words in devtest set
unique3 = len(corpus_gold)

# Number of unique words in test set
unique4 = len(corpus_gold)

# Save number of tokens in dev set
full1 = len(temp_gold_dev)
# Save number of tokens in training set        
full2 = len(temp_gold_train)   
# Save number of tokens in devset   
full3 = len(temp_gold_devtest) 
# Save number of tokens in test set
full4 = len(temp_gold_test)

lexicon = pd.DataFrame({'unique':[0,unique1,unique2,unique3,unique4],'total':[0,full1,full1+full2,full1+full2+full3,full1+full2+full3+full4]})
lexicon.plot(x ='total', y='unique', kind = 'line', title='Gold Dataset',marker='.', markersize=15,legend=None)
plt.xlabel("Total Number of Words")
plt.ylabel("Unique Words in Lexicon")


# 13. Class distribution of positive, neutral, and negative tweets in training dataset
# 3017 positive, 2001 neutral, 850 negative
classes = train['class'].value_counts()

classes2 = dev['class'].value_counts()

print('Tweet class distribution is as follows:\n')
print(str(classes.values[0] + classes2.values[0]) + ' ' + classes.index[0] + ' tweets.\n')
print(str(classes.values[1] + classes2.values[1]) + ' ' + classes.index[1] + ' tweets.\n')
print(str(classes.values[2] + classes2.values[2]) + ' ' + classes.index[2] + ' tweets.\n')



# 14. Look at top word types across three classes

positive = train[train['class'] == 'positive']
neutral = train[train['class'] == 'neutral']
negative = train[train['class'] == 'negative']


temp_pos = []

for text in positive['tweet'].values:
    # Each time the REGEX matches, add the matched string to the running list of all words
    for matches in my_extensible_tokenizer.findall(text):
         #just in case get empty matches
        if matches != '':
            # Add string to master list
            temp_pos.append(matches)


temp_neu = []

for text in neutral['tweet'].values:
    # Each time the REGEX matches, add the matched string to the running list of all words
    for matches in my_extensible_tokenizer.findall(text):
         #just in case get empty matches
        if matches != '':
            # Add string to master list
            temp_neu.append(matches)
 
           
temp_neg = []

for text in negative['tweet'].values:
    # Each time the REGEX matches, add the matched string to the running list of all words
    for matches in my_extensible_tokenizer.findall(text):
         #just in case get empty matches
        if matches != '':
            # Add string to master list
            temp_neg.append(matches)





corpus_pos = {}

# Looking through entire list of (repeated) words, count instances of distinct words
for word in temp_pos:
    # If word has already been seen, add one to its count
    if word in corpus_pos:
        corpus_pos[word] += 1
    # If word has not already been seen, add word to list
    else:
        corpus_pos[word] = 1
    
pos_df = pd.DataFrame(list(corpus_pos.items()), columns = ['word', 'count'])
# Sort by count, in descending fashion
pos_df = pos_df.sort_values(by=['count'],ascending=False)


    
corpus_neu = {}

# Looking through entire list of (repeated) words, count instances of distinct words
for word in temp_neu:
    # If word has already been seen, add one to its count
    if word in corpus_neu:
        corpus_neu[word] += 1
    # If word has not already been seen, add word to list
    else:
        corpus_neu[word] = 1

neu_df = pd.DataFrame(list(corpus_neu.items()), columns = ['word', 'count'])
# Sort by count, in descending fashion
neu_df = neu_df.sort_values(by=['count'],ascending=False)


corpus_neg = {}

# Looking through entire list of (repeated) words, count instances of distinct words
for word in temp_neg:
    # If word has already been seen, add one to its count
    if word in corpus_neg:
        corpus_neg[word] += 1
    # If word has not already been seen, add word to list
    else:
        corpus_neg[word] = 1

neg_df = pd.DataFrame(list(corpus_neg.items()), columns = ['word', 'count'])
# Sort by count, in descending fashion
neg_df = neg_df.sort_values(by=['count'],ascending=False)


## Tons of similarity between the three sets. However, a few notes:
## "Donald", "Trump", "Erdogan", and "Jeb" appear only in the top 50 of the negative tweets.
## "Tomorrow" appears more frequently as tweets get more positive.
## "But" appears less frequently as tweets get more positive.
## "Friday" and "Jurassic" appear only in the top 50 of the positive tweets.
## "Apple" appears only in the top 50 of the positive and neutral tweets.
## "Amazon" and "Prime" both appear higher in the list of negative tweet frequency than "Amazon" does in the list of positive tweet frequencies.
Top51 = pd.DataFrame()
Top51['pos'] = pos_df['word'].values[0:50]
Top51['neu'] = neu_df['word'].values[0:50]
Top51['neg'] = neg_df['word'].values[0:50]
Top51



## 15. Compare most common words in training and dev datasets
# Dev dataset has frequent occurrences of
# Obama, SCOTUS, Sunday, Minecraft, Snoop, Rick, Sarah, planned, Ric, Palin,
# Dogg, Netflix, Nike, Serena, and Michelle. Clearly political/pop culture in nature.
# Training dataset has frequent occurrences of Amazon, Friday, Apple, and night.
# The two sets are otherwise fairly similar, but the frequent words in the training
# set more closely match those seen in the overall positive, neutral, and negative
# subsets, suggesting that the training set is either substantially larger or
# less niche.

# Save training dictionary to dataframe so that it can be ordered by count
train_df = pd.DataFrame(list(corpus_gold_train.items()), columns = ['word', 'count'])
# Sort by count, in descending fashion
train_df = train_df.sort_values(by=['count'],ascending=False)

# Save dev dictionary to dataframe so that it can be ordered by count
dev_df = pd.DataFrame(list(corpus_gold_dev.items()), columns = ['word', 'count'])
# Sort by count, in descending fashion
dev_df = dev_df.sort_values(by=['count'],ascending=False)

Top61 = pd.DataFrame()
Top61['train'] = train_df['word'].values[0:60]
Top61['dev'] = dev_df['word'].values[0:60]
Top61