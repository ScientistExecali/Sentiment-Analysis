#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis - A Comparison of Machine Learning Models
# 
# __Name:__ Krithika Venkatanath <br>
# __Student ID:__ 103539219
# 
# Social media is widely used to express one’s opinion on a particular topic. The textual data available in tweets on Twitter, Facebook and Instagram posts, comments on YouTube, etc. is present in an unstructured format. Further, people may express their thoughts using different colloquialisms or slangs and emoticons. As a result, understanding or processing this data is a challenge. However, processing this data can yield significant insights into what people think about a company’s products and can be leveraged in marketing campaigns.
# 
# Sentiment analysis, often known as opinion mining, is a technique used in natural language processing (NLP) to determine the emotional element of a document. This is a common technique used by organisations to analyse product reviews, customer feedback and survey results.
# 
# In this project, sentiment analysis is performed on tweets using different supervised machine learning approaches. A performance comparison of each model is provided.

# # 1. Setup
# 
# To run this Jupyter notebook, please uncomment and run the following cells. This will setup Twint, a tool to extract tweets from Twitter.
# 
# ## 1.1. Twint Installation

# In[1]:


# # Install Twint

# !git clone --depth=1 https://github.com/twintproject/twint.git
# %cd twint
# !git pull origin master
# !pip install -r requirements.txt
# !python setup.py install


# In[2]:


# !pip install --upgrade aiohttp && pip install --force-reinstall aiohttp-socks


# In[3]:


# # Install Nest Asyncio

# !pip install nest_asyncio


# ## 1.2. Import Packages

# In[4]:


#Importing packages and modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import datetime
from collections import Counter

#nltk
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')

#gensim
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
from gensim.models import Word2Vec

#wordcloud 
from wordcloud import WordCloud

#Multiprocessing
import multiprocessing

# #Sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, recall_score, get_scorer, f1_score, roc_auc_score,precision_score
from sklearn.feature_extraction.text import TfidfVectorizer

#Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

#filter warnings
import warnings
warnings.filterwarnings('ignore')


# # 2. Data Acquisition
# 
# To acquire the data, two main steps are followed:
# 
# * Tweets are scraped from Twitter using Twint
# * The CSV generated is loaded onto the Jupyter Notebook
# 
# ## 2.1. Data Scraping

# In[5]:


# Move into the Twint directory

get_ipython().run_line_magic('cd', 'twint')


# In[6]:


import nest_asyncio
nest_asyncio.apply()

import twint


# Extract data from Twitter using the keyword "Pizza Hut" to find at least 10,000 tweets on Pizza Hut.

# In[7]:


config = twint.Config()
config.Search = "Pizza Hut"
config.Lang = "en"
config.Limit = 10000
config.Store_csv = True
config.Output = "pizza"

twint.run.Search(config)


# ## 2.2. Data Loading

# In[8]:


df = pd.read_csv('pizza/tweets.csv')
df.head()


# In[9]:


df.shape


# In[10]:


df.info()


# In[11]:


df.describe(include =[np.object])


# In[12]:


df.describe(include =[np.number])


# # 3. Data Preparation
# 
# In data preparation, the following three steps take place:
# 
# * Data Labelling
# * Data Cleaning
# * Data Processing
# 
# ## 3.1. Data Labelling
# 
# Data labelling was performed using VADER (Valence Aware Dictionary for sEntiment Reasoning) (Hutto, 2022). VADER is a sentiment analysis tool that detects both polarity and intensity of the sentiment (i.e.) it can detect whether a text expresses positive, negative or neutral sentiment and how strong the sentiment is.
# 
# The tweets are labelled as -1 if they display a negative sentiment, 0 if they display a neutral sentiment and 1 if they display a positive sentiment. These labels are given in the ‘label’ column. Numerical labelling is done to make the training process easier since a supervised learning approach is adopted. However, corresponding ‘negative’, ‘neutral’ and ‘positive’ values are given in the ‘sentiment’ column.
# 
# ### VADER - Sentiment Intensity Analyser

# In[13]:


import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()


# In[14]:


# Function to label the tweets with negative, neutral or positive sentiments
def label_tweets(sentence):
    
    """
    The function labels each tweet as negative, neutral or positive based
    on the following values
    
    Negative -> -1
    Neutral  ->  0
    Positive ->  1
    """
    
    result = sia.polarity_scores(sentence)

    if result['compound'] >= 0.05:
        return 1
    elif result['compound'] <= -0.05:
        return -1
    else:
        return 0


# In[15]:


# Labelling the tweets with the sentiment
df['label'] = df['tweet'].apply(label_tweets)


# In[16]:


df['sentiment'] = df['label'].map({-1:'negative', 0:'neutral', 1:'positive'})
df


# In[17]:


# Analysing the distribution of sentiments across the data set
df['sentiment'].value_counts()


# ## 3.2. Data Cleaning

# In[18]:


df


# ### 3.2.1. Duplicates
# 
# Duplicates are identified and removed from the data set.

# In[19]:


df[df.duplicated()].shape


# In[20]:


df = df.drop_duplicates(ignore_index=True)


# In[21]:


df['sentiment'].value_counts()


# ### 3.2.2. Empty Columns
# 
# Columns having more than 90% null values are dropped.

# In[22]:


df.info()


# In[23]:


# Dropping columns that are completely empty
df = df.dropna(axis=1, how='all')
df.info()


# In[24]:


# Dropping columns that are more than 90% empty
df = df.drop(columns=['place', 'quote_url', 'thumbnail'])
df.info()


# ### 3.2.3. Data Type Check
# 
# Checking if the data type of each column is appropriate.
# 
# Here, the 'date' column is changed to datetime format for easy processing.

# In[25]:


df['date'] = pd.to_datetime(df['date'])
df['date']


# In[26]:


df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df.info()


# ### 3.2.4. Missing Values
# 
# A missing value is found in the 'name' column. It is replaced with the 'username' value for the corresponding row.

# In[27]:


df['name'].isnull().any()


# In[28]:


df[df['name'].isnull()]


# In[29]:


df['name'] = df['name'].fillna(df['username'])


# In[30]:


df.info()


# ## 3.3. Data Processing
# 
# Natural Language Processing (NLP) processes are applied in this section. It consists of the following steps:
# 
# * Filter data set for English tweets
# * Cleaning of Tweets
# * Tokenisation
# * Removal of Stop Words
# * Lemmatisation
# * Generation of Ngrams (Bigrams)
# * Creation of Word Embeddings

# ### 3.3.1. Filter English Tweets

# In[31]:


df['language'].value_counts()


# In[32]:


df = df[df['language']=='en']
df.shape


# In[33]:


df['label'].value_counts()


# ### 3.3.2. Cleaning Tweets

# In[34]:


stop_words = stopwords.words("english")
stop_words


# In[35]:


nltk.download('wordnet')
lemma = WordNetLemmatizer()


# ### 3.3.3. Tokenisation, Removal of Stop Words and Lemmatisation

# In[36]:


def clean_tweet(tweet):
    '''
    tweet: String
           Input Data
    tweet: String
           Output Data
           
    func: Convert tweet to lower case
          Replace URLs with a space in the message
          Replace ticker symbols with space. The ticker symbols are any stock symbol that starts with $.
          Replace  usernames with space. The usernames are any word that starts with @.
          Replace everything not a letter or apostrophe with space
          Remove single letter words
          lemmatize, tokenize (nouns and verb), remove stop words, filter all the non-alphabetic words, then join
          them again

    '''
    
    tweet = tweet.lower()
    tweet = re.sub('https?:\/\/[a-zA-Z0-9@:%._\/+~#=?&;-]*', ' ', tweet)
    tweet = re.sub('\$[a-zA-Z0-9]*', ' ', tweet)
    tweet = re.sub('\@[a-zA-Z0-9]*', ' ', tweet)
    tweet = re.sub('[^a-zA-Z\']', ' ', tweet)
    tweet = ' '.join( [w for w in tweet.split() if len(w)>1] )
    
    tweet=' '.join([lemma.lemmatize(x) for x in nltk.wordpunct_tokenize(tweet) if x not in stop_words])
    tweet=[lemma.lemmatize(x,nltk.corpus.reader.wordnet.VERB) for x in nltk.wordpunct_tokenize(tweet) if x not in stop_words]
    return tweet


# In[37]:


def clean_hashtags(hashtags):
    '''
    hashtags: String
              Input Data
    hashtags: String
              Output Data
           
    func: Convert hashtags to lower case
          Replace ticker symbols with space. The ticker symbols are any stock symbol that starts with $.
          Replace everything not a letter or apostrophe with space
          Removes any spaces or specified characters at the start and end of hashtags.
          
    '''
    if hashtags:
        hashtags = hashtags.lower()
        hashtags = re.sub('\$[a-zA-Z0-9]*', ' ', hashtags)
        hashtags = re.sub('[^a-zA-Z]', ' ', hashtags)
        hashtags=hashtags.strip() 
    return hashtags


# In[38]:


df['tweet_tokens']=df['tweet'].apply(lambda x:clean_tweet(x))
df["clean_tweet"]=df["tweet_tokens"].apply(lambda x:' '.join(x))
df.head()


# In[39]:


df['hashtags'] = df['hashtags'].apply(lambda x:clean_hashtags(x))
df.head()


# In[40]:


df['hashtags']


# ### 3.3.4. Generating Ngrams and Word Embeddings

# In[41]:


sentences = [row for row in df['tweet_tokens']]

phrases = Phrases(sentences, min_count=1)
bigram = Phraser(phrases)
sentences = bigram[sentences]
sentences[2]


# In[42]:


word2vec_model = Word2Vec(min_count=1,
                         window=5,
                         vector_size=100,
                         sample=1e-5, 
                         alpha=0.03, 
                         min_alpha=0.0007, 
                         negative=20,
                         seed=42,
                         workers=multiprocessing.cpu_count()-1)

#building vocab of the word2vec model from the custom data
word2vec_model.build_vocab(sentences)


# In[43]:


word2vec_model.train(sentences, total_examples=word2vec_model.corpus_count, epochs=30, report_delay=1)


# Analysing the similarity between the word 'tomato' and other word vectors

# In[44]:


word2vec_model.wv.most_similar(positive=["tomato"])


# In[45]:


word2vec_model.wv.most_similar(positive=["pizza"])


# In[46]:


word2vec_model.wv.most_similar(positive=["mushroom"])


# In[47]:


word2vec_model.save("word2vec.model")


# In[48]:


word_vectors = word2vec_model.wv
words = pd.DataFrame(word_vectors.index_to_key)
words.columns = ['words']
words['vectors'] = words.words.apply(lambda x: word_vectors[f'{x}'])
words.head(10)


# # 4. Data Exploration
# ## 4.1. Univariate Analysis

# Visualising the most common words used in tweets overall as well as based on sentiment.
# 
# It can be seen that words such as like, good, thank, please, etc. are classified as containing positive sentiments. Similarly, offensive words are classified as containing negative sentiments. Neutral sentiments mostly contain verbs such as try, make, eat, need, know, etc.

# ### 4.1.1. Top Words Used

# In[49]:


top = Counter([item for sublist in df['tweet_tokens'] for item in sublist])
temp = pd.DataFrame(top.most_common(20))
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap='Blues')


# In[50]:


plt.figure(figsize=(10,10))

sns.barplot(y='Common_words', x='count', data=temp)

plt.title("Top Words Used", fontsize=20)
plt.xlabel("Count", fontsize=16)
plt.ylabel("Common Words", fontsize=16)
plt.xticks(fontsize=14); #';' is used to suppress text output describing xticks
plt.yticks(fontsize=14); #';' is used to suppress text output describing yticks


# #### Creating Separate Data Frames for each Sentiment

# In[51]:


Positive_tweets = df[df['sentiment']=='positive']
Negative_tweets = df[df['sentiment']=='negative']
Neutral_tweets = df[df['sentiment']=='neutral']


# ### 4.1.2. Top Positive Words Used

# In[52]:


top = Counter([item for sublist in Positive_tweets['tweet_tokens'] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(20))
temp_positive.columns = ['Common_words','count']
temp_positive.style.background_gradient(cmap='Greens')


# In[53]:


plt.figure(figsize=(10,10))

sns.barplot(y='Common_words', x='count', data=temp_positive)

plt.title("Top Positive Words Used", fontsize=20)
plt.xlabel("Count", fontsize=16)
plt.ylabel("Common Words", fontsize=16)
plt.xticks(fontsize=14); #';' is used to suppress text output describing xticks
plt.yticks(fontsize=14); #';' is used to suppress text output describing yticks


# ### 4.1.3. Top Negative Words Used

# In[54]:


top = Counter([item for sublist in Negative_tweets['tweet_tokens'] for item in sublist])
temp_negative = pd.DataFrame(top.most_common(20))
temp_negative.columns = ['Common_words','count']
temp_negative.style.background_gradient(cmap='Reds')


# In[55]:


plt.figure(figsize=(10,10))

sns.barplot(y='Common_words', x='count', data=temp_negative)

plt.title("Top Negative Words Used", fontsize=20)
plt.xlabel("Count", fontsize=16)
plt.ylabel("Common Words", fontsize=16)
plt.xticks(fontsize=14); #';' is used to suppress text output describing xticks
plt.yticks(fontsize=14); #';' is used to suppress text output describing yticks


# ### 4.1.4. Top Neutral Words Used

# In[56]:


top = Counter([item for sublist in Neutral_tweets['tweet_tokens'] for item in sublist])
temp_neutral = pd.DataFrame(top.most_common(20))
temp_neutral.columns = ['Common_words','count']
temp_neutral.style.background_gradient(cmap='Purples')


# In[57]:


plt.figure(figsize=(10,10))

sns.barplot(y='Common_words', x='count', data=temp_neutral)

plt.title("Top Neutral Words Used", fontsize=20)
plt.xlabel("Count", fontsize=16)
plt.ylabel("Common Words", fontsize=16)
plt.xticks(fontsize=14); #';' is used to suppress text output describing xticks
plt.yticks(fontsize=14); #';' is used to suppress text output describing yticks


# ### 4.1.5. Count of Tweets based on Sentiment
# 
# It is seen that there is a total of 3,784 positive tweets, 3,140 neutral tweets and 1,835 negative tweets about Pizza Hut.

# In[58]:


plt.figure(figsize=(7,7))

sns.countplot(x='sentiment', data=df)

plt.title("Number of Tweets based on Sentiment", fontsize=20)
plt.xlabel("Sentiment", fontsize=16)
plt.ylabel("Number of Tweets", fontsize=16)
plt.xticks(fontsize=14); #';' is used to suppress text output describing xticks
plt.yticks(fontsize=14); #';' is used to suppress text output describing yticks


# ### 4.1.6. Proportional Distribution of Tweets based on Sentiment
# 
# The visualisation portrays the proportional distribution of tweets comprising of 43.2% positive tweets, 35.8% neutral tweets and 20.9% negative tweets.

# In[59]:


plt.figure(figsize=(10,10))

plt.pie(df['sentiment'].value_counts(),
        labels=df['sentiment'].value_counts().index,
        autopct="%0.1f%%",
        textprops={'fontsize':18})

plt.title("Percentage of Tweets based on Sentiment", fontsize=20);


# ## 4.2. Bivariate Analysis
# 
# ### 4.2.1. Distribution of Tweets based on Date and Sentiment
# 
# It can be seen that 4th, 5th and 6th November 2022 recorded the top three highest number of tweets regarding Pizza Hut. This is possibly due to a greater number of people ordering pizza on weekends. This hypothesis is proven by Figure 9 since 4th, 5th and 6th November 2022 are Friday, Saturday and Sunday, respectively. Hence, there are more number of tweets on days with more orders.

# In[60]:


plt.figure(figsize=(7,7))

temp = pd.DataFrame()
temp['date'] = df['date'].dt.date
temp['sentiment'] = df['sentiment']
temp = temp.sort_values(by='date')

sns.countplot(x='date', hue='sentiment', data=temp)

plt.title("Number of Tweets based on Sentiment", fontsize=20)
plt.xlabel("Date", fontsize=16)
plt.ylabel("Number of Tweets", fontsize=16)
plt.xticks(rotation=90, fontsize=14); #';' is used to suppress text output describing xticks
plt.yticks(fontsize=14); #';' is used to suppress text output describing yticks


# ### 4.2.2. Distribution of Tweets based on Token Count
# 
# From the graph, it can be seen that on an average, all sentiments usually have approximately five tokens in their tweets.

# In[61]:


plt.figure(figsize=(20,10))

temp = pd.DataFrame()
temp['sentiment'] = df['sentiment']
temp['token_count'] = pd.Series([len(item) for item in df['tweet_tokens']])

sns.histplot(x='token_count', data=temp, fill=True, hue='sentiment', kde=True)

plt.title("Distribution of Tweets based on Token Count", fontsize=20)
plt.xlabel("Number of Tokens in Each Tweet", fontsize=20)
plt.ylabel("Count", fontsize=20)
plt.xticks(fontsize=18);
plt.yticks(fontsize=18);


# # 5. Data Modelling
# 
# The feature was set to the cleaned tweets generated and the target variable was set to the label column. The data set was then split into training and testing sets with 80% as training data and 20% as testing data. Feature engineering was performed to convert the tokens and bigrams into a Term Frequency – Inverse Document Frequency (TF-IDF) matrix to facilitate in the training process.
# 
# Once the data was ready for modelling, stratified K-fold cross-validation was performed with five folds. Stratified K-fold cross-validation was chosen as it is better for imbalanced data sets since it ensures equal representation of all classes across each fold. Further, hyperparameter tuning was performed for the random forest classifier and the linear SVC. The random forest classifier’s number of trees were set to the values 100, 150, 200, 250 and 300. Similarly, the linear SVC’s hyperparameter C was set to values 0.01, 0.1, 1, 10 and 100.

# In[62]:


df.head()


# ## 5.1. Feature Engineering

# In[63]:


# Selecting Feature and Target
X = df['clean_tweet']
y = df['label']

# Train-Test Split to have 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[64]:


print("\nData Partitioning:\n")
print("Training Data Size: ", X_train.shape)
print("Testing Data Size: ", X_test.shape)


# In[65]:


# Vectorise the feature
vectorizer = TfidfVectorizer(min_df=3,sublinear_tf=True,encoding="latin-1", ngram_range=(1,2),
                             stop_words='english')


# In[66]:


X_train_tf= vectorizer.fit_transform(X_train.reset_index()["clean_tweet"]).toarray()
X_test_tf = vectorizer.transform(X_test.reset_index()["clean_tweet"]).toarray()


# In[67]:


X_train_tf.shape


# In[68]:


X_train_tf


# In[69]:


# Displaying feature names
feature_names = vectorizer.get_feature_names()
feature_names


# ## 5.2. Stratified K-fold Cross-Validation (K=5)

# In[70]:


from sklearn.model_selection import StratifiedKFold

skfCV5 = StratifiedKFold(n_splits=5)

# Creating a Pandas Dataframe to store the results of
# all the cross-validations

skfCVResults = {'Model':[],
                'Number_of_Folds':[],
                'Fold_Number':[],
                'Accuracy':[],
                'Precision':[],
                'Recall':[],
                'F1_Score':[]}

# Initialising the Dataframe
skfCVResultsDataframe = pd.DataFrame(skfCVResults)

skfCVResultsDataframe


# In[71]:


def calculate_performance(model_name, foldNumber, vlData_y, y_pred):

    print("\nConfusion Matrix for ", model_name, " model")
    print(confusion_matrix(vlData_y, y_pred))
    
    print("\nClassification Report for ", model_name, " model")
    print(classification_report(vlData_y, y_pred, target_names=['negative', 'neutral', 'positive']))
    
    accuracy = accuracy_score(vlData_y, y_pred)
    precision = precision_score(vlData_y, y_pred, average='macro')
    recall = recall_score(vlData_y, y_pred, average='macro')
    f1Score = f1_score(vlData_y, y_pred, average='macro')

    # Appending iteration details to the cross-validation results dataframe
    newRow = {'Model': model_name,
              'Number_of_Folds': 5,
              'Fold_Number': foldNumber,
              'Accuracy': accuracy,
              'Precision': precision,
              'Recall': recall,
              'F1_Score': f1Score}
    
    return newRow


# In[72]:


# Creating Numpy arrays to facilitate easy processing
y_train_array = y_train.to_numpy()
y_test_array = y_test.to_numpy()


# In[73]:


print("Stratified 5-fold Cross Validation")

foldNumber = 0

for train_index, test_index in skfCV5.split(X_train_tf, y_train_array):
    
    # Printing Fold Information
    foldNumber = foldNumber + 1
    print (60*"-") # Prints a line of hyphens for presentation purposes
    print("Fold number:", foldNumber)
    print (60*"-")
    
    # Printing trData and vlData
    trData_X = X_train_tf[train_index]
    trData_y = y_train_array[train_index]
    vlData_X = X_train_tf[test_index]
    vlData_y = y_train_array[test_index]

    # Perform Cross-Validation on different Models
    # Logistic Regression
    model = LogisticRegression(random_state=0)
    model.fit(trData_X, trData_y)
    y_pred = model.predict(vlData_X)
    
    newRow = calculate_performance('Logistic Regression', foldNumber, vlData_y, y_pred)
    skfCVResultsDataframe = skfCVResultsDataframe.append(newRow, ignore_index=True)
    
    # Random Forest Classifier
    for value in range(100, 301, 50):
        model = RandomForestClassifier(n_estimators=value, random_state=0)
        model.fit(trData_X, trData_y)
        y_pred = model.predict(vlData_X)

        model_name = 'Random Forest Classifier (trees='+str(value)+')'
        newRow = calculate_performance(model_name, foldNumber, vlData_y, y_pred)
        skfCVResultsDataframe = skfCVResultsDataframe.append(newRow, ignore_index=True)
        
    # Linear SVC
    for value in [0.01, 0.1, 1, 10, 100]:
        model = LinearSVC(C=value, random_state=0)
        model.fit(trData_X, trData_y)
        y_pred = model.predict(vlData_X)

        model_name = 'Linear SVC (C='+str(value)+')'
        newRow = calculate_performance(model_name, foldNumber, vlData_y, y_pred)
        skfCVResultsDataframe = skfCVResultsDataframe.append(newRow, ignore_index=True)

    # Multinomial Naive Bayes
    model = MultinomialNB()
    model.fit(trData_X, trData_y)
    y_pred = model.predict(vlData_X)
    
    newRow = calculate_performance('Multinomial NB', foldNumber, vlData_y, y_pred)
    skfCVResultsDataframe = skfCVResultsDataframe.append(newRow, ignore_index=True)


# In[74]:


skfCVResultsDataframe


# In[75]:


# The cross-validation results are given in skfCVFinalResult
skfCVFinalResult = skfCVResultsDataframe.groupby(['Model'])['Accuracy','Precision','Recall','F1_Score'].mean()
skfCVFinalResult


# In[76]:


# Graphical Visualisation of Model Comparison

skfCVFinalResult.plot(kind='barh', rot=0, figsize=(15,50))

plt.title("Comparison of Classification Models", fontsize=20)
plt.xlabel("Classification Model", fontsize=18)
plt.ylabel("Model", fontsize=18)
plt.xticks(fontsize=16);
plt.yticks(fontsize=16);
plt.legend(loc='upper right')


# It is observed from the above graph that Linear SVC with hyperparameter C set to the value 1 and Logistic Regression models have the highest mean accuracy, mean precision and mean recall. Hence, the Linear SVC (C=1) and Logistic Regression models will be selected for training and testing using the training and testing sets created earlier. Their performances will be compared using the confusion matrix and classification reports.

# ## 5.3. Training and Testing

# ### 5.3.1. Linear SVC (C=1)

# In[77]:


svc = LinearSVC(C=1,random_state=0)
svc.fit(X_train_tf, y_train_array)
y_pred = svc.predict(X_test_tf)


# In[78]:


# Confusion Matrix
cf_matrix = confusion_matrix(y_test_array, y_pred)

# Graphical Visualisation
plt.figure(figsize=(5,5))

cf_matrix_heatmap = sns.heatmap(cf_matrix, annot=True, fmt='g')
cf_matrix_heatmap.set_xticklabels(['negative', 'neutral', 'positive'])
cf_matrix_heatmap.set_yticklabels(['negative', 'neutral', 'positive'])

plt.title("Confusion Matrix - Linear SVC (C=1)", fontsize=18)
plt.xlabel("Predicted Labels", fontsize=16)
plt.ylabel("Actual Labels", fontsize=16)
plt.xticks(fontsize=14);
plt.yticks(fontsize=14);


# In[79]:


print("\nClassification Report - Linear SVC (C=1)")
print(classification_report(y_test_array, y_pred, target_names=['negative', 'neutral', 'positive']))


# ### 5.3.2. Logistic Regression

# In[80]:


logreg = LogisticRegression(random_state=0)
logreg.fit(X_train_tf, y_train_array)
y_pred = logreg.predict(X_test_tf)


# In[81]:


# Confusion Matrix
cf_matrix = confusion_matrix(y_test_array, y_pred)

# Graphical Visualisation
plt.figure(figsize=(5,5))

cf_matrix_heatmap = sns.heatmap(cf_matrix, annot=True, fmt='g')
cf_matrix_heatmap.set_xticklabels(['negative', 'neutral', 'positive'])
cf_matrix_heatmap.set_yticklabels(['negative', 'neutral', 'positive'])

plt.title("Confusion Matrix - Logistic Regression", fontsize=18)
plt.xlabel("Predicted Labels", fontsize=16)
plt.ylabel("Actual Labels", fontsize=16)
plt.xticks(fontsize=14);
plt.yticks(fontsize=14);


# In[82]:


print("\nClassification Report - Logistic Regression")
print(classification_report(y_test_array, y_pred, target_names=['negative', 'neutral', 'positive']))


# By measuring the accuracies of the two models, it can be seen that although their performances are comparable, the Linear SVC (C=1) model is more accurate than the Logistic Regression model.
# 
# ## 5.4. Verification
# 
# To further verify sentiment analysis performed by the Linear SVC (C=1) and Logistic Regression models, two random tweets are selected with their labels. The label predicted by the Linear SVC (C=1) and Logistic Regression models is given.
# 
# ### 5.4.1. Neutral Sentiment Tweet

# In[83]:


tweet = df.iloc[100]['tweet']
tweet


# In[84]:


df.iloc[100]['label']


# In[85]:


vector = vectorizer.transform([tweet])

# Linear SVC prediction
result = svc.predict(vector)
result[0]


# In[86]:


# Logistic Regression prediction
result = logreg.predict(vector)
result[0]


# ### 5.4.2. Positive Sentiment Tweet

# In[87]:


tweet = df.iloc[101]['tweet']
tweet


# In[88]:


df.iloc[101]['label']


# In[89]:


vector = vectorizer.transform([tweet])

# Linear SVC prediction
result = svc.predict(vector)
result[0]


# In[90]:


# Logistic Regression prediction
result = logreg.predict(vector)
result[0]


# # Storing the Model
# 
# The Linear SVC (C=1) model is stored as a pickle file for future use.

# In[91]:


# Importing Pickle package
import pickle


# In[92]:


# Saving the model

modelFile = open('model.pickle', 'wb')
pickle.dump(svc, modelFile)
modelFile.close()


# # Conclusion
# 
# Twint was used to collect tweets from Twitter containing the keyword ‘Pizza Hut’, which were labelled using VADER. For the data set acquired, different supervised machine learning models were applied on to find out the best performing model.
# 
# The Linear SVC (C=1) model has optimum performance while implementing sentiment analysis on the tweets to categorise them as having positive, neutral or negative sentiment. It has an accuracy of 80%.
