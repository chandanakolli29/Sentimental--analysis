from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
data= pd.read_csv('SocialMedia.csv')
data.info()
def patterns(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt
data['retweet'] = np.vectorize(patterns)(data['Tweet'], "@[\w]*")
#replace special characters with space
data['retweet'] = data['retweet'].str.replace("[^a-zA-Z#]", " ")
data.head()

!pip install streamlit
!pip install vaderSentiment

# Commented out IPython magic to ensure Python compatibility.
# 
# %%writefile app.py
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from wordcloud import WordCloud
# data= pd.read_csv('SocialMedia.csv')
# st.set_option('deprecation.showPyplotGlobalUse', False)
# data['time'] = pd.to_datetime(data['time'],errors='coerce')
# data = data.dropna(subset=['time'])
# # Extract month from the 'time' column
# data['month'] = data['time'].dt.month
# months = sorted(data['month'].unique())
# selected_month = st.sidebar.selectbox('Select Month', months)
# selected_data = data[data['month'] == selected_month]
# num_tweets = selected_data.shape[0]
# num_replies = selected_data['replies'].sum()
# st.write('## Number of Tweets for Selected Month:', num_tweets)
# st.write('## Number of Replies for Selected Month:', num_replies)
# st.write('## Number of Replies for Each Tweet')
# sns.countplot(x='replies', data=selected_data)
# plt.xlabel('Number of Replies')
# plt.ylabel('Count')
# st.pyplot()
# st.write('## Scatter Plot: Engagements vs Impressions')
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='impressions', y='engagements', data=selected_data)
# plt.xlabel('Impressions')
# plt.ylabel('Engagements')
# st.pyplot()
# st.write('Line Graph: Retweets vs Likes')
# plt.figure(figsize=(10, 6))
# plt.plot(selected_data['retweets'], label='Retweets')
# plt.plot(selected_data['likes'], label='Likes')
# plt.xlabel('Index')
# plt.ylabel('Count')
# plt.title('Retweets vs Likes')
# plt.legend()
# st.pyplot()
# st.write('## Line Graph: Engagements over Time')
# plt.figure(figsize=(10, 6))
# selected_data.set_index('time')['engagements'].plot()
# plt.xlabel('Time')
# plt.ylabel('Engagements')
# plt.title('Engagements over Time')
# st.pyplot()
# analyzer = SentimentIntensityAnalyzer()
# monthly_sentiments = {}
# for month in data['month'].unique():
#     month_data = data[data['month'] == month]
#     sentiments = []
#     for tweet, replies in zip(month_data['Tweet'], month_data['replies']):
#         sentiment_score = analyzer.polarity_scores(tweet)
#         if sentiment_score['compound'] >= 0.05 or replies == 0:
#             sentiments.append('Positive')
#         elif sentiment_score['compound'] <= -0.05 or replies == 1:
#             sentiments.append('Negative')
#         else:
#             sentiments.append('Neutral')
#     monthly_sentiments[month] = sentiments
# st.write(f'## Distribution of Sentiment Labels for Month {selected_month}')
# plt.figure(figsize=(8, 6))
# sentiment_counts = pd.Series(monthly_sentiments[selected_month]).value_counts()
# sentiment_counts.plot(kind='bar')
# plt.xlabel('Sentiment')
# plt.ylabel('Count')
# plt.title(f'Distribution of Sentiment Labels for Month {selected_month}')
# st.pyplot()
# all_words = " ".join(data.loc[data['replies'] == 0, 'Tweet'])
# wordcloud = WordCloud(width=600, height=400, random_state=35, max_font_size=50,background_color='white').generate(all_words)
# # Display word cloud for negative tweets
# st.write('## Word Cloud for Negative Tweets')
# plt.figure(figsize=(10, 6))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot()
# all_words = " ".join(data.loc[data['replies'] == 0, 'Tweet'])
# #from wordcloud import WordCloud
# wordcloud = WordCloud(width=600, height=400, random_state=35, max_font_size=50,background_color='white').generate(all_words)
# st.write('## Word Cloud for Positive Tweets')
# plt.figure(figsize=(10, 6))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot()
#

!npm install localtunnel

!streamlit run app.py &>/content/logs.txt & npx localtunnel --port 8501 & curl ipv4.icanhazip.com

#consider each word as a token
single_tweet = data['retweet'].apply(lambda x: x.split())
single_tweet.head()

#steeming is the important in NLP, it reduces the vocabulary size.
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
single_tweet = single_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
single_tweet.head()

for i in range(len(single_tweet)):
    single_tweet[i] = " ".join(single_tweet[i])

data['retweet'] = single_tweet
data.head()

#remove all the short words
data['retweet'] = data['retweet'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))
data.head()

#visualise all the words that has positive tweet and negative tweet
#!pip install wordcloud
#positive tweets
all_words = " ".join(data.loc[data['replies'] == 1, 'retweet'])
from wordcloud import WordCloud
wordcloud = WordCloud(width=600, height=400, random_state=35, max_font_size=50,background_color='white').generate(all_words)
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#frequent words for negative tweets
all_words = " ".join(data.loc[data['replies'] == 0, 'retweet'])
#from wordcloud import WordCloud
wordcloud = WordCloud(width=600, height=400, random_state=35, max_font_size=50,background_color='white').generate(all_words)
#plt.title('negatuve')
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

positive = data[data['replies'] == 1]['retweet']
negative = data[data['replies'] == 0]['retweet']
plt.bar(['Negative', 'Positive'], [len(negative), len(positive)], color=['red','blue'])
plt.xlabel('Responses')
plt.ylabel('count Tweets')
plt.title('Negative and Positive Tweets')
plt.show()

#extract the words and split the dataset based on word vectors.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
vector_of_words = CountVectorizer(max_df=0.90, min_df=3, max_features=500, stop_words='english')
words = vector_of_words.fit_transform(data['retweet'])
x_train, x_test, y_train, y_test = train_test_split(words, data['labels'], random_state=42, test_size=0.05)
#len(x_train)
# calculate accuracy using SVM method
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
SVCmodel = LinearSVC()
SVCmodel.fit(x_train, y_train)
#model_Evaluate(SVCmodel)
pred = SVCmodel.predict(x_test)
acc=accuracy_score(y_test,pred)
acc

# confusion matrix
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_test, pred)
labels = np.asarray(group_percentages).reshape(2, 2)
sns.heatmap(conf, annot=labels, cmap='bwr', fmt='',
            xticklabels=categories, yticklabels=categories)

plt.xlabel("Predicted values", fontdict={'size': 20}, labelpad=10)
plt.ylabel("Actual values", fontdict={'size': 20}, labelpad=10)
plt.title("Confusion Matrix", fontdict={'size': 20}, pad=20)

plt.show()

!pip install streamlit

!pip install vaderSentiment

# Commented out IPython magic to ensure Python compatibility.
# !pip install streamlit
# !pip install vaderSentiment
# %%writefile app.py
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from wordcloud import WordCloud
# data= pd.read_csv('SocialMedia.csv')
# st.set_option('deprecation.showPyplotGlobalUse', False)
# data['time'] = pd.to_datetime(data['time'],errors='coerce')
# data = data.dropna(subset=['time'])
# # Extract month from the 'time' column
# data['month'] = data['time'].dt.month
# months = sorted(data['month'].unique())
# selected_month = st.sidebar.selectbox('Select Month', months)
# selected_data = data[data['month'] == selected_month]
# num_tweets = selected_data.shape[0]
# num_replies = selected_data['replies'].sum()
# st.write('## Number of Tweets for Selected Month:', num_tweets)
# st.write('## Number of Replies for Selected Month:', num_replies)
# st.write('## Number of Replies for Each Tweet')
# sns.countplot(x='replies', data=selected_data)
# plt.xlabel('Number of Replies')
# plt.ylabel('Count')
# st.pyplot()
# st.write('## Scatter Plot: Engagements vs Impressions')
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='impressions', y='engagements', data=selected_data)
# plt.xlabel('Impressions')
# plt.ylabel('Engagements')
# st.pyplot()
# st.write('Line Graph: Retweets vs Likes')
# plt.figure(figsize=(10, 6))
# plt.plot(selected_data['retweets'], label='Retweets')
# plt.plot(selected_data['likes'], label='Likes')
# plt.xlabel('Index')
# plt.ylabel('Count')
# plt.title('Retweets vs Likes')
# plt.legend()
# st.pyplot()
# st.write('## Line Graph: Engagements over Time')
# plt.figure(figsize=(10, 6))
# selected_data.set_index('time')['engagements'].plot()
# plt.xlabel('Time')
# plt.ylabel('Engagements')
# plt.title('Engagements over Time')
# st.pyplot()
# analyzer = SentimentIntensityAnalyzer()
# monthly_sentiments = {}
# for month in data['month'].unique():
#     month_data = data[data['month'] == month]
#     sentiments = []
#     for tweet, replies in zip(month_data['Tweet'], month_data['replies']):
#         sentiment_score = analyzer.polarity_scores(tweet)
#         if sentiment_score['compound'] >= 0.05 or replies == 0:
#             sentiments.append('Positive')
#         elif sentiment_score['compound'] <= -0.05 or replies == 1:
#             sentiments.append('Negative')
#         else:
#             sentiments.append('Neutral')
#     monthly_sentiments[month] = sentiments
# st.write(f'## Distribution of Sentiment Labels for Month {selected_month}')
# plt.figure(figsize=(8, 6))
# sentiment_counts = pd.Series(monthly_sentiments[selected_month]).value_counts()
# sentiment_counts.plot(kind='bar')
# plt.xlabel('Sentiment')
# plt.ylabel('Count')
# plt.title(f'Distribution of Sentiment Labels for Month {selected_month}')
# st.pyplot()
# all_words = " ".join(data.loc[data['replies'] == 0, 'Tweet'])
# wordcloud = WordCloud(width=600, height=400, random_state=35, max_font_size=50,background_color='white').generate(all_words)
# # Display word cloud for negative tweets
# st.write('## Word Cloud for Negative Tweets')
# plt.figure(figsize=(10, 6))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot()
# all_words = " ".join(data.loc[data['replies'] == 0, 'Tweet'])
# #from wordcloud import WordCloud
# wordcloud = WordCloud(width=600, height=400, random_state=35, max_font_size=50,background_color='white').generate(all_words)
# st.write('## Word Cloud for Positive Tweets')
# plt.figure(figsize=(10, 6))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot()
# 
# 
#

!npm install localtunnel

!streamlit run app.py &>/content/logs.txt & npx localtunnel --port 8501 & curl ipv4.icanhazip.com
