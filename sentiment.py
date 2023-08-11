import pandas as pd
import numpy as np
import seaborn as sns
import nltk
import streamlit as st


def overalSentiment(df):
    st.write('') 
    st.write('') 
    import plotly.graph_objs as go
    import pandas as pd

    # Calculate the total sentiment count
    total_sentiments = len(df)
    # Calculate the count of each sentiment category
    sentiment_counts = df['Sentiment'].value_counts()
    positive_count = sentiment_counts.get('Positive',0)
    negative_count = sentiment_counts.get('Negative', 0)
    neutral_count = sentiment_counts.get('Neutral', 0)

    col1, col2, col3,col4 = st.columns(4)
    col1.subheader(total_sentiments)
    col1.write("Total Sentiments")
    col2.markdown(f'<h3 style="color:green">{positive_count}</h3>', unsafe_allow_html=True)
    col2.write("Positive")
    col3.markdown(f'<h3 style="color:red">{negative_count}</h3>', unsafe_allow_html=True)
    col3.write("Negative")
    col4.markdown(f'<h3 style="color:blue">{neutral_count}</h3>', unsafe_allow_html=True)
    col4.write("Neutral")


    # Group the data by 'MonthsFromReview' and 'Sentiment' and count the occurrences
    grouped_data = df.groupby(['Date of Review', 'Sentiment']).size().unstack().fillna(0)

    # Calculate the cumulative sum of sentiment counts
    cumulative_data = grouped_data.cumsum()

    # Create a Plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=cumulative_data.index, y=cumulative_data['Positive'], name='Positive ', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=cumulative_data.index, y=cumulative_data['Negative'], name='Negative', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=cumulative_data.index, y=cumulative_data['Neutral'], name='Neutral', line=dict(color='blue')))


    fig.update_layout(
        xaxis=dict(title='Months from Review', showgrid=True),
        yaxis=dict(title='Sentiment Count', showgrid=True),
        title='Sentiment Trend over Months',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    # Display the Plotly figure using st.plotly_chart()
    config = {'displayModeBar': False}
    st.plotly_chart(fig,config=config)



#This Function Prints Bar and Pie Charts of THe Ratings 
def categorical_variable_summary(df, column_name) :

    from  matplotlib import pyplot as plt
    plt.style.use('ggplot')
    from plotly.subplots import make_subplots
    from  plotly import graph_objs as go
    from plotly.offline import init_notebook_mode, iplot

    constraints = ['#581845','#C70039','#2E4053','#1ABC9C','#7F8C8D']
    fig = make_subplots(rows=1,cols=2,
                        subplot_titles=('Countplot','Percentages'),
                        specs=[[{"type": "xy"}, {'type':'domain'}]])

    fig.add_trace(go.Bar( y = df[column_name].value_counts().values.tolist(), 
                          x = [str(i) for i in df[column_name].value_counts().index], 
                          text = df[column_name].value_counts().values.tolist(),
                          textfont = dict(size=15),
                          name = column_name,
                          textposition = 'auto',
                          showlegend=False,
                          marker=dict(color = constraints,
                                      line=dict(color='#DBE6EC',
                                                width=1))),
                  row = 1, col = 1)
    
    fig.add_trace(go.Pie(labels= df[column_name].value_counts().keys(),
                         values= df[column_name].value_counts().values,
                         textfont = dict(size = 20),
                         textposition='auto',
                         showlegend = False,
                         name = column_name,
                         marker=dict(colors=constraints)),
                  row = 1, col = 2)
    
    fig.update_layout(title={'text': 'Given Ratings',
                             'y':0.9,
                             'x':0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                      template='plotly_white')
    config = {'displayModeBar': False}
    st.plotly_chart(fig,config=config)




def vader(df) : 

    import nltk
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    # define a function to classify the sentiment label and compound score for each review
    def classify_sentiment(review):
        scores = analyzer.polarity_scores(review)
        if scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        return pd.Series({'Sentiment': sentiment, 'Compound': scores['compound']})


    # apply the classify_sentiment function to each review in the 'Reviews' column
    df[['Sentiment', 'Compound']] = df['Reviews'].apply(classify_sentiment)


    # Define a function to calculate the sentiment score based on the review and rating
    analyzer = SentimentIntensityAnalyzer()
    def calculate_sentiment_score(review, rating):
        sentiment_score = analyzer.polarity_scores(review)["compound"]
        sentiment_score = sentiment_score * (int(rating) / 5) * 45 + 55
        return sentiment_score


    # Apply the sentiment analysis function to each row in the DataFrame
    df['SentimentScore'] = df.apply(lambda row: calculate_sentiment_score(row['Reviews'], row['Ratings']), axis=1)

    return df



def last18months(df):
    from  matplotlib import pyplot as plt
    import pandas as pd
    from  plotly import graph_objs as go
    import datetime
    current_date = datetime.datetime.now()


    max_score = round(df['SentimentScore'].max(), 2)
    min_score = round(df['SentimentScore'].min(), 2)
    average_score = round(df['SentimentScore'].mean(), 2)

    # col1, col2, col3 = st.columns(3)
    # col1.write(["Average SCore : ",average_score])
    # col2.write(["Maximum ",max_score])
    # col3.write(["Minimum ",max_score])
  
    


    st.write("Sentiment Score:", average_score)
    st.write("Highest :", max_score)
    st.write("Lowest :", min_score)

    # Group the data by 'Date of Review' and calculate the average sentiment score
    df1 = df[df['Date of Review'] <= 18]
    grouped_data = df1.groupby('Date of Review')['SentimentScore'].mean()

    # Create the bar graph
    fig = go.Figure(data=[go.Bar(
        x=grouped_data.index[::-1],
        y=grouped_data.values[::-1],
        marker_color='green',
        width=0.8
    )])
    config = {'displayModeBar': False}

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=grouped_data.index[::-1],
            ticktext=[(current_date - pd.DateOffset(months=int(i))).strftime('%m/%Y') if i != 0 else current_date.strftime('%m/%Y') for i in grouped_data.index[::-1]],
            tickangle=45
        ),
        yaxis=dict(
            range=[10, 100],
            tick0=10,
            dtick=10
        ),
        title='Monthly Score (Limit 24 Months)',
        xaxis_title='Date of Review',
        yaxis_title='Sentiment Score',
    )

    st.plotly_chart(fig,config=config)


def commonPhrases(df) :
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
        
    from collections import Counter

    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.util import ngrams
    import string
    from termcolor import colored

    # Combine all the reviews into a single text corpus
    corpus = ' '.join(df['Reviews'])

    # Preprocess the text corpus
    corpus = corpus.lower()
    corpus = corpus.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(corpus)
    filtered_tokens = [token for token in tokens if token not in stop_words]

    import streamlit as st

    # Generate n-grams from the filtered tokens
    ngrams_dict = {}
    for n in range(3, 7):
        ngrams_list = list(ngrams(filtered_tokens, n))
        ngrams_counter = Counter(ngrams_list)
        if(n==3) : 
            ngrams_dict[n] = {ngram: count for ngram, count in ngrams_counter.items() if count >= 5 and len(set(ngram)) == n}
        elif(n==4) :
            ngrams_dict[n] = {ngram: count for ngram, count in ngrams_counter.items() if count >= 4 and len(set(ngram)) == n}
        else :
            ngrams_dict[n] = {ngram: count for ngram, count in ngrams_counter.items() if count >= 2  and len(set(ngram)) == n}

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    # Radio button selection

    selected_option = st.radio(" **Select phrase length** ", ("3-word", "4-word", "5-word", "6-word"),help=' Most Used Phrases  by the users')

    # Get the selected n-gram dictionary based on the user's choice
    if selected_option == "3-word":
        ngram_dict = ngrams_dict[3]
    elif selected_option == "4-word":
        ngram_dict = ngrams_dict[4]
    elif selected_option == "5-word":
        ngram_dict = ngrams_dict[5]
    elif selected_option == "6-word":
        ngram_dict = ngrams_dict[6]

    # Print the selected word phrases with frequencies
    gram= []
    st.write(f"{selected_option} phrases:")
    for ngram, count in ngram_dict.items():
        ngram_str = ' '.join(ngram)
        ngram_str=ngram_str+' ['+str(count)+']'
        gram.append(ngram_str)
    st.write(gram)



def wordCloud(df) :
        
    from nltk.corpus import stopwords
    from wordcloud import WordCloud,STOPWORDS
    from  matplotlib import pyplot as plt
    import nltk
    import random

    stopwords = set(stopwords.words('english'))

    train_pos = df[df['Sentiment'] == 'Positive']
    train_pos = train_pos['Reviews']
    train_neg = df[df['Sentiment'] == 'Negative']
    train_neg = train_neg['Reviews']

    # randomly select half of the reviews
    train_pos = random.sample(list(train_pos), len(train_pos) // 2)
    train_neg = random.sample(list(train_neg), len(train_neg) // 2)

    # add positive words to stopwords
    pos_words = ['good', 'great', 'excellent', 'awesome', 'love', 'like', 'best', 'fantastic', 'amazing', 'wonderful', 'perfect']
    stopwords.update(pos_words)

    def wordcloud_draw(data, color='black'):
        words = ' '.join(data)
        cleaned_word = " ".join([word for word in words.split()
                                if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                                and word not in stopwords
                                ])
        wordcloud = WordCloud(stopwords=STOPWORDS,
                            background_color=color,
                            width=2500,
                            height=2000
                            ).generate(cleaned_word)
        plt.figure( figsize=(10, 4))
        plt.imshow(wordcloud)
        plt.axis('off')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    st.write("Positive words")
    wordcloud_draw(train_pos, 'white')

    st.write("Negative words")
    wordcloud_draw(train_neg)


def emojiAnalysis(df) :
    import re

    # Calculate the total counts for each emoji
    total_heart_count = df['HeartEmojiCount'].sum()
    total_thumbs_up_count = df['ThumbsUpEmojiCount'].sum()
    total_thumbs_down_count = df['ThumbsDownEmojiCount'].sum()
    total_crying_count = df['CryingEmojiCount'].sum()
    '''
    print(f"Total heart emoji count: {total_heart_count}")
    print(f"Total thumbs-up emoji count: {total_thumbs_up_count}")
    print(f"Total thumbs-down emoji count: {total_thumbs_down_count}")
    print(f"Total crying emoji count: {total_crying_count}")
    '''
  

    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    # Create the emoji labels and counts
    emojis = ['❤️', '👍', '👎', '😢']
    counts = [total_heart_count, total_thumbs_up_count, total_thumbs_down_count, total_crying_count]

   # Create a bar trace
    trace = go.Bar(x=emojis, y=counts, marker_color=['red', 'blue', 'green', 'purple'])

    # Create the layout
    layout = go.Layout(
        title='Emoji Counts',
        xaxis=dict(title='Emoji'),
        yaxis=dict(title='Count'),
    )

    # Create the figure
    fig = go.Figure(data=[trace], layout=layout)

    # Display the figure using Plotly chart in Streamlit
    st.plotly_chart(fig)







        