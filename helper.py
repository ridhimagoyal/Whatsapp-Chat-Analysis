
from urlextract import URLExtract
import pandas as pd
from wordcloud import WordCloud
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

extract = URLExtract()


# Modify fetch_stats to include sentiment analysis stats and topic modeling stats
def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Fetch the number of messages
    num_messages = df.shape[0]

    # Fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # Fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # Fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    # Sentiment analysis stats
    sentiments, positive_percent, negative_percent, neutral_percent = sentiment_analysis(df)

    # Topic Modeling Stats (LDA)
    topics = extract_topics(df)

    return num_messages, len(words), num_media_messages, len(links), sentiments, positive_percent, negative_percent, neutral_percent, topics


def extract_topics(df):
    # Preparing the text for topic extraction
    stop_words = stopwords.words('english')
    text_data = df['message'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

    # TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=0.85, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)

    # LDA Model
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_topics = lda.fit_transform(tfidf_matrix)

    # Extracting top words for each topic
    feature_names = tfidf_vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        topics.append(f"Topic {topic_idx + 1}: " + ", ".join(topic_words))

    return topics


def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x, df


def create_wordcloud(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc


def most_common_words(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df


def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap


# Sentiment analysis function
def sentiment_analysis(df):
    sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}

    for message in df['message']:
        analysis = TextBlob(message)
        # Get the sentiment polarity
        polarity = analysis.sentiment.polarity

        # Classify sentiment based on polarity
        if polarity > 0:
            sentiments['positive'] += 1
        elif polarity < 0:
            sentiments['negative'] += 1
        else:
            sentiments['neutral'] += 1

    total_messages = df.shape[0]
    positive_percent = (sentiments['positive'] / total_messages) * 100
    negative_percent = (sentiments['negative'] / total_messages) * 100
    neutral_percent = (sentiments['neutral'] / total_messages) * 100

    return sentiments, positive_percent, negative_percent, neutral_percent
