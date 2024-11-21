
import re
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords

def preprocess(data):
    # Regular expression pattern to split the messages by timestamp
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'

    # Split messages and extract dates
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    # Create a DataFrame with messages and corresponding dates
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Convert message_date to datetime
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %H:%M - ')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # If there's a username
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')  # For group notifications
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # Extract date parts
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Period of the day (hour range)
    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))
    df['period'] = period

    # Sentiment analysis
    df['polarity'] = df['message'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['message'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    # Sentiment label categorization
    def get_sentiment_label(polarity):
        if polarity > 0:
            return 'Positive'
        elif polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment_label'] = df['polarity'].apply(get_sentiment_label)

    # Adding message count per user per day for activity summary
    df['message_count'] = 1  # Count 1 message per row

    # Topic Extraction: TF-IDF Keywords
    def extract_keywords(messages, top_n=5):
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(messages)
        feature_names = vectorizer.get_feature_names_out()

        keywords = []
        for row in tfidf_matrix:
            sorted_indices = row.toarray()[0].argsort()[-top_n:][::-1]
            top_keywords = [feature_names[idx] for idx in sorted_indices]
            keywords.append(", ".join(top_keywords))
        return keywords

    df['tfidf_keywords'] = extract_keywords(df['message'])

    # Topic Modeling: LDA
    def extract_topics(messages, n_topics=5):
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(messages)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(tfidf_matrix)

        feature_names = vectorizer.get_feature_names_out()
        topic_keywords = []
        for topic_idx, topic in enumerate(lda.components_):
            top_keywords = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
            topic_keywords.append(", ".join(top_keywords))

        # Assigning dominant topic to each message
        topic_distributions = lda.transform(tfidf_matrix)
        dominant_topics = topic_distributions.argmax(axis=1)
        return dominant_topics, topic_keywords

    df['dominant_topic'], topics = extract_topics(df['message'])
    df['topic_keywords'] = df['dominant_topic'].apply(lambda x: topics[x])

    return df
