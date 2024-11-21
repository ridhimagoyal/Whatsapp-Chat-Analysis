
import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np

nltk.download('punkt')


def generate_tfidf_summary(text, num_sentences=5):

    sentences = nltk.sent_tokenize(text)

    tfidf = TfidfVectorizer(stop_words='english')

    tfidf_matrix = tfidf.fit_transform(sentences)

    sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)

    ranked_sentences = [sentences[i] for i in sentence_scores.argsort()[-num_sentences:]]

    return ' '.join(ranked_sentences)


st.sidebar.title("Whatsapp Chat Analyzer")

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "txt"])
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")

    # Preprocess data
    try:
        df = preprocessor.preprocess(data)
    except Exception as e:
        st.error(f"Error in data preprocessing: {e}")
        st.stop()


    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Analyze"):

        st.title("WhatsApp Chat Analysis")

        stats = helper.fetch_stats(selected_user, df)

        st.write("Returned stats:", stats)

        if len(stats) == 9:
            num_messages, words, num_media, num_links, sentiments, positive_percent, negative_percent, neutral_percent, topics = stats
        else:
            st.error(f"Unexpected number of values returned by fetch_stats. Expected 9, but got {len(stats)}.")
            st.stop()

        st.header("Overall Stats")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Messages", num_messages)
        col2.metric("Words", words)
        col3.metric("Media Shared", num_media)
        col4.metric("Links Shared", num_links)


        # # Sentiment Analysis
        st.title("Sentiment Analysis")
        st.subheader("Sentiment Distribution")
        col1, col2, col3 = st.columns(3)
        col1.metric("Positive", f"{positive_percent:.2f}%")
        col2.metric("Negative", f"{negative_percent:.2f}%")
        col3.metric("Neutral", f"{neutral_percent:.2f}%")

        # Sentiment bar chart
        fig, ax = plt.subplots()
        sentiment_labels = ['Positive', 'Negative', 'Neutral']
        sentiment_values = [sentiments['positive'], sentiments['negative'], sentiments['neutral']]
        ax.bar(sentiment_labels, sentiment_values, color=['green', 'red', 'gray'])
        st.pyplot(fig)

        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # Word Cloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title('Most common words')
        st.pyplot(fig)

        # Monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='blue')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='blue')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # TF-IDF Summary Generation
        st.title("Chat Summary")
        if selected_user == 'Overall':
            chat_text = ' '.join(df['message'].dropna().tolist())
            summary = generate_tfidf_summary(chat_text, num_sentences=1)
            st.subheader("TF-IDF Summary")
            st.write(summary)

else:
    st.info("Please upload a file to get started.")

# Download processed data option
if uploaded_file is not None:
    st.sidebar.download_button(
        label="Download Processed Data as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='processed_whatsapp_data.csv',
        mime='text/csv'
    )
