import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import zipfile

# Set plot style and color palette
sns.set_style('whitegrid')
sns.set_palette('pastel')

st.set_page_config(page_title="Music Popularity Analysis", layout="wide")

# --- Data Loading and Caching ---
@st.cache_data
def load_data():
    try:
        with zipfile.ZipFile('dataset.zip') as z:
            with z.open('dataset.csv') as f:
                df = pd.read_csv(f, encoding='latin1', on_bad_lines='skip')
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        df.dropna(inplace=True)
        df.drop_duplicates(subset=['track_id'], inplace=True)
        return df
    except FileNotFoundError:
        st.error("The 'dataset.zip' file was not found. Please make sure it is in the same directory as the app.")
        return None

df = load_data()

if df is None:
    st.stop()

# --- Sidebar ---
st.sidebar.title("Options")


# Genre Filter
all_genres = sorted(df['track_genre'].unique())
selected_genres = st.sidebar.multiselect("Select Genres", all_genres, default=all_genres[:5])

# Top N Genres
top_n_genres = st.sidebar.slider("Number of Top Genres to Display", 5, 20, 10)

# --- Main Content ---
st.title("Music Popularity Analysis")
st.markdown("An interactive dashboard to explore the characteristics of popular music.")

# --- Raw Data Preview ---
st.header("Raw Data Preview")
st.dataframe(df.head(100))


# Filtered Data
if selected_genres:
    filtered_df = df[df['track_genre'].isin(selected_genres)]
else:
    filtered_df = df

# --- EDA Section ---
st.header("Exploratory Data Analysis")


st.subheader("Distribution of Song Popularity")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(filtered_df['popularity'], bins=30, kde=True, ax=ax, color='green')
ax.set_title(f'Popularity Distribution for Selected Genres')
ax.set_xlabel('Popularity')
ax.set_ylabel('Frequency')
st.pyplot(fig)


st.subheader(f"Top {top_n_genres} Most Popular Genres")
top_genres = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False).head(top_n_genres)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis', ax=ax)
ax.set_title(f'Top {top_n_genres} Most Popular Genres')
ax.set_xlabel('Average Popularity')
ax.set_ylabel('Genre')
st.pyplot(fig)

st.subheader("Correlation Matrix of Audio Features")
audio_features = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
correlation_matrix = filtered_df[audio_features].corr()
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
ax.set_title('Correlation Matrix for Selected Genres')
st.pyplot(fig)

# --- Predictive Modeling Section ---
st.header("Predictive Modeling")

with st.expander("Train a Popularity Prediction Model"):
    features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    X = filtered_df[features]
    y = filtered_df['popularity']

    if len(filtered_df) > 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("### Model Performance")
        st.write(f'**Mean Squared Error:** {mse:.2f}')
        st.write(f'**R-squared:** {r2:.2f}')
        st.info("This model predicts song popularity based on audio features for the selected genres.")
    else:
        st.warning("Not enough data to train a model for the selected genres.")

# --- Data Preview ---
st.header("Data Preview")
st.dataframe(filtered_df.head(100))


# --- Key Insights ---
st.header("Key Insights")
st.markdown("""
- **Popularity Distribution:** The popularity of songs varies across genres, with some genres having a higher average popularity.
- **Top Genres:** Pop, rock, and dance-related genres consistently rank among the most popular.
- **Feature Correlations:** Loudness and energy are strong indicators of popularity, while acousticness is negatively correlated.
- **Predictive Power:** While audio features provide some predictive signal, they are not the only factors determining a song's success.
""")