# Music Data Analysis

## Project Overview

This project performs an in-depth exploratory data analysis (EDA) on a dataset of songs. The goal is to uncover insights into the characteristics of popular music, understand the relationships between different audio features, and build a predictive model to estimate a song's popularity based on these features.

## Dataset

The dataset, `dataset.csv`, contains various audio features and metadata for a large collection of songs. Key columns include:

*   `track_name`: The name of the song.
*   `artists`: The artist(s) of the song.
*   `popularity`: A score from 0 to 100, indicating the song's popularity.
*   `track_genre`: The genre of the track.
*   Audio features like `danceability`, `energy`, `loudness`, `acousticness`, `instrumentalness`, `valence`, and `tempo`.

## Installation

To run this analysis, you need Python and the following libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

The entire analysis is contained within the `project_music.ipynb` Jupyter Notebook. To view and run the analysis, start Jupyter Lab:

```bash
jupyter lab
```

Then, open the `project_music.ipynb` file.

## Analysis Summary

The analysis covers three main stages:

1.  **Data Cleaning:** The raw data is loaded, and necessary cleaning steps are performed, such as removing duplicates, handling missing values, and dropping irrelevant columns.
2.  **Exploratory Data Analysis (EDA):** This stage involves a deep dive into the data using visualizations to understand distributions, relationships, and trends. We analyze song popularity, genre characteristics, and correlations between audio features.
3.  **Predictive Modeling:** A simple machine learning model is trained to predict song popularity based on its audio features. The model's performance is evaluated to determine how well these features can explain popularity.

## Key Findings

*   **Popularity:** The most popular songs tend to have higher `energy` and `loudness`.
*   **Genre Insights:** Genres like 'pop', 'rock', and 'dance' have the highest average popularity scores.
*   **Feature Correlation:** `energy` and `loudness` are strongly positively correlated, while `acousticness` and `energy` are strongly negatively correlated.
*   **Modeling:** The predictive model shows that while audio features have some predictive power, they don't fully explain popularity, suggesting that other factors (like artist fame, marketing, etc.) are also highly influential.
# music-popularity-analysis
