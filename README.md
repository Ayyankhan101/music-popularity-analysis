# Music Popularity Analysis

## Project Overview

This project analyzes a dataset of songs to uncover insights into the characteristics of popular music. It includes an exploratory data analysis (EDA), a predictive model, and an interactive Streamlit dashboard to visualize the findings.

## Features

*   **Exploratory Data Analysis:** In-depth analysis of audio features, genres, and song popularity.
*   **Predictive Modeling:** A linear regression model to predict song popularity based on audio features.
*   **Interactive Dashboard:** A Streamlit application to visualize the data and model insights.

## Dataset

The dataset used in this project is `dataset.csv`, which contains various audio features and metadata for a large collection of songs. The dataset is compressed in `dataset.zip`.

## Getting Started

### Prerequisites

*   Python 3.7+
*   pip

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/music-popularity-analysis.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd music-popularity-analysis
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To explore the data and the model, you can use the Jupyter Notebook or the Streamlit dashboard.

#### Jupyter Notebook

The `project_music.ipynb` notebook contains the full analysis, from data cleaning to model building. To use it, start Jupyter Lab:

```bash
jupyter lab
```

Then, open the `project_music.ipynb` file.

#### Streamlit Dashboard

The Streamlit dashboard provides an interactive way to explore the data. To run it, use the following command:

```bash
streamlit run app.py
```

This will open the dashboard in your web browser.

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

