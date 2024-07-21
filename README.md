# Social Media Sentiment Analysis

## Project Description

The goal of this project is to build a model that accurately classifies the sentiment expressed in social media text as positive, negative, or neutral. Utilizing Natural Language Processing (NLP) techniques and machine learning algorithms, this project performs sentiment classification on a dataset of social media text data.

## Dataset

The project uses a sentiment dataset, presumably named `sentimentdataset.csv`. The dataset details, including source and features, are assumed to be documented separately.

## Libraries Used

The following Python libraries are employed in this project:

- `re`
- `pandas`
- `numpy`
- `nltk`
- `sklearn`
- `matplotlib`
- `seaborn`
- `wordcloud`
- `pickle`

## Preprocessing

The raw text data undergoes several preprocessing steps:

1. **Categorizing Sentiments:**
   - Defines lists of positive, negative, and neutral words.
   - Assigns a sentiment category ("Positive", "Negative", or "Neutral") to each text based on the presence of these words.

2. **Text Cleaning and Lemmatization:**
   - Converts text to lowercase.
   - Removes special characters, URLs, and mentions.
   - Removes stop words (common words like "a", "the", etc.).
   - Lemmatizes words (reduces words to their base form).

3. **Sentiment Intensity Analysis (VADER):**
   - Uses NLTK's VADER lexicon to calculate sentiment scores for each text.
   - Stores the compound sentiment score in the 'VADER_Sentiment' column.

4. **Data Preparation:**
   - Maps sentiment categories to numerical labels (Positive: 0, Negative: 1, Neutral: 2).
   - Splits the data into features (preprocessed text) and labels (sentiment categories).

## Feature Extraction

- **TF-IDF Vectorization:** Converts the preprocessed text data into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency) to represent the importance of words in the corpus.

## Model Training and Evaluation

The project trains and evaluates two classification models:

1. **Logistic Regression:**
   - Trains a Logistic Regression model on the training data.
   - Evaluates the model using F1-score and cross-validation.
   - Visualizes the confusion matrix to analyze classification performance.

2. **Support Vector Machine (SVM):**
   - Trains an SVM model with a linear kernel.
   - Performs similar evaluation steps as the Logistic Regression model, including F1-score, cross-validation, and confusion matrix visualization.

## Visualization

The project includes visualizations for:

- **Sentiment Distribution:** A count plot showing the distribution of sentiments in the dataset.
- **Confusion Matrices:** Heatmaps representing the confusion matrices for both Logistic Regression and SVM models.
- **Cross-Validation Scores:** A line plot comparing the cross-validation F1 scores of the two models across different folds.
- **Word Clouds:** Visualizations of the most frequent words in positive, negative, and neutral sentiment categories.

## Model Persistence

- The trained Logistic Regression model and the TF-IDF vectorizer are saved to disk using `pickle` for future use without retraining.

## Fork this to further implement the following:

1. **Explore Other Classification Algorithms:** Experiment with different machine learning models such as Naive Bayes, Random Forest, or deep learning models for potentially better performance.
2. **Hyperparameter Tuning:** Fine-tune the models by optimizing their hyperparameters using techniques like grid search or randomized search.
3. **Ensemble Methods:** Combine the predictions of multiple models to potentially improve overall accuracy and robustness.
4. **Real-Time Sentiment Analysis:** Explore integrating the model into a real-time system for monitoring and analyzing social media streams.
