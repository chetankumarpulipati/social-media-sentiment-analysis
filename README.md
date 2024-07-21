# E-commerce Product Recommendation System

## Project Description

This project implements a simple product recommendation system for an e-commerce platform using collaborative filtering with TensorFlow/Keras. The system suggests products to users based on their past purchase history by learning user and product embeddings to predict potential product interactions.

## Dataset

The project utilizes an e-commerce dataset (name and source to be specified if available). The dataset contains the following information:

- **User ID:** Unique identifier for each user.
- **Product ID:** Unique identifier for each product.
- **Event Type:** User actions such as 'view', 'cart', 'purchase', 'remove_from_cart'.
- **Brand:** Brand of the product.
- **Price:** Price of the product.

## Preprocessing

The following data preprocessing steps are performed:

1. **Data Cleaning:**
   - Removed irrelevant columns like 'event_time', 'user_session', 'category_id', and 'category_code'.
   - Filled missing values in the 'brand' column with "Unknown".
   - Dropped rows with missing 'user_id'.

2. **Data Encoding:**
   - Mapped 'event_type' to numerical values (e.g., 'view': 0, 'cart': 1, 'purchase': 2).
   - Converted 'user_id' and 'product_id' to integer format.
   - Encoded user and product IDs into numerical indices using `user2user_encoded` and `product2product_encoded` dictionaries.

3. **Data Splitting:**
   - Split the preprocessed interaction data into training and testing sets (80% train, 20% test).

## Model Building

The recommendation model is a neural network implemented in Keras with the following architecture:

- **Input Layers:** Two input layers for user IDs and product IDs.
- **Embedding Layers:** User and product IDs are passed through embedding layers to learn dense vector representations.
- **Concatenation:** User and product embeddings are concatenated.
- **Dense Layers:** The concatenated vector is fed through a dense layer with ReLU activation, followed by an output dense layer with sigmoid activation.

The model is compiled using the Adam optimizer and binary cross-entropy loss, aiming to predict the likelihood of a user interacting with a product.

## Training and Evaluation

- **Training:** The model is trained for 5 epochs using the training data. Training history is stored for potential analysis of loss and accuracy over epochs.
- **Evaluation:** The current script does not include evaluation on the test set, which is crucial for assessing the model's generalization ability.

## Recommendation Generation

The `get_recommendations` function provides product recommendations for a given user:

1. **Filtering:** Identifies products the user hasn't interacted with yet.
2. **Prediction:** The model predicts the user's preference for non-interacted products.
3. **Ranking:** Selects the top N predicted products with the highest scores as recommendations.

**Output:** Recommendations for a sample of users are written to a CSV file named `recommendations.csv`, containing user IDs and their corresponding recommended product IDs.

## Fork this to implement the following:

1. **Evaluation:** Implement evaluation metrics (e.g., precision, recall, NDCG) on the test set to assess model performance.
2. **Hyperparameter Tuning:** Experiment with different embedding sizes, dense layer architectures, and optimizers to improve accuracy.
3. **Data Exploration:** Analyze user behavior and product trends in the dataset for feature engineering or model enhancements.
4. **Cold Start Problem:** Address the cold start problem for new users or products with limited interaction history.
