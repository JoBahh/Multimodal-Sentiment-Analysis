# Multimodal Sentiment Analysis of Amazon Reviews

This project uses Amazon Toys and Games reviews to predict customer sentiment using both review text and product images. I built a text-only baseline model first, then built a multimodal deep learning model that combines text and image features.

The goal was to see how well review sentiment could be predicted from different types of data and whether using both text and images could help classify reviews as negative, neutral, or positive.

## Contributors

- Joseph Barker

## Dataset

- **Category:** Amazon Toys and Games reviews
- **Original format:** Compressed JSON Lines file
- **Final cleaned sample:** 29,313 reviews
- **Data included:**
  - Review text
  - Product rating
  - Associated image URLs
- **Language:** Python

Reviews were filtered to keep entries with review text, ratings, and at least one image. Reviews shorter than 20 characters were removed, and language detection was used to keep English reviews only.

The final cleaned dataset contained 29,313 reviews with three columns:

- `text`
- `rating`
- `images`

> The dataset is not included in this repository because of file-size limits. The notebook expects a cleaned CSV file named `toysgames30kfinal.csv`.

## Sentiment Labels

Customer ratings were converted into three sentiment classes:

- **Negative**
- **Neutral**
- **Positive**

These labels were used as the target variable for the classification models.

## Data Preparation

- Loaded Toys and Games reviews from a compressed JSON Lines file.
- Kept reviews with text, ratings, and image data.
- Removed reviews shorter than 20 characters.
- Used language detection to filter for English reviews.
- Randomly sampled up to 30,000 reviews before cleaning.
- Created a cleaned dataset with 29,313 reviews.
- Created train/test splits using stratified sampling.
- Prepared text data through tokenization, vocabulary creation, and padding.
- Downloaded and transformed review images from their URLs.
- Resized images to 224 × 224 and normalized them for the CNN.

## Exploratory Analysis

The notebook includes:

- Dataset shape and column checks
- Rating distribution analysis
- Review text exploration
- A word cloud of common review terms after stop-word removal
- Inspection of review text, ratings, and image information

## Text-Only Baseline Model

The first model used review text only.

### Model

- TF-IDF Vectorizer
- Unigrams and bigrams
- Logistic Regression
- Balanced class weights
- 70/30 stratified train/test split

### Text-Only Results

The text-only model achieved:

- **Accuracy:** 76.37%
- **Weighted F1-score:** 0.77
- **Test set size:** 8,794 reviews

| Sentiment | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negative | 0.73 | 0.79 | 0.76 |
| Neutral | 0.46 | 0.52 | 0.49 |
| Positive | 0.89 | 0.83 | 0.86 |

The text-only model performed best on positive reviews and had the most difficulty with neutral reviews.

## Multimodal Model

The multimodal model combines review text and image features.

### Text Component

The text branch uses:

- Tokenized review text
- Learned word embeddings
- LSTM network
- Padded sequences with a maximum length of 100 tokens

### Image Component

The image branch uses:

- Review/product image URLs
- Image downloading with error handling
- Image resizing and normalization
- Pretrained ResNet-18 CNN feature extractor
- A fully connected layer to map image features into the shared feature space

### Fusion Method

Text and image features are combined using a learned gating layer. The gate learns how much information to use from the text features and image features before making the final sentiment prediction.

### Training Setup

- Framework: PyTorch
- Optimizer: Adam
- Loss function: Cross-Entropy Loss
- Training epochs: 5
- Train/validation split: 80/20 stratified split
- Batch size: 32
- Best model saved as `best_model.pt`

## Multimodal Results

The multimodal model improved during training:

| Epoch | Validation Accuracy |
|---|---:|
| 1 | 59.07% |
| 2 | 59.07% |
| 3 | 59.07% |
| 4 | 71.50% |
| 5 | 75.52% |

Final evaluation on 5,863 validation reviews:

- **Accuracy:** 76%
- **Weighted F1-score:** 0.71

| Sentiment | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negative | 0.68 | 0.80 | 0.74 |
| Neutral | 0.47 | 0.13 | 0.20 |
| Positive | 0.80 | 0.94 | 0.87 |

The multimodal model performed strongly on positive reviews and reasonably well on negative reviews. Neutral reviews were still the hardest class to identify, especially in the multimodal model.

## Key Findings

- The text-only logistic regression baseline reached 76.37% accuracy.
- The multimodal text-and-image model reached 76% final validation accuracy.
- Positive reviews were the easiest class for both models to classify.
- Neutral reviews were the most difficult class for both models.
- The multimodal model improved significantly over its five training epochs, increasing validation accuracy from 59.07% to 75.52%.
- Review text was highly useful for sentiment prediction, while the image pipeline added a second source of information for the classification task.

## Tools Used

- Python
- Google Colab
- pandas
- NumPy
- scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- PyTorch
- torchvision
- ResNet-18
- LSTM
- PIL
- matplotlib
- seaborn
- WordCloud
- langdetect

## Limitations

- The dataset only contains Amazon Toys and Games reviews, so results may not generalize to other categories.
- Sentiment labels are based on star ratings, which may not always perfectly match the text of a review.
- Some image URLs may fail to download, which causes the model to use a fallback blank image.
- Neutral reviews were much harder to classify than positive or negative reviews.
- The multimodal model was trained for only five epochs, so additional tuning could improve performance.
- Image content may not always directly reflect the reviewer’s sentiment.

## Conclusion

This project shows how text and image data can be combined for sentiment analysis. The text-only baseline reached 76.37% accuracy, while the multimodal model reached 76% accuracy after five training epochs. Both models did well on positive reviews, while neutral reviews remained the hardest group to classify.

The project demonstrates the full machine learning workflow: data cleaning, exploratory analysis, text vectorization, image preprocessing, deep learning, multimodal feature fusion, training, and model evaluation.
