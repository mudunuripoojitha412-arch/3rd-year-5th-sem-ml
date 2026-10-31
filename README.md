Fake News Detection System
Project Overview
This project uses machine learning to classify news articles as Fake News or True News.
Algorithms Used
1. Random Forest Classifier – combines predictions from multiple decision trees.
2. Logistic Regression – performs probability-based binary classification.
Dataset
The project uses two CSV files:
- Fake.csv – fake news articles (label 0)
- True.csv – real news articles (label 1)
The notebook used a dataset of approximately 44,898 articles. Actual record counts can vary if rows are missing or removed during cleaning.
Technologies and Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
Implementation Workflow
1. Import required libraries.
2. Load Fake.csv and True.csv.
3. Assign labels: 0 = Fake News, 1 = True News.
4. Combine both datasets.
5. Clean missing values and combine title + article text.
6. Split data into 80% training and 20% testing sets.
7. Convert text into numerical features using TF-IDF (max_features=30000).
8. Train Random Forest and Logistic Regression.
9. Evaluate both models using accuracy, classification report, and confusion matrix.
10. Plot Random Forest feature importance and compare model accuracy.
11. Save both trained models and the TF-IDF vectorizer using Joblib.
12. Predict new news articles.
How to Run
1. Install dependencies:
   pip install pandas numpy matplotlib scikit-learn joblib
2. Place True.csv and Fake.csv in the same folder as fake_news_detection.py.
   Alternatively, place an archive.zip containing those two CSV files in that folder.
3. Run:
   python fake_news_detection.py
Saved Model Files
The program creates:
- random_forest_fake_news_model.pkl
- logistic_regression_fake_news_model.pkl
- tfidf_vectorizer.pkl
Keep the TF-IDF vectorizer with the models. New text must be transformed using the same fitted vectorizer before prediction.
Important Notes
- The script calculates accuracy from the current dataset and split. Do not hardcode accuracy values in graphs.
- Results may differ slightly if the dataset, preprocessing, library versions, or train/test split changes.
- A machine-learning prediction is not a guarantee that a news article is factually true. Verify important claims using reliable sources.
