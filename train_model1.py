import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from joblib import dump
from sklearn.model_selection import train_test_split


df = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\project\Spam-Email-Detection\Data Source\spam.csv")

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df['Category'] = df['Category'].map({'spam': 0, 'ham': 1})

X = df['Message']
y = df['Category']

# TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
X_vect = tfidf.fit_transform(X)

# Split (optional)
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Classifiers
svm_clf = SVC(probability=True)
logreg_clf = LogisticRegression(max_iter=1000)
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('svm', svm_clf),
        ('logreg', logreg_clf),
        ('xgb', xgb_clf)
    ],
    voting='soft'
)

# Train ensemble
voting_clf.fit(X_train, y_train)

# Save model and vectorizer
dump(voting_clf, 'model.pkl')
dump(tfidf, 'tfidf.pkl')

print("Ensemble model saved with SVM, Logistic Regression, and XGBoost!")
