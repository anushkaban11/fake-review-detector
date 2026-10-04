import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import nltk
nltk.download('stopwords')

# Sample data - Kaggle se bada dataset le lena
data = {
    'review': [
        'This product is amazing, best purchase ever!!!',
        'Worst product, total waste of money',
        'I love this, highly recommend everyone',
        'Fake review!!! Amazing amazing amazing buy now!!!',
        'Product is ok ok, delivery was late',
        'BEST BEST BEST 5 STAR MUST BUY FAKE PRODUCT'
    ],
    'label': [0, 0, 0, 1, 0, 1] # 0=Real, 1=Fake
}
df = pd.DataFrame(data)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['review'])
y = df['label']

# Model
model = LogisticRegression()
model.fit(X, y)

# Save
joblib.dump(model, 'fake_review_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model ready! fake_review_model.pkl ban gaya")
