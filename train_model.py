"""
train_model.py  —  Greenwashing Detection System (with Ingredient Analysis)
Run this FIRST before running app.py
Command: python train_model.py
"""
 
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report)
import pickle
 
# ── Load dataset ──
data = pd.read_csv("data.csv")
print(f"Dataset loaded  : {len(data)} samples")
print(f"Greenwashing (1): {(data['Label']==1).sum()}")
print(f"Genuine (0)     : {(data['Label']==0).sum()}\n")
 
# ── Preprocessing ──
# Extended stopwords — keep ingredient names intact
STOPWORDS = {
    'is','are','we','our','the','a','an','to','and','of','in','for',
    'on','with','by','this','that','all','from','be','been','was',
    'were','has','have','had','it','its','at','as','or','but','not','so'
}
 
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s%]', ' ', text)
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return ' '.join(tokens)
 
data['clean'] = data['Claim'].apply(preprocess)
 
# ── Train/Test Split ──
X_train, X_test, y_train, y_test = train_test_split(
    data['clean'], data['Label'],
    test_size=0.25,
    random_state=18,
    stratify=data['Label']
)
print(f"Training samples: {len(X_train)}")
print(f"Testing  samples: {len(X_test)}\n")
 
# ── TF-IDF Feature Extraction ──
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=800,
    sublinear_tf=True
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
 
# ── Train Gradient Boosting Classifier ──
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)
 
# ── Evaluate on TEST SET ──
y_pred = model.predict(X_test_vec)
 
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
 
cv_scores = cross_val_score(
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    vectorizer.transform(data['clean']),
    data['Label'],
    cv=5
)
 
print("=" * 50)
print("  Model: Gradient Boosting + TF-IDF")
print("  Now trained on CLAIMS + INGREDIENTS")
print("  Evaluated on unseen TEST SET")
print("=" * 50)
print(f"  Accuracy        : {acc  * 100:.1f}%")
print(f"  Precision       : {prec * 100:.1f}%")
print(f"  Recall          : {rec  * 100:.1f}%")
print(f"  F1 Score        : {f1   * 100:.1f}%")
print(f"  Cross-Val (5-fold): {cv_scores.mean()*100:.1f}% "
      f"+/- {cv_scores.std()*100:.1f}%")
print()
print(classification_report(
    y_test, y_pred,
    target_names=["Genuine (0)", "Greenwashing (1)"]
))
 
# ── Save ──
pickle.dump(model,      open("model.pkl",      "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("model.pkl      saved")
print("vectorizer.pkl saved")
print("\nNow run:  streamlit run app.py")