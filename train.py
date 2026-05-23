import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Charger les données
df = pd.read_csv('data.csv')

# 2. Créer le modèle (n-gram (1,2) pour comprendre les groupes de mots)
model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', LogisticRegression()),
])

# 3. Entraîner
model.fit(df['message'], df['categorie'])

# 4. Sauvegarder
joblib.dump(model, 'modele_sav.pkl')
print("Modèle entraîné et sauvegardé avec succès !")