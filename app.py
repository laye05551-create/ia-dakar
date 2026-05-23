import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1. Charger le fichier
df = pd.read_csv('data.csv')
import pandas as pd
# ... le reste de tes imports ...


df = pd.read_csv('data.csv')

# AJOUTE CECI POUR VOIR TES COLONNES :
print("Colonnes détectées :", df.columns.tolist())
print(df.head()) # Affiche les premières lignes
# ------------------------------------

# ... reste de ton code ...
# 2. Créer le modèle
modele = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])

# 3. Entraîner (L'IA lit ton fichier CSV)
modele.fit(df['message'], df['categorie'])
import joblib

# ... après modele.fit(df['message'], df['categorie']) ...
# SAUVEGARDE LE MODÈLE
joblib.dump(modele, 'modele_sav.pkl')
print("Modèle enregistré avec succès sous le nom 'modele_sav.pkl' !")

# 4. Tester avec une phrase que l'IA n'a jamais vue
test = ["c'est quel prix svp"]
resultat = modele.predict(test)

print(f"Le message '{test[0]}' appartient à la catégorie : {resultat[0]}")
# ... (garde tout le code précédent)

# 4. Boucle de test interactive
print("\n--- Analyseur de SAV prêt ! (Tape 'quitter' pour arrêter) ---")
while True:
    user_input = input("\nEntrez un message client : ")
    if user_input.lower() == 'quitter':
        break
    
    cat = modele.predict([user_input])[0]
    print(f"Catégorie détectée : {cat}")