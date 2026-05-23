import streamlit as st
import joblib

# 1. Barre latérale pour la configuration (Le "Tableau de bord")
st.sidebar.title("⚙️ Paramètres Boutique")
prix_produit = st.sidebar.text_input("Prix du produit (FCFA)", "5000")
quartier = st.sidebar.text_input("Quartier", "Dakar Plateau")
adresse_complete = st.sidebar.text_input("Adresse précise", "À côté de la station Shell")

# 2. Charger le modèle
model = joblib.load('modele_sav.pkl')

st.title("🤖 SAV Pro - Dakar")

# 3. Zone de test
msg = st.text_input("Message du client :")

if msg:
    cat = model.predict([msg])[0]
    
    # 4. Génération de réponse dynamique avec les paramètres de la barre latérale
    if "niatta" in msg.lower() and "deuk" in msg.lower():
        reponse = f"Salam ! Le prix est de {prix_produit} FCFA. Nous sommes à {quartier}, {adresse_complete}. Voulez-vous qu'on vous livre ?"
    elif cat == "prix":
        reponse = f"Salam ! Ce produit coûte {prix_produit} FCFA. Très bonne qualité, on vous en garde un ?"
    elif cat == "adresse":
        reponse = f"Salam ! Nous sommes situés à {quartier}, {adresse_complete}. Voici notre position !"
    else:
        reponse = "Salam ! Merci de nous contacter, nous traitons votre demande immédiatement."
    
    st.write("### Réponse suggérée :")
    st.success(reponse)