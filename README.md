#  Analyse de Sentiments basée sur Naïve Bayes

Projet universitaire visant à développer une application de classification de sentiments (positif, négatif, neutre) à partir de tweets, en utilisant l'algorithme **Naïve Bayes**.

##  Réalisé par
- **Taftaf Wijdane**


Encadré par : **Mr. Ghazdali Abdelghani**  
Année universitaire : 2024 / 2025  
Université : ENSA Khouribga – Université Sultan Moulay Slimane

---

##  Objectif
Développer un système capable de détecter le **sentiment d’un texte** (tweet) à l’aide de techniques de **traitement du langage naturel (NLP)** et d’un **modèle Naïve Bayes**.

---

## 🛠️Outils et Technologies
- **Langage** : Python
- **Bibliothèques** :
  - `NLTK` : nettoyage, tokenisation, stopwords, stemming
  - `scikit-learn` : modèle Naïve Bayes, vectorisation TF-IDF
  - `Pandas` : manipulation de données
  - `Plotly`, `Matplotlib` : visualisations
- **Frontend** : HTML + CSS
- **Backend** : Flask

---

##  Dataset
Le jeu de données contient des tweets avec :
- `text` : le contenu du tweet
- `selected_text` : partie du texte exprimant le sentiment
- `sentiment` : étiquette (`positive`, `neutral`, `negative`)

---

##  Prétraitement
- Nettoyage des caractères inutiles (ponctuation, URLs, emojis)
- Tokenisation
- Suppression des stopwords
- Stemming
- Encodage des sentiments (`-1`, `0`, `1`)
- Transformation en vecteurs numériques (`TF-IDF`)

---

##  Modèle
Le classifieur **Naïve Bayes** a été choisi pour sa **simplicité**, sa **rapidité**, et son **efficacité** dans la classification de texte.

---

##  Résultats
Le modèle a montré de **bonnes performances** :
- Sentiment **positif** : mots-clés fréquents → `love`, `good`, `happy`
- Sentiment **négatif** : → `hate`, `bad`, `terrible`
- Sentiment **neutre** : → `okay`, `fine`, `average`

---

##  Interface Web
Une interface simple développée avec **Flask** permet :
- De saisir un tweet via un champ HTML
- D’obtenir immédiatement le **sentiment prédit**

---

##  Lancer le projet

```bash
# Cloner le dépôt
git clone https://github.com/wijdanetaftaf/Sentiment-Analysis.git
cd Sentiment-Analysis

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application Flask
python app.py
