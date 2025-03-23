import random
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.classify import NaiveBayesClassifier
from flask import Flask, render_template, request
import pickle

# Télécharger les ressources nécessaires de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Initialiser le stemmer et les stopwords
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Charger les données CSV (en supposant que le fichier est train.csv)
df = pd.read_csv('projet_Analyse_des_Sentiments/train.csv')

# Supprimer la colonne 'selected_text' si elle existe
df = df.drop(columns=['selected_text'], errors='ignore')

# Convertir la colonne 'text' en chaîne de caractères
df['text'] = df['text'].astype(str)

# Fonction de prétraitement du texte
def preprocess_text(text):
    # Convertir en minuscules et retirer les stopwords
    words = nltk.word_tokenize(text.lower())
    words = [stemmer.stem(word) for word in words if word not in stop_words and word.isalpha()]
    return words

# Appliquer le prétraitement sur le texte du DataFrame
df['processed_text'] = df['text'].apply(preprocess_text)

# Fonction pour extraire les caractéristiques du texte (à utiliser avec Naive Bayes)
def document_features(doc):
    return {word: True for word in doc}

# Créer les ensembles de données pour l'entraînement
featuresets = [(document_features(text), sentiment) for text, sentiment in zip(df['processed_text'], df['sentiment'])]

# Mélanger les données et diviser en ensemble d'entraînement et de test
random.shuffle(featuresets)
train_set, test_set = featuresets[:int(0.8*len(featuresets))], featuresets[int(0.8*len(featuresets)):]

# Entraîner le modèle Naive Bayes
classifier = NaiveBayesClassifier.train(train_set)

# Sauvegarder le modèle entraîné avec pickle
with open('naive_bayes_model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

# Initialiser Flask
app = Flask(__name__)

# Charger le modèle Naive Bayes (si nécessaire)
with open('naive_bayes_model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

# Fonction pour analyser le sentiment d'un texte
def analyze_sentiment(text):
    words = preprocess_text(text)
    features = document_features(words)
    return classifier.classify(features)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']
        sentiment = analyze_sentiment(text)
        return render_template('index.html', sentiment=sentiment, text=text)

if __name__ == '__main__':
    app.run(debug=True)
