import os
import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from .forms import ReviewForm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import opinion_lexicon
import textstat
import scipy.sparse as sp

# Загрузка сохранённых моделей
log_reg = joblib.load(os.path.join(os.path.dirname(__file__), 'logistic_model.pkl'))
ridge = joblib.load(os.path.join(os.path.dirname(__file__), 'ridge_model.pkl'))
tfidf = joblib.load(os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl'))

# Лемматизация и стоп-слова
stop_words = set(stopwords.words('english'))

# Функция предобработки текста
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text

# Функция для извлечения признаков
def extract_features(reviews):
    polarities = [TextBlob(review).sentiment.polarity for review in reviews]
    positive_words = set(opinion_lexicon.positive())
    negative_words = set(opinion_lexicon.negative())

    pos_counts, neg_counts, adj_counts, adv_counts = [], [], [], []
    exclamation_counts, question_counts, readability_scores = [], [], []

    for review in reviews:
        tokens = word_tokenize(review)
        pos_count = sum(1 for word in tokens if word in positive_words)
        neg_count = sum(1 for word in tokens if word in negative_words)
        pos_counts.append(pos_count)
        neg_counts.append(neg_count)

        pos_tags = pos_tag(tokens)
        adj_count = sum(1 for word, tag in pos_tags if tag.startswith('JJ'))  # Adjective
        adv_count = sum(1 for word, tag in pos_tags if tag.startswith('RB'))  # Adverb
        adj_counts.append(adj_count)
        adv_counts.append(adv_count)

        exclamation_counts.append(review.count('!'))
        question_counts.append(review.count('?'))
        readability_scores.append(textstat.flesch_reading_ease(review))

    features_df = pd.DataFrame({
        'Polarity': polarities,
        'Pos_Count': pos_counts,
        'Neg_Count': neg_counts,
        'Adj_Count': adj_counts,
        'Adv_Count': adv_counts,
        'Exclamation_Count': exclamation_counts,
        'Question_Count': question_counts,
        'Readability_Score': readability_scores
    })

    return features_df

# Главная страница с формой для ввода отзыва
def home(request):
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            review = form.cleaned_data['review']
            
            # Предобработка текста
            preprocessed_review = preprocess_text(review)
            
            # Векторизация текста и извлечение признаков
            review_tfidf = tfidf.transform([preprocessed_review])
            review_features = extract_features([preprocessed_review])
            review_combined = sp.hstack([review_tfidf, review_features.values])
            
            # Первая модель: Логистическая регрессия для определения полярности (sentiment)
            sentiment_pred = log_reg.predict(review_combined)
            
            # Вторая модель: RidgeClassifier для определения рейтинга
            review_with_sentiment = sp.hstack([review_combined, np.array(sentiment_pred).reshape(-1, 1)])
            rating_pred = ridge.predict(review_with_sentiment)
            
            # Декодирование результата
            if sentiment_pred == 1:
                rating = rating_pred + 3  # Рейтинги 7-10
                sentiment = 'Положительный'
            else:
                rating = rating_pred + 1  # Рейтинги 1-4
                sentiment = 'Негативный'
            
            return JsonResponse({
                'sentiment': sentiment,
                'rating': int(rating)
            })
    else:
        form = ReviewForm()

    return render(request, 'reviews/home.html', {'form': form})
