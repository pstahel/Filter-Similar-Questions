import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tkinter as tk
from tkinter import filedialog
import spacy
import string
from langdetect import detect
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Checking if stopwords are available for all required languages
print('English stopwords available:', 'english' in stopwords.fileids())
print('German stopwords available:', 'german' in stopwords.fileids())
print('Spanish stopwords available:', 'spanish' in stopwords.fileids())

# Load spaCy models for different languages
nlp_en = spacy.load("en_core_web_sm")
nlp_de = spacy.load("de_core_news_sm")
nlp_es = spacy.load("es_core_news_sm")

# Load stopwords for each language
stopwords_dict = {
    'en': set(stopwords.words('english')),
    'de': set(stopwords.words('german')),
    'es': set(stopwords.words('spanish'))
}

def get_spacy_model(language):
    if language == 'en':
        return nlp_en
    elif language == 'de':
        return nlp_de
    elif language == 'es':
        return nlp_es
    else:
        raise ValueError("Unsupported language")

def normalize_text(text):
    """
    Function to normalize text: lowercase and remove punctuation.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def preprocess_text(text, language):
    """
    Function to preprocess text by removing stop words and lemmatizing.
    """
    nlp = get_spacy_model(language)
    language_stopwords = stopwords_dict.get(language, set())
    
    doc = nlp(text)
    lemmatized = [token.lemma_ for token in doc if token.text not in language_stopwords]
    return ' '.join(lemmatized)

def find_similar_rows(df, original_column, new_column, similarity_threshold=0.75):
    """
    Identify rows with similar search intent based on the specified column.
    """
    # Determine the language of the first non-empty row
    language = detect(df[original_column].dropna().iloc[0])

     # Print the detected language
    language_map = {'en': 'English', 'de': 'German', 'es': 'Spanish'}
    print(f"Detected language: {language_map.get(language, 'Unknown')} ({language})")
    
    # Normalize the text data in the original column
    df[new_column] = df[original_column].apply(normalize_text)

    # Preprocess the text data for similarity comparison
    df['processed_' + new_column] = df[new_column].apply(lambda x: preprocess_text(x, language))

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['processed_' + new_column])

    cosine_sim = cosine_similarity(tfidf_matrix)

    similar_pairs = np.argwhere(cosine_sim > similarity_threshold)
    similar_pairs = similar_pairs[similar_pairs[:, 0] != similar_pairs[:, 1]]

    similar_pairs = np.sort(similar_pairs, axis=1)
    similar_pairs = np.unique(similar_pairs, axis=0)

    indices_to_remove = set()
    for i, j in similar_pairs:
        indices_to_remove.add(j)

    return indices_to_remove


# Open file dialog for choosing the file
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title='Select CSV file', filetypes=[('CSV files', '*.csv')])

if file_path:
    # Read the original data
    df_original = pd.read_csv(file_path)

    # Check if 'question' column exists, if not, ask for an alternative
    if 'question' not in df_original.columns:
        print("The 'question' column was not found in the file.")
        original_column = input("Please enter the column name to process: ")
    else:
        original_column = 'question'

    # Specify the new column name for normalized text
    new_column = 'normalized_' + original_column

    # Finding similar rows based on the specified column
    similar_questions_indices_only = find_similar_rows(df_original, original_column, new_column)

    # Filtering the DataFrame to remove rows with similar questions
    df_filtered_questions_only = df_original.drop(similar_questions_indices_only)

    # Save the filtered DataFrame in the same directory
    output_file_path = file_path.rsplit('.', 1)[0] + '_filtered.csv'
    df_filtered_questions_only.to_csv(output_file_path, index=False)

    print("Data processing complete. Filtered data saved as:", output_file_path)
    print("Total rows in filtered DataFrame:", len(df_filtered_questions_only))
    print("Rows removed:", len(df_original) - len(df_filtered_questions_only))
else:
    print("No file selected. Exiting.")
