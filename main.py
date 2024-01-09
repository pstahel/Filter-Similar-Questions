import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tkinter as tk
from tkinter import filedialog

def find_similar_rows(df, column, similarity_threshold=0.75):
    """
    Identify rows with similar search intent based on the specified column.
    A similarity threshold is used to determine how close the rows need to be to consider them similar.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df[column])

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

    # Finding similar rows based on 'question' column only
    similar_questions_indices_only = find_similar_rows(df_original, 'question')

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
