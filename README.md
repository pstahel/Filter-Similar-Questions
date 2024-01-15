# Similar Question Filter

This Python script is designed to identify and remove rows with similar content in a specified column of a CSV file. It supports multiple languages (English, German, Spanish) and employs advanced text processing techniques like normalization, stop word removal, and lemmatization.

## Features

- Supports English, German, and Spanish languages.
- Normalizes text by converting to lowercase and removing punctuation.
- Removes stop words and applies lemmatization for more accurate similarity detection.
- Uses TF-IDF vectorization and cosine similarity to identify similar rows.
- Interactive file selection using a file dialog.
- Saves the filtered data in the same directory as the original file.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- numpy
- tkinter
- spacy
- langdetect
- nltk

## Installation

Ensure Python 3.x is installed. Install the required packages using pip:

```
pip install pandas scikit-learn numpy tkinter spacy langdetect nltk
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
python -m spacy download es_core_news_sm
```

# Usage

1. Run the script.
2. A file dialog will open. Select the CSV file you want to process.
3. The script will automatically detect the language of the content, process the file, and save a new file with '_filtered' appended to the original filename in the same directory.
4. The terminal will display the detected language, the location of the saved file, and statistics about the number of rows processed and removed.

## CSV File Format

- The script expects the CSV file to have a column named 'question' by default.
- If the 'question' column is not found, it will prompt to enter an alternative column name.
- The file should be in a standard CSV format with a header row.

## Note

- The script is configured to process text data. Ensure the column you intend to process contains textual data.
- The language detection is performed on the first non-empty row of the specified column.

## License

- This script is provided "as is", without warranty of any kind.

## Contributing

- Feel free to fork, modify, and use this script in your projects.
- Contributions for improvements and bug fixes are welcome.
