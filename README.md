# Similar Questions Remover

This Python script identifies and removes rows with similar content in a specified column of a CSV file. It primarily targets the 'question' column, employing TF-IDF vectorization and cosine similarity for the task.

## Features

- Reads data from a CSV file.
- Finds and removes rows with similar text in the 'question' column.
- Saves the cleaned data as a new CSV file in the same directory.
- Provides a file dialog for easy CSV file selection.
- Outputs success message and statistics in the terminal.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- numpy
- tkinter

## Installation

Ensure Python 3.x is installed on your system. Install the required packages using pip:

```bash
pip install pandas scikit-learn numpy tk
```

## Usage

- Run the script.  
- A file dialog will open. Select the CSV file you want to process.  
- The script will process the file and save a new file with '_filtered' appended to the original filename in the same directory.  
- Check the terminal for a success message and details about the number of rows processed and removed.

## CSV File Format

- The script expects the CSV file to have a column named 'question'.
- The CSV should have a header row with column names.
- The file should be in UTF-8 encoding or similar standard text format.

## Note

- The script is configured to process text data. Ensure the 'question' column contains text data.
- You can modify the script to target a different column by changing the column name in the script.

## License

This script is provided "as is", without warranty of any kind.

## Contributing

Feel free to fork, modify, and use this script in your projects. Contributions for improvements and bug fixes are welcome.
