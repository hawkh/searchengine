# Advanced Search Engine with NLP

## Overview
This project implements a search engine using Natural Language Processing techniques in Python with a Streamlit web interface.

## Features
- Query tokenization
- Stop word removal
- Stemming
- KMP search algorithm
- Normalization and TF-IDF ranking
- Web page text extraction

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your `links.txt` file with web page URLs to search

## Usage
```bash
streamlit run main.py
```

## Dependencies
- Streamlit
- NLTK
- BeautifulSoup
- urllib3

## Methodology
- Tokenizes search queries
- Removes stop words
- Applies Porter Stemming
- Searches across predefined web pages
- Ranks results using normalization and TF-IDF

## 🤝 Contributing

Contributions are welcome! Please check the outstanding issues and feel free to open a pull request.

