import nltk
import math
import urllib.request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup

class SearchEngine:
    def __init__(self):
        # Download required NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        # Initialize stemmer and stop words
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

    def preprocess_query(self, query):
        # Tokenize
        tokens = word_tokenize(query)
        
        # Remove stop words
        filtered_tokens = [w for w in tokens if w not in self.stop_words]
        
        # Stem words
        stemmed_tokens = [self.ps.stem(w) for w in filtered_tokens]
        
        return stemmed_tokens

    def kmp_search(self, pattern, text):
        M, N = len(pattern), len(text)
        lps = [0] * M
        
        # Compute LPS array
        len_lps, i = 0, 1
        while i < M:
            if pattern[i] == pattern[len_lps]:
                len_lps += 1
                lps[i] = len_lps
                i += 1
            else:
                if len_lps != 0:
                    len_lps = lps[len_lps - 1]
                else:
                    lps[i] = 0
                    i += 1
        
        # Search pattern in text
        matches = 0
        i = j = 0
        while i < N:
            if pattern[j] == text[i]:
                i += 1
                j += 1
            
            if j == M:
                matches += 1
                j = lps[j-1]
            
            elif i < N and pattern[j] != text[i]:
                if j != 0:
                    j = lps[j-1]
                else:
                    i += 1
        
        return matches

    def fetch_webpage_text(self, url):
        try:
            resp = urllib.request.urlopen(url)
            html = resp.read()
            
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            
            return '\n'.join(chunk for chunk in chunks if chunk)
        
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return ""

    def normalize_search(self, query):
        # Preprocess query
        stemmed_string = self.preprocess_query(query)
        
        # Track document counts
        normalizedict = {}
        
        # Read links
        with open("links.txt", "r") as text_file:
            links = [link.strip() for link in text_file if link.strip()]
        
        # Search in each link
        for link in links:
            text = self.fetch_webpage_text(link)
            
            # Track word matches
            word_matches = {word: self.kmp_search(word, text) for word in stemmed_string}
            
            # Compute normalized value
            total_matches = sum(word_matches.values())
            normalized_value = total_matches * 1.0 / len(text) if text else 0
            
            if normalized_value > 0:
                normalizedict[link] = normalized_value
        
        # Sort and return top results
        return [k for v, k in sorted([(v, k) for k, v in normalizedict.items()], reverse=True)]

    def tfidf_search(self, query):
        # Preprocess query
        stemmed_string = self.preprocess_query(query)
        
        # Read links
        with open("links.txt", "r") as text_file:
            links = [link.strip() for link in text_file if link.strip()]
        
        # Track document counts
        total_docs = len(links)
        word_doc_counts = {word: 0 for word in stemmed_string}
        score = {}

        # First pass: count documents containing words
        for link in links:
            text = self.fetch_webpage_text(link)
            for word in stemmed_string:
                if self.kmp_search(word, text) > 0:
                    word_doc_counts[word] += 1

        # Second pass: compute TF-IDF
        for link in links:
            text = self.fetch_webpage_text(link)
            
            # Compute term frequencies and IDF
            tf = []
            idf = []
            tfidf = []

            for word in stemmed_string:
                # Term frequency
                word_count = self.kmp_search(word, text)
                total_words = len(text.split())
                tf_value = word_count / total_words if total_words > 0 else 0

                # Inverse document frequency
                idf_value = math.log(total_docs / (word_doc_counts[word] + 1))

                tf.append(tf_value)
                idf.append(idf_value)

            # Compute TF-IDF
            tfidf = [t * i for t, i in zip(tf, idf)]
            score[link] = sum(tfidf)

        # Sort and return top results
        return [k for v, k in sorted([(v, k) for k, v in score.items()], reverse=True)]
