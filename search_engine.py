import streamlit as st
import nltk
import math
import urllib.request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import time
import cachetools
from serpapi import GoogleSearch

class SearchEngine:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))
        self.cache = cachetools.TTLCache(maxsize=100, ttl=3600)

    def preprocess_query(self, query, case_sensitive=False):
        if not case_sensitive:
            query = query.lower()
        tokens = word_tokenize(query)
        filtered_tokens = [w for w in tokens if w not in self.stop_words]
        stemmed_tokens = [self.ps.stem(w) for w in filtered_tokens]
        return stemmed_tokens

    def kmp_search(self, pattern, text, case_sensitive=False):
        if not case_sensitive:
            pattern, text = pattern.lower(), text.lower()
        M, N = len(pattern), len(text)
        lps = [0] * M
        
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
        
        matches = []
        i = j = 0
        while i < N:
            if pattern[j] == text[i]:
                i += 1
                j += 1
            if j == M:
                matches.append(i - j)
                j = lps[j-1]
            elif i < N and pattern[j] != text[i]:
                if j != 0:
                    j = lps[j-1]
                else:
                    i += 1
        return matches

    def fetch_webpage_text(self, url, timeout=10):
        if url in self.cache:
            return self.cache[url]
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            )
            resp = urllib.request.urlopen(req, timeout=timeout)
            html = resp.read()
            soup = BeautifulSoup(html, "html.parser")
            
            for script in soup(["script", "style"]):
                script.extract()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            self.cache[url] = text
            return text
        except urllib.error.HTTPError as e:
            if e.code == 403:
                st.warning(f"Skipping {url}: Access denied (HTTP 403). This website may block automated requests.")
            else:
                st.error(f"Error fetching {url}: {e}")
            return ""
        except Exception as e:
            st.error(f"Error fetching {url}: {e}")
            return ""

    def get_snippet(self, text, position, window=50):
        start = max(0, position - window)
        end = min(len(text), position + window)
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet += "..."
        return snippet

    def normalize_search(self, query, case_sensitive=False, phrase_search=False, domain_filter=None):
        if phrase_search:
            stemmed_string = [query.lower()] if not case_sensitive else [query]
        else:
            stemmed_string = self.preprocess_query(query, case_sensitive)
        normalizedict = {}
        snippets_dict = {}
        
        with open("links.txt", "r") as text_file:
            links = [link.strip() for link in text_file if link.strip()]
            if domain_filter:
                links = [link for link in links if domain_filter in link]
        
        progress_bar = st.progress(0)
        for i, link in enumerate(links):
            text = self.fetch_webpage_text(link)
            if not text:
                continue
            if phrase_search:
                matches = self.kmp_search(stemmed_string[0], text, case_sensitive)
                total_matches = len(matches)
                if matches:
                    snippets_dict[link] = self.get_snippet(text, matches[0])
            else:
                word_matches = {word: self.kmp_search(word, text, case_sensitive) for word in stemmed_string}
                total_matches = sum(len(matches) for matches in word_matches.values())
                if total_matches > 0:
                    first_match = next((m[0] for m in word_matches.values() if m), 0)
                    snippets_dict[link] = self.get_snippet(text, first_match)
            normalized_value = total_matches * 1.0 / len(text) if text else 0
            if normalized_value > 0:
                normalizedict[link] = normalized_value
            progress_bar.progress((i + 1) / len(links))
        
        progress_bar.empty()
        return [(k, v, snippets_dict.get(k, "")) for v, k in sorted([(v, k) for k, v in normalizedict.items()], reverse=True)]

    def tfidf_search(self, query, case_sensitive=False, phrase_search=False, domain_filter=None):
        if phrase_search:
            stemmed_string = [query.lower()] if not case_sensitive else [query]
        else:
            stemmed_string = self.preprocess_query(query, case_sensitive)
        
        with open("links.txt", "r") as text_file:
            links = [link.strip() for link in text_file if link.strip()]
            if domain_filter:
                links = [link for link in links if domain_filter in link]
        
        total_docs = len(links)
        word_doc_counts = {word: 0 for word in stemmed_string}
        score = {}
        snippets_dict = {}

        progress_bar = st.progress(0)
        for i, link in enumerate(links):
            text = self.fetch_webpage_text(link)
            if not text:
                continue
            if phrase_search:
                matches = self.kmp_search(stemmed_string[0], text, case_sensitive)
                if matches:
                    word_doc_counts[stemmed_string[0]] += 1
                    snippets_dict[link] = self.get_snippet(text, matches[0])
            else:
                for word in stemmed_string:
                    matches = self.kmp_search(word, text, case_sensitive)
                    if matches:
                        word_doc_counts[word] += 1
                        if link not in snippets_dict:
                            snippets_dict[link] = self.get_snippet(text, matches[0])
            progress_bar.progress((i + 1) / len(links) / 2)

        for i, link in enumerate(links):
            text = self.fetch_webpage_text(link)
            if not text:
                continue
            tf, idf, tfidf = [], [], []
            for word in stemmed_string:
                matches = self.kmp_search(word, text, case_sensitive)
                word_count = len(matches)
                total_words = len(text.split())
                tf_value = word_count / total_words if total_words > 0 else 0
                idf_value = math.log(total_docs / (word_doc_counts[word] + 1))
                tf.append(tf_value)
                idf.append(idf_value)
            tfidf = [t * i for t, i in zip(tf, idf)]
            score[link] = sum(tfidf)
            progress_bar.progress(0.5 + (i + 1) / len(links) / 2)
        
        progress_bar.empty()
        return [(k, v, snippets_dict.get(k, "")) for v, k in sorted([(v, k) for k, v in score.items()], reverse=True)]

    def web_search(self, query, api_key, num_results=10):
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "num": num_results
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            organic_results = results.get("organic_results", [])
            return [(result["link"], None, result.get("snippet", "")) for result in organic_results]
        except Exception as e:
            st.error(f"Error performing web search: {e}")
            return []

def main():
    st.set_page_config(page_title="Web Search Engine", layout="wide")
    st.title("🔍 Universal Web Search Engine")
    st.markdown("Search the web or a custom list of URLs for anything you want.")

    # Initialize search engine
    search_engine = SearchEngine()

    # Sidebar for advanced options
    with st.sidebar:
        st.header("Search Options")
        search_scope = st.radio("Search Scope", ("Web", "Custom URLs"), index=0)
        if search_scope == "Web":
            api_key = st.text_input("SerpAPI Key", type="password", help="Get your API key from serpapi.com")
        else:
            st.info("Ensure links.txt exists with URLs to search, one per line.")
        case_sensitive = st.checkbox("Case Sensitive Search", value=False)
        phrase_search = st.checkbox("Exact Phrase Search", value=False)
        domain_filter = st.text_input("Filter by Domain (e.g., leetcode.com)", "")
        max_results = st.slider("Max Results", 1, 50, 10)

    # Main content
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Search Query", "", placeholder="e.g., leetcode algorithms, best python tutorials, etc.")
    with col2:
        st.write("")  # Spacer
        search_button = st.button("Search")
        clear_button = st.button("Clear")

    if search_scope == "Custom URLs":
        search_method = st.radio(
            "Select Search Method",
            ("Normalized Search", "TF-IDF Search"),
            horizontal=True
        )
    else:
        search_method = None

    if clear_button:
        st.session_state.query = ""
        st.experimental_rerun()

    if search_button:
        if not query:
            st.warning("Please enter a search query")
        else:
            with st.spinner("Searching..."):
                try:
                    start_time = time.time()
                    if search_scope == "Web":
                        if not api_key:
                            st.error("Please provide a SerpAPI key in the sidebar to perform web searches.")
                        else:
                            results = search_engine.web_search(query, api_key, max_results)
                    else:
                        # Check if links.txt exists
                        with open("links.txt", "r") as f:
                            pass
                        if search_method == "Normalized Search":
                            results = search_engine.normalize_search(query, case_sensitive, phrase_search, domain_filter)
                        else:
                            results = search_engine.tfidf_search(query, case_sensitive, phrase_search, domain_filter)
                    end_time = time.time()

                    # Display results
                    if results:
                        st.success(f"Search completed in {end_time - start_time:.2f} seconds. Found {len(results)} results:")
                        for i, (url, score, snippet) in enumerate(results[:max_results], 1):
                            with st.expander(f"{i}. {url} {'(Score: {:.6f})'.format(score) if score is not None else ''}"):
                                st.write(f"**Snippet:** {snippet}")
                                st.write(f"[Visit {url}]({url})")
                    else:
                        st.info("No results found. Try broadening your search or checking your links.txt file (for Custom URLs).")

                except FileNotFoundError:
                    st.error("Error: links.txt file not found. Please create a file named 'links.txt' with URLs to search")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

