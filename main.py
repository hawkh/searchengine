import streamlit as st
from search_engine import SearchEngine

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Advanced Search Engine",
        page_icon=":mag_right:",
        layout="wide"
    )

    # Custom CSS
    with open("static/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Title and description
    st.title("Advanced Search Engine")
    st.markdown("### Powered by Natural Language Processing")

    # Search input
    search_query = st.text_input("Enter your search query", placeholder="Type your search here...")

    # Search button
    if st.button("Search"):
        if search_query:
            # Initialize search engine
            search_engine = SearchEngine()
            
            # Perform search
            with st.spinner("Searching across web pages..."):
                # Normalization results
                st.subheader("Normalization Results")
                norm_results = search_engine.normalize_search(search_query)
                if norm_results:
                    for url in norm_results:
                        st.markdown(f"- {url}")
                else:
                    st.warning("No results found using normalization.")

                # TF-IDF results
                st.subheader("TF-IDF Ranked Results")
                tfidf_results = search_engine.tfidf_search(search_query)
                if tfidf_results:
                    for url in tfidf_results:
                        st.markdown(f"- {url}")
                else:
                    st.warning("No results found using TF-IDF ranking.")

        else:
            st.error("Please enter a search query.")

    # Additional information
    st.markdown("""
    #### How It Works
    - Tokenizes your search query
    - Removes stop words
    - Applies stemming
    - Searches across predefined web pages
    - Ranks results using normalization and TF-IDF
    """)

if __name__ == "__main__":
    main()
