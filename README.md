# mmr
Advanced Semantic Search Engine with Dual-Model Architecture and MMR
This project is a powerful, interactive web application for semantic search, built with Streamlit and LangChain. It leverages two distinct sentence-transformer models to provide nuanced and comparable search results. The application implements Maximal Marginal Relevance (MMR) to ensure a balance between result accuracy and diversity, and it connects to a MySQL database to fetch real-time data.

The core feature is its ability to perform a search using a primary model and then score the results using both a paraphrase-focused model and a static similarity model, offering a unique comparative view of semantic relevance.

Project Screenshot Placeholder
(Suggestion: Replace the placeholder above with an actual screenshot of your Streamlit application)

✨ Key Features
Dual-Model Architecture: Utilizes both paraphrase-multilingual-MiniLM-L12-v2 (for understanding rephrasing and context) and static-similarity-mrl-multilingual-v1 (for direct semantic similarity) simultaneously.
Maximal Marginal Relevance (MMR): Goes beyond simple similarity search by promoting diversity in results, preventing a list of very similar-sounding documents.
Interactive UI: A user-friendly interface built with Streamlit allows for easy configuration of all search parameters.
Comparative Scoring: For every search result, similarity scores from both models are calculated and displayed, allowing for deeper analysis of relevance.
Configurable Search Parameters: Users can dynamically adjust:
The primary model for the search index.
k: The number of final results.
fetch_k: The number of initial candidates for MMR.
lambda_mult: The balance between similarity and diversity in MMR.
similarity_threshold: A post-search filter to discard irrelevant results.
Database Integration: Fetches text data directly from a MySQL database.
Detailed JSON Export: Saves all search parameters, timings, initial results, and final filtered results to a results.json file for further analysis or logging.
⚙️ How It Works
The application follows a systematic pipeline from data ingestion to result presentation:

Data Loading: Connects to a MySQL database to fetch the source texts (e.g., news summaries).
Model Caching: On startup, both sentence-transformer models are loaded into memory and cached using st.cache_resource for high performance.
User Input: The user provides a search query and configures search parameters through the Streamlit interface, including selecting the primary model for the initial search.
Vector Indexing: A FAISS vector index is built in real-time using the embeddings from the selected primary model. This process is also cached.
MMR Retrieval: The FAISS index is queried using the MMR algorithm to retrieve an initial set of fetch_k documents that are both relevant to the query and diverse.
Dual-Model Scoring: The query and each retrieved document are embedded using both models. Cosine similarity is calculated for each model, providing two distinct relevance scores for every document.
Filtering & Ranking: The results are filtered based on the similarity_threshold applied to the primary model's score. They are then sorted by this primary score and truncated to the top k results.
Display & Export: The final, ranked results—showing both similarity scores—are displayed in the UI. The complete session details are saved to results.json.
🛠️ Core Technologies
Backend: Python
Web Framework: Streamlit
AI/ML Orchestration: LangChain
Embedding Models: Sentence-Transformers (from Hugging Face)
Vector Store: FAISS (Facebook AI Similarity Search)
Database: MySQL
Numerical Operations: NumPy
