import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import mysql.connector
import numpy as np
import json
from datetime import datetime
from time import time

# ---------- Model Definitions ----------
# نام مدل‌ها را در یک دیکشنری برای مدیریت آسان‌تر تعریف می‌کنیم
MODEL_NAMES = {
    "Paraphrase (MiniLM)": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "Static Similarity (MRL)": "sentence-transformers/static-similarity-mrl-multilingual-v1"
}

# ---------- Database Connection ----------
def connect_to_db():
    """Establishes a connection to the MySQL database."""
    return mysql.connector.connect(
        host='192.168.1.14',
        user='root',
        password="12345",
        database="niki7",
        port=3306
    )

def get_crypto_news_from_db(limit=None):
    """Fetches news summaries from the database."""
    connection = None
    cursor = None
    try:
        connection = connect_to_db()
        cursor = connection.cursor(dictionary=True)
        sql = "SELECT summary FROM news_topics LIMIT %s"
        cursor.execute(sql, (limit,))
        results = cursor.fetchall()
        return results
    except mysql.connector.Error as e:
        st.error(f"❌ Database connection error: {e}")
        return []
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()

def get_total_news_count():
    """Gets the total number of news articles in the database."""
    connection = None
    cursor = None
    try:
        connection = connect_to_db()
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM news_topics")
        total_count = cursor.fetchone()[0]
        return total_count
    except mysql.connector.Error as e:
        st.error(f"❌ Error getting total news count: {e}")
        return 100
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()

# ---------- Caching ----------
@st.cache_data(show_spinner="📡 Reading from the database...")
def get_texts_from_db(limit):
    """Caches the text data fetched from the database."""
    news = get_crypto_news_from_db(limit)
    return [item['summary'] for item in news if 'summary' in item]

@st.cache_resource(show_spinner="🧠 Loading embedding models...")
def load_embedding_models():
    """
    Loads both sentence-transformer models and caches them.
    Returns a dictionary mapping model names to their HuggingFaceEmbeddings instances.
    """
    models = {}
    for model_key, model_name in MODEL_NAMES.items():
        models[model_name] = HuggingFaceEmbeddings(model_name=model_name)
    st.success("✅ Embedding models loaded successfully!")
    return models

@st.cache_resource(show_spinner="🛠️ Building FAISS index...")
def build_faiss_index(texts, _embedding_model, model_name_key):
    """
    Builds and caches the FAISS index for a given model.
    The model_name_key is added to the cache key to ensure a new index is built
    if the model changes.
    """
    docs = [Document(page_content=t) for t in texts]
    return FAISS.from_documents(docs, _embedding_model)

# ---------- Cosine Similarity ----------
def cosine_similarity(a, b):
    """Calculates cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Dual-Model MMR Search", page_icon="🔍")
st.title("🔍 Search Texts with Two Models (MMR + Cosine Similarity)")

# --- Load data and models ---
total_count = get_total_news_count()
all_embedding_models = load_embedding_models()

# --- UI Controls ---
limit = st.slider("📥 Number of texts to load from database", min_value=10, max_value=total_count, value=total_count)
texts = get_texts_from_db(limit=limit)

query = st.text_input("❓ Enter your search query:", value="Price of bitcoin in the last 24 hours")

# --- Model Selection ---
primary_model_key = st.selectbox(
    "🤖 Select the primary model for search (MMR & Filtering)",
    options=list(MODEL_NAMES.keys()),
    index=0, # Default to Paraphrase model
    help="This model will be used to build the search index and for initial filtering."
)
primary_model_name = MODEL_NAMES[primary_model_key]
primary_embedding_model = all_embedding_models[primary_model_name]

# --- Search Parameters ---
k = st.slider("🔢 Number of final results to show", 1, 30, 20)
fetch_k = st.slider("🎯 Number of initial candidates (fetch_k)", k, 50, 40)
lambda_mult = st.slider("⚖️ Similarity/Diversity balance (lambda)", 0.0, 1.0, 0.7)
similarity_threshold = st.slider(f"📈 Cosine similarity threshold (for {primary_model_key})", 0.0, 1.0, 0.55, 0.01)

if st.button("🔎 Search"):
    if not query:
        st.warning("Please enter a search query.")
    else:
        start_time = time()
        st.info("⏳ Processing embeddings and performing MMR search...")

        # --- Get both models for scoring ---
        paraphrase_model = all_embedding_models[MODEL_NAMES["Paraphrase (MiniLM)"]]
        static_model = all_embedding_models[MODEL_NAMES["Static Similarity (MRL)"]]

        # --- Build index using the selected primary model ---
        vectorstore = build_faiss_index(texts, primary_embedding_model, primary_model_name)

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": fetch_k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
        )

        # --- Perform MMR search ---
        mmr_results_docs = retriever.invoke(query)

        # --- Embed query with both models ONCE for efficiency ---
        query_emb_paraphrase = paraphrase_model.embed_query(query)
        query_emb_static = static_model.embed_query(query)

        # --- Calculate scores with both models for all MMR results ---
        all_scored_results = []
        with st.spinner("Calculating similarity scores with both models..."):
            for doc in mmr_results_docs:
                doc_text = doc.page_content
                # Embed document text with both models
                doc_emb_paraphrase = paraphrase_model.embed_query(doc_text)
                doc_emb_static = static_model.embed_query(doc_text)
                
                # Calculate cosine similarity for both
                score_paraphrase = cosine_similarity(query_emb_paraphrase, doc_emb_paraphrase)
                score_static = cosine_similarity(query_emb_static, doc_emb_static)
                
                all_scored_results.append({
                    "text": doc_text,
                    "paraphrase_similarity_score": round(score_paraphrase, 4),
                    "static_similarity_score": round(score_static, 4),
                    # Helper field to know which score is the primary one for filtering
                    "primary_score": round(score_paraphrase, 4) if primary_model_name == MODEL_NAMES["Paraphrase (MiniLM)"] else round(score_static, 4)
                })

        # --- Filter results based on the primary model's score threshold ---
        filtered_results = [
            res for res in all_scored_results if res["primary_score"] >= similarity_threshold
        ]

        # --- Sort by primary score and limit to k results ---
        # The key for sorting depends on which model was selected as primary
        sort_key = 'paraphrase_similarity_score' if primary_model_name == MODEL_NAMES["Paraphrase (MiniLM)"] else 'static_similarity_score'
        final_results = sorted(filtered_results, key=lambda x: x[sort_key], reverse=True)[:k]

        end_time = time()
        duration = end_time - start_time

        # --- Display Results ---
        st.subheader(f"Showing {len(final_results)} results:")
        for i, result in enumerate(final_results, 1):
            st.markdown(f"---")
            st.markdown(f"**{i}.**")
            st.markdown(f"> {result['text']}")
            st.markdown(
                f"🔹 **Paraphrase Score:** `{result['paraphrase_similarity_score']}`\n\n"
                f"🔸 **Static Score:** `{result['static_similarity_score']}`"
            )

        # --- Prepare and save JSON output ---
        # Remove the temporary 'primary_score' key before saving
        for res in all_scored_results: res.pop('primary_score', None)
        for res in final_results: res.pop('primary_score', None)
        
        output_data = {
            "query": query,
            "primary_search_model": primary_model_name,
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "fetch_k": fetch_k,
                "k": k,
                "lambda_mult": lambda_mult,
                "similarity_threshold": similarity_threshold
            },
            "execution_time_seconds": round(duration, 2),
            "initial_mmr_results_scored": sorted(all_scored_results, key=lambda x: x[sort_key], reverse=True),
            "final_filtered_results": final_results
        }
        
        try:
            with open("results.json", "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            st.success(f"✅ Results saved to 'results.json'.")
        except Exception as e:
            st.error(f"❌ Failed to save JSON file: {e}")

        st.info(f"⏱️ Total execution time: {duration:.2f} seconds")