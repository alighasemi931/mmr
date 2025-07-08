import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import mysql.connector
import numpy as np
import json
from datetime import datetime
from time import time
import os
from dotenv import load_dotenv
load_dotenv(override=True)
# ---------- Database Connection ----------
def connect_to_db():
    return mysql.connector.connect(
        host=os.getenv('HOST'),
        user=os.getenv('USER'),
        password=os.getenv('PASSWORD'),
        database=os.getenv('DATABASE'),
        port=os.getenv('PORT')
    )

def get_crypto_news_from_db(limit=None):
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
    news = get_crypto_news_from_db(limit)
    return [item['summary'] for item in news if 'summary' in item]

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_resource
def build_faiss_index(texts, _embedding_model): 
    docs = [Document(page_content=t) for t in texts]
    return FAISS.from_documents(docs, _embedding_model)

# ---------- Cosine Similarity ----------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------- Streamlit UI ----------
st.set_page_config(page_title="MMR Search with Cosine Similarity", page_icon="🔍")
st.title("🔍 Search texts using MMR + Cosine Similarity Filter")

total_count = get_total_news_count()
limit = st.slider("📥 Number of texts to load from database", min_value=10, max_value=total_count, value=total_count)
texts = get_texts_from_db(limit=limit)

query = st.text_input("❓ Enter your search query:", value="Price of bitcoin in the last 24 hours")

k = st.slider("🔢 Number of final results to show", 1, 30, 20)
fetch_k = st.slider("🎯 Number of initial candidates (fetch_k)", k, 50, 40)
lambda_mult = st.slider("⚖️ Similarity/Diversity balance (lambda)", 0.0, 1.0, 0.7)
similarity_threshold = st.slider("📈 Cosine similarity threshold", 0.0, 1.0, 0.55, 0.01)

if st.button("🔎 Search"):

    start_time = time()  # ⏱ زمان شروع
    st.info("⏳ Processing embeddings and performing MMR search...")

    embedding_model = load_embedding_model()
    vectorstore = build_faiss_index(texts, embedding_model)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": fetch_k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult
        }
    )

    mmr_results = retriever.invoke(query)
    query_emb = embedding_model.embed_query(query)

    # ذخیره نتایج اولیه
    initial_results = []
    for doc in mmr_results:
        doc_emb = embedding_model.embed_query(doc.page_content)
        score = cosine_similarity(query_emb, doc_emb)
        initial_results.append({
            "text": doc.page_content,
            "similarity_score": round(score, 4)
        })
    initial_results = sorted(initial_results, key=lambda x: x["similarity_score"], reverse=True)
    # فیلتر نهایی بر اساس آستانه
    filtered_results = [
        (doc, cosine_similarity(query_emb, embedding_model.embed_query(doc.page_content)))
        for doc in mmr_results
        if cosine_similarity(query_emb, embedding_model.embed_query(doc.page_content)) >= similarity_threshold
    ]
    filtered_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:k]

    final_results = [
        {
            "text": doc.page_content,
            "similarity_score": round(score, 4)
        }
        for doc, score in filtered_results
    ]

    # ذخیره در فایل JSON
    output = {
        "query": query,
        "model_name": embedding_model.model_name,
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "fetch_k": fetch_k,
            "k": k,
            "lambda_mult": lambda_mult,
            "similarity_threshold": similarity_threshold
        },
        "initial_mmr_results": initial_results,
        "final_filtered_results": final_results
    }

    # زمان کل اجرا
    end_time = time()
    duration = end_time - start_time
    

    # نمایش نتایج
    for i, result in enumerate(final_results, 1):
        st.markdown(f"**{i}.** 🔹 Similarity Score: `{result['similarity_score']}`\n\n{result['text']}")
    
    st.success("✅ نتایج در فایل 'results.json' ذخیره شد.")
    st.success(f"⏱ زمان کل اجرا: {duration:.2f} ثانیه")
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)