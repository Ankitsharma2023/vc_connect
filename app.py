import streamlit as st
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer

# --- Load Data & Model (cache to avoid reload on every run) ---
@st.cache_resource(show_spinner=True)
def load_data():
    df = pd.read_csv("investor_data.csv")
    return df

@st.cache_resource(show_spinner=True)
def load_faiss_index():
    index = faiss.read_index("investor_index.faiss")
    return index

@st.cache_resource(show_spinner=True)
def load_model():
    model = SentenceTransformer('BAAI/bge-small-en', device='cuda' if torch.cuda.is_available() else 'cpu')
    return model

df = load_data()
index = load_faiss_index()
model = load_model()

# --- Streamlit UI ---
st.title("Startup Founder - Investor Matchmaking")

startup_idea = st.text_area("Describe your startup idea", height=150)

if st.button("Find Investors"):
    if not startup_idea.strip():
        st.warning("Please enter your startup idea first!")
    else:
        with st.spinner("Searching for best investor matches..."):
            # Encode input
            query_embedding = model.encode([startup_idea], convert_to_tensor=False)
            
            # FAISS expects float32 numpy array
            query_embedding = query_embedding.astype('float32')
            
            # Search top 5 matches
            k = 5
            distances, indices = index.search(query_embedding, k)
            
            st.write(f"Top {k} investor matches:")
            for rank, idx in enumerate(indices[0]):
                investor = df.iloc[idx]
                st.markdown(f"### {rank+1}. {investor['Investor name']}")
                st.markdown(f"**Type:** {investor['Investor type']}")
                st.markdown(f"**Focus:** {investor['Investment thesis']}")
                st.markdown(f"**Stage:** {investor['Stage of investment']}")
                st.markdown(f"**Location:** {investor['Countries of investment']}")
                st.markdown(f"**Check Size:** ${investor['First cheque minimum']} to ${investor['First cheque maximum']}")
                st.markdown("---")
