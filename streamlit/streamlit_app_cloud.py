import streamlit as st
from pathlib import Path
import sys
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.predict import clean_text, load_model, predict_tags

@st.cache_resource(show_spinner="Loading model (first run takes a while)...")
def get_model():
    return load_model()

st.set_page_config(layout="wide")


def rss_mb() -> float:
    """Resident memory of this process in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

tokenizer, model, id2label = get_model()

title = st.text_input("Title")
abstract = st.text_area("Abstract")
top_k = st.slider("Number of tags", 5, 15, 5)
st.caption("The model is optimized for the top 5 tags.")

if st.button("Get tags"):
    st.sidebar.metric("Memory (MB)", f"{rss_mb():.0f}")
    cleaned_title = clean_text(title)
    cleaned_abstract = clean_text(abstract)

    if not cleaned_title.strip() and not cleaned_abstract.strip():
        st.error("Title and Abstract fields are empty.")
    else:
        if not cleaned_title.strip() or not cleaned_abstract.strip():
            st.warning("Results will be more accurate with both a title and abstract.")
        with st.spinner("Predicting..."):
            tags = predict_tags(cleaned_title, cleaned_abstract, top_k,
                                tokenizer, model, id2label)
        for idx,tag in enumerate(tags,1):
            st.write(f"{idx}. {tag}")