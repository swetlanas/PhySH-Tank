#streamlit/front_end.py

#Streamlit front end for getting titles and abstracts.


import streamlit as st
import requests
import os


# Uses environment variable if present (inside Docker), otherwise defaults to localhost
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_URL = f"{BACKEND_URL}/predict"

def main():
    st.title("APS Physical Review B PhySH Tag Recommender")
    st.write("Enter the title and abstract of your preprint and get a list of 5 ordered [PhySH](https://physh.org/) tags. Use the slider to get up to 15 tags, though the model is optimized for top 5 tags.")

    title_text = st.text_input('Paper title')
    abstract_text = st.text_area('Paper abstract')

    k_tags = st.slider("Select the number of tags to display:", min_value=5, max_value=15, value=5)
    st.write("Number of tags:", k_tags)
    
    if st.button('Get tags'):
        payload = {"title" : title_text,
            "abstract" : abstract_text,
            "top_k" : k_tags}
        try:
            response = requests.post(API_URL, json=payload)
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend. FastAPI server is down.")
        else:
            result = response.json() 
            if response.status_code==200:
                if result.get("warning"):
                    st.warning(result["warning"], icon="⚠️")
                st.success(f"Top {k_tags} tags are: ")
                for idx, items in enumerate(result["tags"],1):
                    st.markdown(f"{idx}. {items}")
            elif response.status_code==400:
                st.error(result["detail"])
            else:
                st.error(f"Unexpected error (status {response.status_code}): {result}")

if __name__ == "__main__":
    main()
