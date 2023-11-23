import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Load your product dataset
product_data = pd.read_csv(r"C:\Users\shrut\Downloads\bigBasketProducts.csv")

# Load the index from a file
index = faiss.read_index("product_index.faiss")

# Load a pre-trained model (e.g., MiniLM for sentence embeddings)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Streamlit app
st.title("Product Query App")

# User input for custom question
custom_question = st.text_input("Enter your question:")

# Button to trigger the search
if st.button("Get Answer"):
    if custom_question:
        # Encode the custom question
        question_embedding = model.encode([custom_question])

        # Search for the nearest neighbors of the question vector
        k = 5  # Number of neighbors to retrieve
        _, neighbors = index.search(question_embedding, k)

        # Retrieve the details of the nearest neighbors
        neighbor_indices = neighbors[0]
        neighbor_details = product_data.loc[neighbor_indices]

        # Display the results
        st.subheader("Top 5 Recommendations:")
        for i, (_, details) in enumerate(neighbor_details.iterrows()):
            st.write(f"{i+1}. {details['product']} - {details['description']}")
