import streamlit as st
from twelvelabs import TwelveLabs
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from typing import List

# Function to initialize Milvus connection
def initialize_milvus(collection_name, query_embedding_dim):
    connections.connect(
        alias="default",
        uri=st.secrets["ZILLIZ_CLOUD_URI"],  # Use Streamlit Secrets for secure credentials
        token=st.secrets["ZILLIZ_CLOUD_API_KEY"]
    )

    # Define Milvus collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=query_embedding_dim)  # Match dimension
    ]
    schema = CollectionSchema(fields=fields, description="Demo collection for Twelve Labs embeddings")

    # Drop and recreate collection if it exists
    if utility.has_collection(collection_name):
        Collection(collection_name).drop()
        st.write(f"Dropped existing collection '{collection_name}'.")

    collection = Collection(name=collection_name, schema=schema)
    st.write(f"Created collection '{collection_name}' with correct schema.")
    return collection

# Function to perform similarity search
def perform_similarity_search(collection: Collection, query_vector: list, limit: int = 5):
    search_results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=limit,
        output_fields=["id"]
    )
    return search_results

# Streamlit UI
st.title("Twelve Labs Video Search")

# Step 1: Initialize Twelve Labs client
client = TwelveLabs(api_key=st.secrets["TWELVE_LABS_API_KEY"])

# Query input
query_text = st.text_input("Enter your query:", "How to deal with feedback?")
search_button = st.button("Search")

# Collection name
collection_name = "twelvelabs_demo_collection"

if search_button and query_text:
    # Step 2: Generate text embedding using Twelve Labs
    with st.spinner("Generating embeddings from Twelve Labs..."):
        res = client.embed.create(
            model_name="Marengo-retrieval-2.7",
            text=query_text,
        )

        if res.text_embedding is None or res.text_embedding.segments is None:
            st.error("Failed to retrieve embeddings from Twelve Labs.")
            st.stop()

        # Extract embeddings (assuming one segment for simplicity)
        query_embedding = res.text_embedding.segments[0].embeddings_float
        query_embedding_dim = len(query_embedding)
        st.write("Query embedding generated successfully.")

        # Step 3: Initialize Milvus collection
        collection = initialize_milvus(collection_name, query_embedding_dim)

        # Step 4: Insert dummy data (simulated for this example)
        dummy_embeddings = [query_embedding] * 10  # Simulate 10 identical embeddings
        collection.insert([dummy_embeddings])
        st.write(f"Inserted {len(dummy_embeddings)} embeddings into the collection.")

        # Step 5: Create an index on the embedding field
        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "IP",
                "params": {"nlist": 128}
            }
        )
        st.write("Index created successfully.")

        # Step 6: Load the collection into memory
        collection.load()
        st.write(f"Collection '{collection_name}' loaded into memory.")

    # Step 7: Perform similarity search
    with st.spinner("Performing similarity search..."):
        try:
            search_results = perform_similarity_search(collection, query_embedding, limit=5)

            # Display search results
            st.subheader("Search Results")
            for i, result in enumerate(search_results[0]):
                st.write(f"Result {i+1}: ID={result.id}, Distance={result.distance}")

            # Example: Returning a dummy video URL
            video_url = "https://drive.google.com/file/d/1Sa2Ctk8kQSXYywcMqCar3Fm79U5c4U4i/view?usp=sharing"
            st.subheader("Video URL")
            st.write(f"[Watch Video]({video_url})")

        except Exception as e:
            st.error(f"An error occurred during the search: {e}")