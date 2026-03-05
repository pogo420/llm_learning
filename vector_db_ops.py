import chromadb
from datetime import datetime

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="reviews",
    metadata={
        "description": "Product Review",
        "created": str(datetime.now())
    }
)

collection.add(
    documents=[
        "The delivery was fast and product was awesome",
        "Not able to increase brightness of tv so returned it",
        "Size of shoe was small",
        "Great customer support. Issue resolved"
    ],
    ids=[
        "id1", "id2", "id3", "id4"
    ],
    metadatas=[
        {"product_category": "electronics", "rating": 5},
        {"product_category": "electronics", "rating": 2},
        {"product_category": "apparel", "rating": 3},
        {"product_category": "services", "rating": 4}
    ]
)

print(f"All embeddings: {collection.peek().get("embeddings")}")

result = collection.query(
    query_texts=["fast shipping"],
    n_results=1,
    where={"product_category": "electronics"}  # for filtering in metadata
)
print(result)
