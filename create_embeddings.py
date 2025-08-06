"""
SKIN GENIUS - EMBEDDING GENERATOR v1.0
======================================
Purpose:
- Converts ingredients.csv → ChromaDB embeddings.

Steps:
1. Loads CSV (name, benefits, properties).
2. Generates embeddings (all-MiniLM-L6-v2).
3. Stores in ChromaDB with metadata.

Output:
- ingredients_embeddings.db (for retriever.py)

Note:
- Disables Chroma telemetry for privacy.
- Tests with pre-computed embeddings.
"""

#I did some changes, so that my data does not be saved repeatedly.

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import os
import warnings

# Mute all warnings (optional but cleaner output)
warnings.filterwarnings("ignore")

# Disable Chroma telemetry and ONNX downloads
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_DISABLE_ONNX"] = "True"

#path
CSV_PATH = "data/ingredients.csv"
DB_PATH = "data/ingredients_embeddings.db"

# Loading CSV
print(" Loading CSV...")
df = pd.read_csv(CSV_PATH)

#Combine all text for embeddings
text_to_embed = df["name"] + " " + df["benefits"] + " " + df["properties"]

#Load embedding model
print(" Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

#Generate embeddings
print(" Generating embeddings (this takes a few minutes)...")
embeddings = model.encode(text_to_embed, show_progress_bar=True)

#storing embedding data into chromaDB
print("Storing to chromaDB .....")
client = chromadb.PersistentClient(path=DB_PATH)
# Delete existing collection (if any)
try:
    client.delete_collection("ingredients")
    print(" Deleted old collection.")
except:
    print(" No existing collection found.")

collection = client.create_collection(
    name="ingredients",
    embedding_function=None
)

# Adding data into db
collection.add(
    ids=[str(i) for i in range(len(df))],
    embeddings=embeddings.tolist(),
    documents=df["name"].tolist(),
    metadatas=df[["benefits", "properties", "avoid_with"]].to_dict("records")
)

print(" Done! Embeddings saved to:", DB_PATH)

#  SAFE TEST: Use pre-computed embeddings only
print("\n Testing with PRE-COMPUTED EMBEDDINGS...")
test_embedding = embeddings[0].tolist()  # Use first ingredient's embedding
results = collection.query(
    query_embeddings=[test_embedding],  # Bypasses Chroma's text processing
    n_results=3
)
print("Top matches:", results["documents"][0])

# Optional: Show what a real query would return
print("\n For text queries later, use these ingredients:")
print("Example matches for 'brightening':",
      [x for x in df["name"] if "brighten" in df[df["name"]==x]["benefits"].values[0].lower()][:3])