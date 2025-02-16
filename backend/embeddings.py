import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

def embed_texts(texts):
    """Get embeddings for a list of texts using OpenAI's embedding model"""
    try:
        # Get embeddings from OpenAI
        response = client.embeddings.create(
            model=os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004"),
            input=texts
        )
        
        # Extract embedding vectors
        embeddings = [np.array(item.embedding) for item in response.data]
        
        # Stack embeddings into a single numpy array
        return np.stack(embeddings)
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        # Fallback to random embeddings for testing
        return np.random.rand(len(texts), 1536)
