import os
import sys

sys.pycache_prefix = os.path.expanduser("~/.cache/python-pycache")
# os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from SemanticCache import SemanticCache
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
PATH = "CachedText.json"

def embed_text(text: str) -> list[float]:
    return model.encode(text).tolist()

cache = SemanticCache(
    embedding_fn=embed_text,
    threshold=0.80,
    max_size=1000,
)

cache.load_cache_from_json(PATH)

query_1 = "What is the capital of France?"
query_2 = "How to train a model?"
query_veri = "The Verilog code defines a module called invert that takes a single input i and produces an output o that is the logical negation (inversion) of i. If i is 1, o will be 0, and if i is 0, o will be 1."

score_list, docs_list = cache.get_highest_sim_pair(query_veri, 5)

print(f"Query: {query_veri}")

for score, doc in zip(score_list, docs_list):
    print(f"Score: {score:.4f}, Doc: {doc['query']}")
print("==================\n")

    
    
    
    
