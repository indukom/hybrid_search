"""Hybrid Search & Retrieval — runnable local version (Parts 1-10)."""

import json
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    SparseVector,
    PointStruct,
    QueryRequest,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    ScoredPoint,
)
from transformers import AutoTokenizer
import fastembed
from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding

print(f"FastEmbed version: {fastembed.__version__}")

# ── Part 5: Load & Prepare Data ──────────────────────────────────────────────

print("\n📦 Loading ESCI dataset (streaming 1000 records)...")
streaming_dataset = load_dataset("tasksource/esci", split="train", streaming=True)
list_data = list(streaming_dataset.take(1000))
dataset = Dataset.from_list(list_data)
dataset = dataset.filter(lambda x: x["product_locale"] == "us")
print(f"Dataset ready: {len(dataset)} US query-product pairs")

source_df = dataset.to_pandas()
df = source_df.drop_duplicates(
    subset=["product_text", "product_title", "product_bullet_point", "product_brand"]
)
df = df.dropna(subset=["product_text", "product_title", "product_bullet_point", "product_brand"])
print(f"Unique products: {len(df)} | Total queries: {len(source_df)}")

sample = df.iloc[0]
print(f"Sample — Title: {sample['product_title']}, Brand: {sample['product_brand']}")

df["combined_text"] = (
    df["product_title"] + "\n" + df["product_text"] + "\n" + df["product_bullet_point"]
)
print(f"Documents to embed: {len(df)}")

# ── Part 6: Generate Embeddings ──────────────────────────────────────────────

sparse_model_name = "prithvida/Splade_PP_en_v1"
dense_model_name = "BAAI/bge-small-en-v1.5"

print("\n⏳ Loading embedding models (first run downloads ~500MB)...")
sparse_model = SparseTextEmbedding(model_name=sparse_model_name, batch_size=32)
dense_model = TextEmbedding(model_name=dense_model_name, batch_size=32)
print("✅ Both models loaded!")


def make_sparse_embedding(texts):
    return list(sparse_model.embed(texts, batch_size=32))

def make_dense_embedding(texts):
    return list(dense_model.embed(texts))


# Demo: inspect a sparse embedding
test_text = "InterviewKickstart is a great place to learn Applied AI!"
sparse_embedding = make_sparse_embedding([test_text])
print(f"\n🔍 Sparse embedding for: '{test_text}'")
print(f"   Non-zero entries: {len(sparse_embedding[0].indices)} / 30,522")


def get_tokens_and_weights(sparse_emb, model_name):
    tokenizer_source = None
    for model_info in SparseTextEmbedding.list_supported_models():
        if model_info["model"].lower() == model_name.lower():
            tokenizer_source = model_info["sources"]["hf"]
            break
    if tokenizer_source is None:
        raise ValueError(f"Model {model_name} not found")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    token_weight_dict = {}
    for i in range(len(sparse_emb.indices)):
        token = tokenizer.decode([sparse_emb.indices[i]])
        weight = sparse_emb.values[i]
        token_weight_dict[token] = weight
    return dict(sorted(token_weight_dict.items(), key=lambda item: item[1], reverse=True))


tokens = get_tokens_and_weights(sparse_embedding[0], sparse_model_name)
original_words = set(test_text.lower().replace("!", "").split())
print(f"\n{'Token':<20} {'Weight':>8}  Source")
print("─" * 55)
for token, weight in tokens.items():
    is_original = any(token.replace("##", "") in word for word in original_words)
    source = "← in original text" if is_original else "← EXPANDED"
    print(f"{token:<20} {weight:>8.4f}  {source}")

# Dense embedding demo
dense_embedding = make_dense_embedding([test_text])
print(f"\nDense vector shape: {dense_embedding[0].shape}")

# Embed full catalog
product_texts = df["combined_text"].tolist()
print(f"\n⏳ Generating sparse embeddings for {len(product_texts)} products...")
df["sparse_embedding"] = make_sparse_embedding(product_texts)
print("✅ Sparse done!")

print("⏳ Generating dense embeddings...")
df["dense_embedding"] = make_dense_embedding(product_texts)
print("✅ Dense done!")

# ── Part 7: Index with Qdrant ────────────────────────────────────────────────

client = QdrantClient(":memory:")
collection_name = "esci"

client.create_collection(
    collection_name,
    vectors_config={
        "text-dense": VectorParams(size=384, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "text-sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
    },
)
print(f"\n✅ Collection '{collection_name}' created")


def make_points(df):
    sparse_vectors = df["sparse_embedding"].tolist()
    product_texts = df["combined_text"].tolist()
    dense_vectors = df["dense_embedding"].tolist()
    rows = df.to_dict(orient="records")
    points = []
    for idx, (text, sv, dv) in enumerate(zip(product_texts, sparse_vectors, dense_vectors)):
        sparse_vector = SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())
        point = PointStruct(
            id=idx,
            payload={"text": text, "product_id": rows[idx]["product_id"]},
            vector={"text-sparse": sparse_vector, "text-dense": dv.tolist()},
        )
        points.append(point)
    return points


points = make_points(df)
client.upsert(collection_name, points)
print(f"✅ {len(points)} points indexed!")

# ── Part 8: Hybrid Search ────────────────────────────────────────────────────

def search(query_text, top_k=10):
    query_sparse = make_sparse_embedding([query_text])
    query_dense = make_dense_embedding([query_text])
    results = client.query_batch_points(
        collection_name=collection_name,
        requests=[
            QueryRequest(query=query_dense[0].tolist(), using="text-dense", limit=top_k, with_payload=True),
            QueryRequest(
                query=SparseVector(indices=query_sparse[0].indices.tolist(), values=query_sparse[0].values.tolist()),
                using="text-sparse", limit=top_k, with_payload=True,
            ),
        ],
    )
    return [results[0].points, results[1].points]


query_text = "revent 80 cfm"
search_results = search(query_text)
dense_results, sparse_results = search_results

print(f"\nQuery: '{query_text}'")
print(f"\n{'='*60}\nDENSE (Semantic) Results — Top 5:\n{'='*60}")
for i, point in enumerate(dense_results[:5]):
    title = point.payload['text'].split('\n')[0][:80]
    print(f"  {i+1}. [Score: {point.score:.4f}] {title}")

print(f"\n{'='*60}\nSPARSE (Lexical/SPLADE) Results — Top 5:\n{'='*60}")
for i, point in enumerate(sparse_results[:5]):
    title = point.payload['text'].split('\n')[0][:80]
    print(f"  {i+1}. [Score: {point.score:.4f}] {title}")

# ── Part 9: Reciprocal Rank Fusion ───────────────────────────────────────────

def rrf(rank_lists, alpha=60, default_rank=1000):
    all_items = set(item for rank_list in rank_lists for item, _ in rank_list)
    item_to_index = {item: idx for idx, item in enumerate(all_items)}
    rank_matrix = np.full((len(all_items), len(rank_lists)), default_rank)
    for list_idx, rank_list in enumerate(rank_lists):
        for item, rank in rank_list:
            rank_matrix[item_to_index[item], list_idx] = rank
    rrf_scores = np.sum(1.0 / (alpha + rank_matrix), axis=1)
    sorted_indices = np.argsort(-rrf_scores)
    return [(list(item_to_index.keys())[idx], rrf_scores[idx]) for idx in sorted_indices]


# Toy example
print("\n── RRF Toy Example ──")
rank_list1 = [("A", 1), ("B", 2), ("C", 3)]
rank_list2 = [("B", 1), ("C", 2), ("D", 3)]
rank_list3 = [("A", 2), ("D", 1), ("E", 3)]
fused = rrf([rank_list1, rank_list2, rank_list3])
for item, score in fused:
    appearances = sum(1 for rl in [rank_list1, rank_list2, rank_list3] if any(i == item for i, _ in rl))
    print(f"  {item}: {score:.6f} (in {appearances}/3 lists)")


def rank_list(search_result):
    return [(point.id, rank + 1) for rank, point in enumerate(search_result)]


dense_rank_list = rank_list(search_results[0])
sparse_rank_list = rank_list(search_results[1])
rrf_rank_list = rrf([dense_rank_list, sparse_rank_list])

print(f"\n🏆 HYBRID RESULTS for: '{query_text}'\n{'='*70}")
fused_records = client.retrieve(collection_name=collection_name, ids=[item[0] for item in rrf_rank_list])
record_by_id = {r.id: r for r in fused_records}

for rank, (item_id, score) in enumerate(rrf_rank_list, 1):
    record = record_by_id[item_id]
    title = record.payload['text'].split('\n')[0][:75]
    in_dense = any(id == item_id for id, _ in dense_rank_list)
    in_sparse = any(id == item_id for id, _ in sparse_rank_list)
    source = "BOTH" if (in_dense and in_sparse) else ("Dense only" if in_dense else "Sparse only")
    print(f"  {rank:>2}. [{source:<12}] {title}")

# ── Part 10: Evaluation ──────────────────────────────────────────────────────

ids = [item[0] for item in rrf_rank_list]
print(f"\n📊 Evaluation for: '{query_text}'\n{'─'*70}")
label_counts = {}
for rank, idx in enumerate(ids, 1):
    label = df.iloc[idx]["esci_label"]
    title = df.iloc[idx]["product_title"][:60]
    emoji = {"Exact": "✅", "Substitute": "🔄", "Complement": "➕", "Irrelevant": "❌"}.get(label, "?")
    print(f"  {rank:<4} {emoji} {label:<12} {title}")
    label_counts[label] = label_counts.get(label, 0) + 1

print(f"\n{'='*70}\nSummary:")
for label, count in sorted(label_counts.items()):
    print(f"   {label}: {count}/{len(ids)} ({count/len(ids)*100:.0f}%)")

print("\n🎉 Done! Parts 1-10 complete.")
