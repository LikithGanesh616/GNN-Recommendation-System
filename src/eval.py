"""
Evaluation module for GNN recommendation model: computes Recall@K and MRR on a sampled test set.

Functions:
- evaluate(): loads checkpoint, reconstructs graph, performs forward pass, and prints metrics.
"""
import torch
import random
import numpy as np
from torch_geometric.data import HeteroData

from src.graph import build_graph
from src.model import RecSysGNN


def evaluate(
    refs_path: str = "data/processed/refs.parquet",
    inter_path: str = "data/processed/interactions.parquet",
    checkpoint_path: str = "models/rec_sage.pt",
    k: int = 10,
    num_samples: int = 1000,
    num_neg: int = 100
):
    """
    Evaluate the trained GNN on link prediction metrics.

    Args:
        refs_path: path to referrals Parquet (unused here)
        inter_path: path to interactions Parquet with user-product edges
        checkpoint_path: path to saved checkpoint
        k: Recall@K threshold
        num_samples: number of positive edges to sample for evaluation
        num_neg: number of negative samples per positive
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild graph
    data, user_map, product_map = build_graph(refs_path, inter_path)
    metadata = (data.node_types, data.edge_types)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Initialize embeddings and model
    num_users = data['user'].num_nodes
    num_products = data['product'].num_nodes
    user_emb = torch.nn.Embedding(num_users, ckpt['user_emb']["weight"].shape[1]).to(device)
    prod_emb = torch.nn.Embedding(num_products, ckpt['prod_emb']["weight"].shape[1]).to(device)
    user_emb.load_state_dict(ckpt['user_emb'])
    prod_emb.load_state_dict(ckpt['prod_emb'])

    model = RecSysGNN(metadata, hidden_channels=ckpt['model_state_dict'][list(ckpt['model_state_dict'].keys())[0]].shape[0],
                      out_channels=prod_emb.embedding_dim).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Prepare edge index and features
    edge_index = data['user', 'buys', 'product'].edge_index.to(device)
    x_dict = {
        'user': user_emb.weight,
        'product': prod_emb.weight
    }

    # Forward pass
    with torch.no_grad():
        out_dict = model(x_dict, data.edge_index_dict)

    # Sample positives
    src_all, dst_all = edge_index
    total_edges = src_all.size(0)
    sample_indices = random.sample(range(total_edges), min(num_samples, total_edges))

    hits = 0
    mrr_total = 0.0

    for idx in sample_indices:
        u = src_all[idx]
        pos_p = dst_all[idx]
        pos_score = (out_dict['user'][u] * out_dict['product'][pos_p]).sum().item()

        # Negative samples for same user
        neg_ps = torch.randint(0, num_products, (num_neg,), device=device)
        neg_scores = (out_dict['user'][u].unsqueeze(0) * out_dict['product'][neg_ps]).sum(dim=1).cpu().numpy()

        # Combine and compute rank
        scores = np.concatenate(([pos_score], neg_scores))
        ranks = (-scores).argsort()
        rank_pos = int(np.where(ranks == 0)[0][0]) + 1

        if rank_pos <= k:
            hits += 1
        mrr_total += 1.0 / rank_pos

    recall_at_k = hits / len(sample_indices)
    mrr = mrr_total / len(sample_indices)

    print(f"Recall@{k}: {recall_at_k:.4f}")
    print(f"MRR: {mrr:.4f}")


if __name__ == '__main__':
    evaluate()
