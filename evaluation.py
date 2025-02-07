import tqdm
import numpy as np

def ranking_metrics_at_k(model, train_user_items, test_user_items, K=10, show_progress=True):
    """
    Calculates ranking metrics (Precision@K, MAP@K, NDCG@K, AUC) for a trained model.

    Parameters:
        model : Trained ALS model (or other Implicit model).
        train_user_items : csr_matrix
            User-item interaction matrix used for training.
        test_user_items : csr_matrix
            User-item interaction matrix for evaluation.
        K : int
            Number of items to evaluate.
        show_progress : bool
            Show a progress bar during evaluation.

    Returns:
        dict : Dictionary with precision, MAP, NDCG, and AUC scores.
    """

    # Ensure matrices are in CSR format
    train_user_items = train_user_items.tocsr()
    test_user_items = test_user_items.tocsr()

    num_users, num_items = test_user_items.shape
    relevant = 0
    total_precision_div = 0
    total_map = 0
    total_ndcg = 0
    total_auc = 0
    total_users = 0

    # Compute cumulative gain for NDCG normalization
    cg = 1.0 / np.log2(np.arange(2, K + 2))  # Discount factor
    cg_sum = np.cumsum(cg)  # Ideal DCG normalization

    # Get users with at least one item in the test set
    users_with_test_data = np.where(np.diff(test_user_items.indptr) > 0)[0]

    # Progress bar
    progress = tqdm.tqdm(total=len(users_with_test_data), disable=not show_progress)

    batch_size = 1000
    start_idx = 0

    while start_idx < len(users_with_test_data):
        batch_users = users_with_test_data[start_idx:start_idx + batch_size]
        recommended_items, _ = model.recommend(batch_users, train_user_items[batch_users], n=K)
        start_idx += batch_size

        for user_idx, user_id in enumerate(batch_users):
            test_items = set(test_user_items.indices[test_user_items.indptr[user_id]:test_user_items.indptr[user_id + 1]])
            
            if not test_items:
                continue  # Skip users without test data

            num_relevant = len(test_items)
            total_precision_div += min(K, num_relevant)

            ap = 0
            hit_count = 0
            auc = 0
            idcg = cg_sum[min(K, num_relevant) - 1]  # Ideal Discounted Cumulative Gain (IDCG)
            num_negative = num_items - num_relevant

            for rank, item in enumerate(recommended_items[user_idx]):
                if item in test_items:
                    relevant += 1
                    hit_count += 1
                    ap += hit_count / (rank + 1)
                    total_ndcg += cg[rank] / idcg
                else:
                    auc += hit_count  # Accumulate hits for AUC calculation

            auc += ((hit_count + num_relevant) / 2.0) * (num_negative - (K - hit_count))
            total_map += ap / min(K, num_relevant)
            total_auc += auc / (num_relevant * num_negative)
            total_users += 1
        
        progress.update(len(batch_users))

    progress.close()

    # Compute final metrics
    precision = relevant / total_precision_div if total_precision_div > 0 else 0
    mean_ap = total_map / total_users if total_users > 0 else 0
    mean_ndcg = total_ndcg / total_users if total_users > 0 else 0
    mean_auc = total_auc / total_users if total_users > 0 else 0

    return {
        "precision": precision,
        "map": mean_ap,
        "ndcg": mean_ndcg,
        "auc": mean_auc
    }
