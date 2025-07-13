# train_unsupervised_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import torch
import pickle
import warnings
import os  # Import os for path creation

warnings.filterwarnings("ignore")

# --- Configuration ---
CLICKBAIT_CSV_PATH = "data/clickbait_data.csv"
NON_CLICKBAIT_CSV_PATH = "data/non_clickbait_data.csv"
MODEL_SAVE_PATH = "models/kmeans_clickbait_model.pkl"
CLUSTER_MAPPING_SAVE_PATH = "models/cluster_id_to_label_mapping.pkl"
# New paths for derived label CSVs
CLICKBAIT_DERIVED_CSV_PATH = "data/clickbait_derived_labels.csv"
NON_CLICKBAIT_DERIVED_CSV_PATH = "data/non_clickbait_derived_labels.csv"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
SAMPLE_SIZE = None  # Use None for full dataset, or an integer for a sample

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch device: {torch_device}")

# Ensure the 'models' and 'data' directories exist
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CLICKBAIT_DERIVED_CSV_PATH), exist_ok=True)


# --- 1. Load Data (for embeddings and cluster interpretation) ---
def load_titles_and_labels(
    clickbait_path: str, non_clickbait_path: str, sample_limit: int = None
):
    """Loads titles from CSVs and returns as lists along with their original labels."""
    all_titles = []
    original_labels = []  # To help interpret clusters later

    try:
        clickbait_df = pd.read_csv(clickbait_path)
        clickbait_titles = clickbait_df["title"].dropna().tolist()
        if sample_limit:
            clickbait_titles = clickbait_titles[:sample_limit]
        print(f"Loaded {len(clickbait_titles)} clickbait titles (original label 1).")
        all_titles.extend(clickbait_titles)
        original_labels.extend([1] * len(clickbait_titles))

        non_clickbait_df = pd.read_csv(non_clickbait_path)
        non_clickbait_titles = non_clickbait_df["title"].dropna().tolist()
        if sample_limit:
            non_clickbait_titles = non_clickbait_titles[:sample_limit]
        print(
            f"Loaded {len(non_clickbait_titles)} non-clickbait titles (original label 0)."
        )
        all_titles.extend(non_clickbait_titles)
        original_labels.extend([0] * len(non_clickbait_titles))

        return all_titles, np.array(original_labels)
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return [], np.array([])


all_titles, original_labels = load_titles_and_labels(
    CLICKBAIT_CSV_PATH, NON_CLICKBAIT_CSV_PATH, SAMPLE_SIZE
)

if not all_titles:
    print("Exiting training due to data loading issues.")
    exit()

print(f"\nTotal titles for clustering: {len(all_titles)}")

# --- 2. Generate Embeddings for all titles ---
print(f"\nInitializing SentenceTransformer model: {EMBEDDING_MODEL_NAME}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=torch_device)

print(f"Generating embeddings for {len(all_titles)} titles...")
all_embeddings = embedding_model.encode(
    all_titles, batch_size=512, show_progress_bar=True, convert_to_numpy=True
)

print(f"All embeddings shape: {all_embeddings.shape}")

# --- 3. Apply K-Means Clustering ---
N_CLUSTERS = 2  # We expect two main types: clickbait and non-clickbait
print(f"\nApplying K-Means clustering with K={N_CLUSTERS}...")
kmeans_model = KMeans(
    n_clusters=N_CLUSTERS, random_state=42, n_init=10
)  # n_init for robustness
cluster_assignments = kmeans_model.fit_predict(all_embeddings)
print("K-Means clustering complete.")

# --- 4. Determine Cluster Mapping (Interpret which cluster is "clickbait") ---
print("\nAnalyzing cluster composition to determine mapping...")
cluster_0_labels = original_labels[cluster_assignments == 0]
cluster_1_labels = original_labels[cluster_assignments == 1]

# Count original clickbait (1) and non-clickbait (0) within each cluster
count_0_in_cluster_0 = np.sum(cluster_0_labels == 0)
count_1_in_cluster_0 = np.sum(cluster_0_labels == 1)
count_0_in_cluster_1 = np.sum(cluster_1_labels == 0)
count_1_in_cluster_1 = np.sum(cluster_1_labels == 1)

print(
    f"Cluster 0: {len(cluster_0_labels)} titles. Original labels: {count_0_in_cluster_0} Non-Clickbait, {count_1_in_cluster_0} Clickbait"
)
print(
    f"Cluster 1: {len(cluster_1_labels)} titles. Original labels: {count_0_in_cluster_1} Non-Clickbait, {count_1_in_cluster_1} Clickbait"
)

# Assign clickbait_cluster_id to the cluster with a higher proportion of original clickbait titles
if count_1_in_cluster_0 / (len(cluster_0_labels) + 1e-6) > count_1_in_cluster_1 / (
    len(cluster_1_labels) + 1e-6
):
    clickbait_cluster_id = 0
    non_clickbait_cluster_id = 1
else:
    clickbait_cluster_id = 1
    non_clickbait_cluster_id = 0

cluster_id_to_label = {
    clickbait_cluster_id: 1,  # Map to 1 for clickbait
    non_clickbait_cluster_id: 0,  # Map to 0 for non-clickbait
}
print(f"\nDetermined Cluster Mapping: {cluster_id_to_label}")
print(f"Cluster ID {clickbait_cluster_id} is mapped to Clickbait (1)")
print(f"Cluster ID {non_clickbait_cluster_id} is mapped to Non-Clickbait (0)")


# --- NEW SECTION: Create new CSVs with derived labels ---
print("\nCreating new CSVs with derived labels...")

# Create a DataFrame for all titles with original and derived labels
df_derived_labels = pd.DataFrame(
    {
        "title": all_titles,
        "original_label": original_labels,
        "predicted_cluster_id": cluster_assignments,
    }
)

# Map cluster IDs back to derived 0/1 labels
df_derived_labels["derived_label"] = df_derived_labels["predicted_cluster_id"].map(
    cluster_id_to_label
)

# Filter based on derived label
df_clickbait_derived = df_derived_labels[df_derived_labels["derived_label"] == 1].drop(
    columns=["predicted_cluster_id"]
)
df_non_clickbait_derived = df_derived_labels[
    df_derived_labels["derived_label"] == 0
].drop(columns=["predicted_cluster_id"])

# Save to CSV
try:
    df_clickbait_derived.to_csv(CLICKBAIT_DERIVED_CSV_PATH, index=False)
    print(
        f"Saved {len(df_clickbait_derived)} titles with derived 'Clickbait' labels to '{CLICKBAIT_DERIVED_CSV_PATH}'"
    )

    df_non_clickbait_derived.to_csv(NON_CLICKBAIT_DERIVED_CSV_PATH, index=False)
    print(
        f"Saved {len(df_non_clickbait_derived)} titles with derived 'Non-Clickbait' labels to '{NON_CLICKBAIT_DERIVED_CSV_PATH}'"
    )

except Exception as e:
    print(f"Error saving derived label CSVs: {e}")


# --- Optional: Visualize the Clusters with original labels and K-Means labels ---
print("\nGenerating PCA visualization of clusters...")
if all_embeddings.shape[1] > 2:
    pca = PCA(n_components=2, random_state=42)
    reduced_embeddings = pca.fit_transform(all_embeddings)
else:
    reduced_embeddings = all_embeddings

plt.figure(figsize=(14, 7))

# Plot with original labels
plt.subplot(1, 2, 1)
plt.scatter(
    reduced_embeddings[original_labels == 1, 0],
    reduced_embeddings[original_labels == 1, 1],
    c="red",
    alpha=0.6,
    s=5,
    label="Original Clickbait",
)
plt.scatter(
    reduced_embeddings[original_labels == 0, 0],
    reduced_embeddings[original_labels == 0, 1],
    c="blue",
    alpha=0.6,
    s=5,
    label="Original Non-Clickbait",
)
plt.title("Embeddings by Original Labels (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot with K-Means assigned clusters
plt.subplot(1, 2, 2)
plt.scatter(
    reduced_embeddings[cluster_assignments == clickbait_cluster_id, 0],
    reduced_embeddings[cluster_assignments == clickbait_cluster_id, 1],
    c="red",
    alpha=0.6,
    s=5,
    label=f"K-Means Clickbait (Cluster {clickbait_cluster_id})",
)
plt.scatter(
    reduced_embeddings[cluster_assignments == non_clickbait_cluster_id, 0],
    reduced_embeddings[cluster_assignments == non_clickbait_cluster_id, 1],
    c="blue",
    alpha=0.6,
    s=5,
    label=f"K-Means Non-Clickbait (Cluster {non_clickbait_cluster_id})",
)
plt.title("Embeddings by K-Means Clusters (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- 5. Save the Trained K-Means Model and Mapping ---
try:
    with open(MODEL_SAVE_PATH, "wb") as file:
        pickle.dump(kmeans_model, file)
    print(f"\nK-Means model saved successfully to '{MODEL_SAVE_PATH}'")

    with open(CLUSTER_MAPPING_SAVE_PATH, "wb") as file:
        pickle.dump(cluster_id_to_label, file)
    print(
        f"Cluster ID to Label mapping saved successfully to '{CLUSTER_MAPPING_SAVE_PATH}'"
    )

except Exception as e:
    print(f"Error saving models: {e}")

print("\nUnsupervised training script finished.")
