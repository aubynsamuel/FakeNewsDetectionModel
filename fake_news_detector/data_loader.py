import pandas as pd


def load_data(fake_path, true_path):
    """
    Loads fake and true news datasets, combines them, and prepares the content and labels.

    Returns:
        tuple: (list of preprocessed content strings, numpy array of labels, combined DataFrame)
    """
    print("Loading datasets...")
    try:
        df_fake = pd.read_csv(fake_path)
        df_true = pd.read_csv(true_path)
        print(f"Loaded {len(df_fake)} fake and {len(df_true)} true articles")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure '{fake_path}' and '{true_path}' are in the same directory.")
        return None, None, None

    # Prepare data
    df_fake["content"] = (
        df_fake["title"].fillna("") + " " + df_fake["text"].fillna("")
    ).str[:2000]
    df_true["content"] = (
        df_true["title"].fillna("") + " " + df_true["text"].fillna("")
    ).str[:2000]

    df_fake["label"] = 0
    df_true["label"] = 1

    # Combine and shuffle
    df = pd.concat([df_fake[["content", "label"]], df_true[["content", "label"]]])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Remove very short articles
    df = df[df["content"].str.len() >= 50].reset_index(drop=True)
    print(f"Final dataset size: {len(df)} articles")

    return df["content"].tolist(), df["label"].values, df


if __name__ == "__main__":
    texts, labels, _ = load_data()
    if texts:
        print(f"Sample preprocessed text: {texts[0][:100]}...")
        print(f"Sample label: {labels[0]}")
