import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import os

# Import sample_data from dataset.py
from dataset import sample_data

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# === Configurations ===
EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L12-v2"
CLASSIFIER_OUTPUT_DIR = "./classifier_output"
CLASSIFIER_PATH = os.path.join(CLASSIFIER_OUTPUT_DIR, "classification_head.pth")

# Hyperparameters for the Classification Head training
BATCH_SIZE = 32
NUM_EPOCHS = 20  # Increased epochs for better training
LEARNING_RATE = 1e-4

# Define your labels and their mapping to numerical IDs
LABEL_MAP = {
    "strongly_supported": 0,
    "mildly_supported": 1,
    "not_supported": 2,
    "contradictory": 3,
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)

# === 1. Load Embedding Model ===
# This model will be used as a feature extractor, its weights will be frozen.
logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embedding_model.eval()  # Set to evaluation mode to freeze weights

# Get embedding dimension
EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()
# We will concatenate claim and content embeddings, so input dimension will be 2 * EMBEDDING_DIM
CLASSIFIER_INPUT_DIM = EMBEDDING_DIM * 2


# === 2. Define the Classification Head (Feed-Forward Neural Network) ===
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Added dropout for regularization
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# === 3. Prepare Dataset ===
def prepare_data(data):
    logger.info("Generating embeddings and preparing dataset...")
    features = []
    labels = []

    for item in data:
        claim = item["claim"]
        content = item["content"]
        label = item["label"]

        # Generate embeddings
        claim_embedding = embedding_model.encode(claim, convert_to_tensor=True)
        content_embedding = embedding_model.encode(content, convert_to_tensor=True)

        # Combine embeddings by concatenation
        combined_embedding = torch.cat((claim_embedding, content_embedding), dim=0)
        features.append(combined_embedding)
        labels.append(LABEL_MAP[label])

    features = torch.stack(features)
    labels = torch.tensor(labels, dtype=torch.long)

    return features, labels


# === 4. Training Function ===
def train_classifier(features, labels):
    logger.info("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ClassificationHead(CLASSIFIER_INPUT_DIM, NUM_LABELS)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info(f"Starting training on {device}...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(
                device
            )

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation phase
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(
                    device
                ), batch_labels.to(device)
                outputs = model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(batch_labels.cpu().numpy())

        val_accuracy = accuracy_score(val_true, val_preds)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_true, val_preds, average="weighted", zero_division=0
        )

        logger.info(
            f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, "
            f"Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}"
        )

    # Save the trained classification head
    os.makedirs(CLASSIFIER_OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), CLASSIFIER_PATH)
    logger.info(f"Classification head saved to {CLASSIFIER_PATH}")

    return model


# === 5. Prediction Function (for later use) ===
def predict_with_classifier(
    claim: str, content: str, trained_classifier: ClassificationHead
) -> dict:
    embedding_model.eval()
    trained_classifier.eval()

    claim_embedding = embedding_model.encode(claim, convert_to_tensor=True)
    content_embedding = embedding_model.encode(content, convert_to_tensor=True)
    combined_embedding = torch.cat(
        (claim_embedding, content_embedding), dim=0
    ).unsqueeze(
        0
    )  # Add batch dimension

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combined_embedding = combined_embedding.to(device)
    trained_classifier.to(device)

    with torch.no_grad():
        outputs = trained_classifier(combined_embedding)
        probabilities = torch.softmax(outputs, dim=1)[
            0
        ]  # Get probabilities for the single sample
        _, predicted_id = torch.max(outputs, 1)

    predicted_label = ID_TO_LABEL[predicted_id.item()]
    all_probabilities = {
        ID_TO_LABEL[i]: prob.item() for i, prob in enumerate(probabilities)
    }

    return {
        "predicted_label": predicted_label,
        "predicted_probability": all_probabilities[predicted_label],
        "all_probabilities": all_probabilities,
    }


# === Main Execution ===
if __name__ == "__main__":
    # Prepare data
    features, labels = prepare_data(sample_data)

    # Train the classifier
    trained_classifier_model = train_classifier(features, labels)

    logger.info("\n--- Example Prediction ---")
    example_claim = "The Earth is flat."
    example_content = "Scientific evidence and observations, including satellite images, confirm the Earth is an oblate spheroid."

    prediction_result = predict_with_classifier(
        example_claim, example_content, trained_classifier_model
    )
    logger.info(f"Claim: {example_claim}")
    logger.info(f"Content: {example_content}")
    logger.info(f"Predicted Label: {prediction_result['predicted_label']}")
    logger.info(f"Confidence: {prediction_result['predicted_probability']:.4f}")
    logger.info("All Probabilities:")
    for label, prob in prediction_result["all_probabilities"].items():
        logger.info(f"  {label}: {prob:.4f}")

    logger.info("\n--- Another Example Prediction ---")
    example_claim_2 = (
        "Regular exercise can improve mental health and reduce symptoms of depression."
    )
    example_content_2 = "Multiple clinical studies have demonstrated that consistent physical activity releases endorphins and neurotransmitters like serotonin, which can significantly alleviate depressive symptoms and improve overall mood regulation."

    prediction_result_2 = predict_with_classifier(
        example_claim_2, example_content_2, trained_classifier_model
    )
    logger.info(f"Claim: {example_claim_2}")
    logger.info(f"Content: {example_content_2}")
    logger.info(f"Predicted Label: {prediction_result_2['predicted_label']}")
    logger.info(f"Confidence: {prediction_result_2['predicted_probability']:.4f}")
    logger.info("All Probabilities:")
    for label, prob in prediction_result_2["all_probabilities"].items():
        logger.info(f"  {label}: {prob:.4f}")
