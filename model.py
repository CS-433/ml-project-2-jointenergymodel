import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from config import DATASET_PATH


def load_dataset(csv_path):
    """
    Load the dataset made of labels and TMD features stored ar csv_path
        - csv_path: String giving the path where the dataset is stored
    Returns features and labels as Pandas DataFrames
    """
    df = pd.read_csv(csv_path)
    labels = df.iloc[:, 0].values.astype(int)
    features = df.iloc[:, 1:].values
    return features, labels


def standardize_features(features):
    """
    Make every feature centered and scaled to unit variance
        - features: Pandas DataFrame
    """
    scaler = StandardScaler()
    return scaler.fit_transform(features)


def create_mlp(input_size, hidden_sizes, output_size=2):
    """
    Defines a function to create a MLP model with variable architecture
        -input_size, output_size : Integers giving the size of input and input
        -hidden_size : List of integers corresponding to the hidden layers widths
    """
    layers = []
    sizes = [input_size] + hidden_sizes + [output_size]
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class TwoStageMLP(nn.Module):
    def __init__(
        self,
        image_size,
        num_other_features,
        k,
        hidden_sizes_stage1,
        hidden_sizes_stage2,
        output_size,
    ):
        super(TwoStageMLP, self).__init__()

        self.image_size = image_size
        self.num_other_features = num_other_features

        # MLP for the persistent image
        self.mlp_stage1 = create_mlp(image_size, hidden_sizes_stage1, output_size=k)

        # Combined MLP for the extracted features and other features
        input_size_stage2 = k + num_other_features
        self.mlp_stage2 = create_mlp(input_size_stage2, hidden_sizes_stage2, output_size)

    def forward(self, features):
        # Split features into image_features and other_features
        image_features = features[:, :self.image_size]
        other_features = features[:, self.image_size:]

        # MLP for the persistent image
        image_output = self.mlp_stage1(image_features)

        # Concatenate image features with other features
        combined_features = torch.cat([image_output, other_features], dim=1)

        # MLP for the combined features
        final_output = self.mlp_stage2(combined_features)

        return final_output


def cross_validation(
    model,
    features,
    labels,
    num_epochs=10,
    batch_size=32,
    learning_rate=0.001,
    num_splits=5,
):
    """
    For a given model and dataset (features, labels) this function performs K_fold cross validation
        -model: Pytorch MLP model with appropriate input and output sizes
        -features, labels = Dataset in the form of two Pandas data-frames
        -num_epoch, batch_size, learning_rate : usual NN parameters
        -num_split : number of splits for the K_fold CV
    Returns a list with the num_splits values of validation F1-score
    """
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    criterion = nn.CrossEntropyLoss()

    all_val_f1_scores = []

    for fold, (train_index, test_index) in enumerate(skf.split(features, labels)):
        train_features, test_features = features[train_index], features[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        train_dataset = TensorDataset(
            torch.tensor(train_features, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.long),
        )
        test_dataset = TensorDataset(
            torch.tensor(test_features, dtype=torch.float32),
            torch.tensor(test_labels, dtype=torch.long),
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        val_f1_scores = []

        for epoch in range(num_epochs):
            model.train()
            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

            model.eval()
            all_predictions = []
            all_true_labels = []
            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    outputs = model(batch_features)
                    _, predicted = torch.max(outputs.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_true_labels.extend(batch_labels.cpu().numpy())

            f1 = f1_score(all_true_labels, all_predictions, average="weighted")
            val_f1_scores.append(f1)
            print(
                f"Fold {fold + 1}, Epoch [{epoch + 1}/{num_epochs}], Test F1-Score: {val_f1_scores[-1]}"
            )

        all_val_f1_scores.append(val_f1_scores)

    return all_val_f1_scores


def hyperparameter_search(
    features,
    labels,
    hidden_size_options_stage1,
    hidden_size_options_stage2,
    ks,
    num_other_features,
    num_splits=5,
    num_epochs=10,
    batch_size=32,
    learning_rate=0.001,
):
    """
    Performs hyperparameters search in terms of width and depth for the TwoStageMLP
        -features, labels, num_splits=5, num_epochs=10, batch_size=32, learning_rate=0.001 : same as cross_validation
        -hidden_size_options_stage1, hidden_size_options_stage2 : lists of tuples of integers giving the sizes of each hidden layer in each architecture to be tested for stage 1 and stage 2. Must be consistent with num_layers_options
    """

    # Consistency check
    best_score = 0
    best_model = None
    best_options = None

    for hidden_size_option_stage1 in hidden_size_options_stage1:
        for k in ks:
          for hidden_size_option_stage2 in hidden_size_options_stage2:
              model = TwoStageMLP(
                  features.shape[1] - num_other_features,
                  num_other_features,
                  k,
                  hidden_size_option_stage1,
                  hidden_size_option_stage2,
                  len(np.unique(labels)),
              )
              print(f"\nHidden Size (Stage 1): {hidden_size_option_stage1}")
              print(f"k: {k}")
              print(f"Hidden Size (Stage 2): {hidden_size_option_stage2}")

              score = np.average(
                  cross_validation(
                      model,
                      features,
                      labels,
                      num_epochs=num_epochs,
                      batch_size=batch_size,
                      learning_rate=learning_rate,
                      num_splits=num_splits,
                  )
              )

              if score > best_score:
                  best_score = score
                  best_model = model
                  best_options = [hidden_size_option_stage1, k, hidden_size_option_stage2]

    print("\nBest Model:")
    print(f"Hidden Size (Stage 1): {best_options[0]}")
    print(f"k: {best_options[1]}")
    print(f"Hidden Size (Stage 2): {best_options[2]}")
    print(f"Test score: {best_score}")
    return best_model


def get_best_model(pers_resolution: int = 100):
    features, labels = load_dataset(DATASET_PATH)
    features = standardize_features(features)
  
    # Compute image size and num_other_features
    image_size = pers_resolution * pers_resolution
    num_other_features = features.shape[1] - image_size

    # Define hyperparameter search space for stage 1 and stage 2
    hidden_size_options_stage1 = [
        [64],
        [128],
        [64, 32],
        [128, 64],
        [128, 64, 32],
        [64, 32, 16],
    ]
    hidden_size_options_stage2 = [
        [16],
        [32],
        [16, 8],
        [32, 16],
        [32, 16, 8],
        [16, 8, 4],
    ]
    ks = [
      1,
      len(np.unique(labels)),
      10,
      pers_resolution,
      pers_resolution * pers_resolution
    ]

    return hyperparameter_search(
        features,
        labels,
        hidden_size_options_stage1,
        hidden_size_options_stage2,
        ks,
        num_other_features,
    )


get_best_model()
