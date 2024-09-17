import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from DataProcessor import Dataprocessor
from Transformer import TransformerRegressor
from trainer import Trainer
from torch.utils.data import DataLoader

seq_length = 110
n_tokens = 4
d_model = 120
n_heads = 8
n_layers = 6
dim_feedforward = 512
layer_norm_eps = 1e-5
dropout = 0 # 0.3
batch_size = 128

SEQUENCE_PATH = "../DeepUTR-main/files/dataset/mRNA_sequences.csv"
EXPR_PATH = "../DeepUTR-main/files/dataset" \
            "/A_plus_normalized_levels.csv"
INDEXES_PATH = "../DeepUTR-main/files/dataset" \
                "/split_to_train_validation_test_disjoint_sets_ids.csv"
def load_model():
    model = TransformerRegressor(seq_length, n_tokens, d_model, n_heads, n_layers,
                                 dim_feedforward, layer_norm_eps, 0)
    model_dict_path = "files/models/6_layers_8head_with_bn/model.pt"
    model_dict = torch.load(model_dict_path, map_location='cpu')
    model.load_state_dict(model_dict)
    return model

def load_data():
    data = Dataprocessor(SEQUENCE_PATH, EXPR_PATH, INDEXES_PATH)
    train_dataset = data.get_train_set()
    validation_dataset = data.get_validation_set()
    return DataLoader(train_dataset, batch_size, shuffle=False)

def predict(model, dataloader):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to('cpu'), targets.to('cpu')

            outputs = model(inputs)

            # Store predictions and targets
            predictions.extend(outputs.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    # Convert lists to numpy arrays for plotting
    predictions = np.array(predictions)
    labels = np.array(labels)
    pd.DataFrame(predictions).to_csv(
        "files/models/6_layers_8head_with_bn/train_pred.csv", index=False)
    pd.DataFrame(labels).to_csv(
        "files/models/6_layers_8head_with_bn/train_labels.csv", index=False)

    return predictions, labels
def plot_predict_vs_true(labels, predictions):
    plt.scatter(labels, predictions)
    # plt.ylim(predictions.min() * 1.1, 2)
    q1 = np.percentile(predictions, 25)
    q3 = np.percentile(predictions, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_preds = predictions[(predictions >= lower_bound) & (predictions
                                                                  <=
                                                                upper_bound)]

    print(f"labels std: {labels.std()}")
    print(f"prediction std: {filtered_preds.std()}")
    plt.ylim(-5, 0)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.show()

def main():
    # model = load_model()
    # data = load_data()
    # prediction, labels = predict(model, data)
    labels = pd.read_csv("files/models/6_layers_8head_with_bn/train_labels"
                         ".csv", header=None).to_numpy()
    predictions = pd.read_csv("files/models/6_layers_8head_with_bn/train_pred"
                              ".csv", header=None).to_numpy()
    plot_predict_vs_true(labels, predictions)


if __name__ == '__main__':
    main()