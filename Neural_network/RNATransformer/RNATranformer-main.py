from Transformer import TransformerRegressor
from DataProcessor import Dataprocessor
from trainer import Trainer


SEQUENCE_PATH = "../DeepUTR-main/files/dataset/mRNA_sequences.csv"
EXPR_PATH = "../DeepUTR-main/files/dataset" \
            "/A_plus_normalized_levels.csv"
INDEXES_PATH = "../DeepUTR-main/files/dataset" \
                "/split_to_train_validation_test_disjoint_sets_ids.csv"


def main():

    # Create Dataset
    data = Dataprocessor(SEQUENCE_PATH, EXPR_PATH, INDEXES_PATH)
    train_dataset = data.get_train_set()
    validation_dataset = data.get_validation_set()
    test_dataset = data.get_test_set()

    # model params
    seq_length = 110
    n_tokens = 4
    d_model = 128
    n_heads = 8
    n_layers = 4
    dim_feedforward = 512
    layer_norm_eps = 1e-4
    dropout = 0  # 0.3

    # trainer params - TODO change criteria function and maybe different reg
    #  factor.
    batch_size = 64
    lr = 1e-4
    reg_factor = 0.03

    # Initialize the model
    model = TransformerRegressor(seq_len=seq_length, n_tokens=n_tokens,
                                 d_model=d_model,
                                 n_heads=n_heads, n_layers=n_layers,
                                 dim_feedforward=dim_feedforward,
                                 dropout=dropout,
                                 layer_norm_eps=layer_norm_eps)

    trainer = Trainer(model, train_dataset, validation_dataset, test_dataset,
                      batch_size=batch_size, lr=lr, reg_factor=reg_factor)

    model = trainer.train()

    trainer.save_model("files/models")  # TODO


if __name__ == '__main__':
    main()
