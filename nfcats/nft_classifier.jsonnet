local transformer_model = "deepset/roberta-base-squad2";
local transformer_dim = 768;

{
  "dataset_reader": {
    "type": "csv_text_label",
    "label_field": "category",
    "text_field": "question",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "add_special_tokens": true
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": 512
      }
    }
  },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALID_DATA_PATH"),
  "model": {
    "type": "nfq_cats_classifier",
    "embedder": {
        "type": "trainable",
        "text_field_embedder": {
          "token_embedders": {
            "tokens": {
              "type": "pretrained_transformer",
              "model_name": transformer_model,
              "max_length": 512,
              "train_parameters": true
            }
          }
        },
        "seq2vec_encoder": {
           "type": "bert_pooler",
           "pretrained_model": transformer_model,
           "dropout": 0.0,
        }
    },
    "feedforward": {
      "input_dim": transformer_dim,
      "num_layers": 2,
      "hidden_dims": [768, 512],
      "activations": "mish",
      "dropout": 0.6
    },
    "dropout": 0.3,
  },
  "data_loader": {
    "batch_sampler": {
      "type": "balanced",
      "num_classes_per_batch": 8,
      "num_examples_per_class": 8
    },
  },
  "validation_data_loader": {
    "batch_size": 128,
  },
  "trainer": {
    "num_epochs": 100,
    "patience": 5,
    "validation_metric": "+weighted_f1",
    "cuda_device": 0,
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-5,
      "weight_decay": 0.1,
    },
    "callbacks": ["wandb"],
  }
}
