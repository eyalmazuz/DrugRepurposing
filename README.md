# DrugRepurposing

## Data
The data is located under the data folder:

1. labels_training_set_w_drugbank_id.csv- contains the data collated manually of drugs that are classified as anti-cancer and those that have indication to not have anti-cancer activity
2. all_data_infer_labels_preds_w_drug_bank_info_no_features.csv- all drugs in DrugBank excluding those that were used to train the model

## Code
The anti-cancer model was train using the chemprop library:
https://github.com/chemprop/chemprop/

To reproduce the anti-cancer without addtional features run the following command from the command-line:
```
chemprop_train --data_path ./data/labels_training_set_w_drugbank_id.csv \
--dataset_type classification \
--extra_metrics prc-auc \
--split_type scaffold_balanced \
--ignore_columns drugBank_id \
--split_sizes 0.9 0.1 0.0 \
--save_dir ./models/anti-cancer \
--smiles_column Smiles \
--save_preds \
--save_smiles_splits \
--show_individual_scores \
--num_workers 0 \
--batch_size 512 \
--separate_test_path ./data/all_data_infer_labels_preds_w_drug_bank_info_no_features.csv \
--no_features_scaling --features_generator rdkit_2d_normalized \
--config_path ./data/full_data_hyperparams_w_rkdit.json
```


To reproduce the anti-cancer with additional features run the multimodal learning code using the ``code/multimodal_learning/interactions_main.py`` to train a DDI model
then train the chemprop model using ``code/multimodal_learning/target_main.py with the task ``cancer`` and uncomment lines 111 and 112 to add the addtional features


## Predict with chemprop model

to predict with the chemprop model, please refer to https://github.com/chemprop/chemprop/#predicting on how to use saved trained models

