#JIT-AGM-replication-package

This repository contains source code that we used to perform experiment in paper titled "JIT-AGM: Just-in-Time Defect Prediction and Localization via Commit-Aware Attention Guided by LLM-Augmented Messages".
Please follow the steps below to reproduce the result.

## Environment Setup

Run the following command in terminal (or command line) to prepare virtual environment

```shell
conda env create --file requirements.yml
conda activate jitagm
```
### **JIT-AGM Implementation**

To train JIT-AGM, run the following command:

```shell
python -m JIT_AGM.concat.run \
    --output_dir=model/jitagm/saved_models_concat/checkpoints \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_train \
    --train_data_file data/changes_train_ger_new.jsonl data/features_train.pkl \
    --eval_data_file data/changes_valid_ger_new.jsonl  data/features_valid.pkl\
    --test_data_file data/changes_test_ger_new.jsonl data/features_test.pkl\
    --epoch 50 \
    --max_seq_length 512 \
    --max_msg_length 64 \
    --train_batch_size 24 \
    --eval_batch_size 128 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --feature_size 14 \
    --patience 10 \
    --seed 42 2>&1| tee model/jitcf/saved_models_concat_cl/train.log

```

To obtain the evaluation, run the following command:

```shell
python -m JIT_AGM.concat.run \
    --output_dir=model/jitagm/saved_models_concat/checkpoints \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_test \
    --train_data_file data/changes_train_ger_new.jsonl data/features_train.pkl \
    --eval_data_file data/changes_valid_ger_new.jsonl  data/features_valid.pkl\
    --test_data_file data/changes_test_ger_new.jsonl data/features_test.pkl\
    --epoch 50 \
    --max_seq_length 512 \
    --max_msg_length 64 \
    --train_batch_size 256 \
    --eval_batch_size 25 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --only_adds \
    --buggy_line_filepath=data/jitfine/changes_complete_buggy_line_level.pkl \
    --seed 42 2>&1 | tee model/jitfine/saved_models_concat/test.log
```


