# GPT2 Training Code

## Usage

### How to train?

You can train GPT-2 by using as follows:

        $ python -m hfgpt2 train --train_corpus  corpus_train.txt \
                                 --eval_corpus   corpus_eval.txt \
                                 --vocab_path    vocab.vocab \
                                 --batch_train   256 \
                                 --batch_eval    256  \

The detail of command-line usage is as follows:

        usage: gpt2 train [-h] --train_corpus TRAIN_CORPUS --eval_corpus EVAL_CORPUS
                  --vocab_path VOCAB_PATH [--seq_len SEQ_LEN]
                  [--vocab_size VOCAB_SIZE] [--layers LAYERS] [--heads HEADS]
                  [--dims DIMS] [--dropout DROPOUT]
                  [--layer_norm_epsilon LAYER_NORM_EPSILON]
                  [--initializer_range INITIALIZER_RANGE]
                  [--batch_train BATCH_TRAIN] [--batch_eval BATCH_EVAL]
                  [--base_lr BASE_LR] [--wd_rate WD_RATE]
                  [--total_steps TOTAL_STEPS] [--eval_steps EVAL_STEPS]
                  [--save_steps SAVE_STEPS]
                  [--save_model_path SAVE_MODEL_PATH]
                  [--save_checkpoint_path SAVE_CHECKPOINT_PATH]
                  [--from_checkpoint FROM_CHECKPOINT]
                  [--from_pretrained FROM_PRETRAINED] [--use_amp]
                  [--gpus GPUS]
