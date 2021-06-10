#!/bin/sh
# BERT_LARGE_DIR=/fs/classhomes/spring2021/cmsc723/c7230008/BERT_MODEL
# SQUAD_DIR=/fs/classhomes/spring2021/cmsc723/c7230008/squad_dir
OUTPUT_DIR=/fs/class-projects/spring2021/cmsc723/c723g001/squad_project/squad-bert/distil/fine_tune_output
# python run_squad.py \
#   --vocab_file=$BERT_LARGE_DIR/vocab.txt \
#   --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
#   --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
#   --do_train=True \
#   --train_file=$SQUAD_DIR/train-v2.0.json \
#   --do_predict=True \
#   --predict_file=$SQUAD_DIR/dev-v2.0.json \
#   --train_batch_size=24 \
#   --learning_rate=3e-5 \
#   --num_train_epochs=2 \
#   --max_seq_length=384 \
#   --doc_stride=128 \
#   --output_dir=$OUTPUT_DIR \
#   --version_2_with_negative=True
#python /fs/class-projects/spring2021/cmsc723/c723g001/bert/transformers/examples/question-answering/run_qa.py --model_name_or_path 'fine_tune_output' --dataset_name squad_v2   --do_eval   --max_seq_length 384   --doc_stride 128 --overwrite_output_dir --output_dir $OUTPUT_DIR

python /fs/class-projects/spring2021/cmsc723/c723g001/bert/transformers/examples/question-answering/run_qa.py --model_name_or_path 'distilbert-base-uncased' --dataset_name squad_v2   --do_train   --do_eval   --per_device_train_batch_size 12   --learning_rate 3e-5   --num_train_epochs 2   --max_seq_length 384   --doc_stride 128 --overwrite_output_dir --output_dir $OUTPUT_DIR
# python ../gpu.py