#!/bin/sh
OUTPUT_DIR=/fs/class-projects/spring2021/cmsc723/c723g001/squad_project/squad-bert/de/balanced_output_base
python /fs/class-projects/spring2021/cmsc723/c723g001/squad_project/squad-bert/transformers/examples/question-answering/run_qa.py \
--model_name_or_path 'bert-base-uncased' \
--dataset_name '/fs/class-projects/spring2021/cmsc723/c723g001/squad_project/squad-bert/de/doubly_eponymous.json_gender_balanced.bert'  \
--max_seq_length 384   --doc_stride 128 \
--overwrite_output_dir \
--output_dir $OUTPUT_DIR \
--do_predict \
--version_2_with_negative \
# --max_val_samples 10