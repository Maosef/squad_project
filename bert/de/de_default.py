
import json
from datasets import load_dataset
path = "/fs/class-projects/spring2021/cmsc723/c723g001/squad_project/doubly-eponymous/doubly_eponymous_easy.json_gender_balanced"
# path = 'doubly_eponymous.json'
with open(path) as f:
  data = json.load(f)

# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
# print(data)

# d = load_dataset('json', data_files={'test':"doubly_eponymous.json_gender_balanced.bert"})
# print(d)
# print(data)
# d = data[1]
# print(data[0])
# print((type(data[0])))
# print(d['name'])
# print(d['question'])
cnt = 0
fn = "doubly_eponymous.json_gender_balanced.bert"
with open('doubly_eponymous.json_gender_balanced.bert', 'w') as outfile:
  for d in data:
    cnt += 1
    d['id'] = cnt
    json.dump(d, outfile)
    outfile.write("\n")
output_dir="/fs/class-projects/spring2021/cmsc723/c723g001/squad_project/squad-bert/de/base_balanced_output"
bashCommand = f"python3 /fs/class-projects/spring2021/cmsc723/c723g001/squad_project/squad-bert/transformers/examples/question-answering/run_qa.py \
--model_name_or_path bert-base-uncased \
--dataset_name {fn}  \
--max_seq_length 384   --doc_stride 128 \
--overwrite_output_dir \
--output_dir {output_dir} \
--do_predict \
--version_2_with_negative"
# import subprocess
import subprocess
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
# export HF_DATASETS_CACHE='/fs/class-projects/spring2021/cmsc723/c723g001/cache' && python /fs/class-projects/spring2021/cmsc723/c723g001/squad_project/squad-bert/transformers/examples/question-answering/run_qa.py \
# --model_name_or_path '../fine_tune_output' \
# --dataset_name $fn  \
# --max_seq_length 384   --doc_stride 128 \
# --overwrite_output_dir \
# --output_dir $output_dir \
# --do_predict \
# --version_2_with_negative