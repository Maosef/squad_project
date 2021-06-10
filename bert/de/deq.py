# naive method: for each gold answer, if >=50% of its tokens are in model answer, then it's correct
import spacy
nlp = spacy.blank("en")
def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]

def evaluate_doubly_eponymous(model_ans, ground_truth):
    
    if model_ans == 'no answer': return 0, []
        
    model_ans_tokens = set((a.lower()) for a in word_tokenize(model_ans))
    num_matches = 0
    missing_answers = []
    # print(model_ans_tokens)
    
    for ans in ground_truth:
        tokens = word_tokenize(ans['name'].lower())
        # print(tokens)
        ans_overlap = 0
        # if most words of gold answer are in model answer then we count it as correct
        for token in tokens:
            if token in model_ans_tokens:
                ans_overlap += 1
                print(ans['gender']=='female')
        if ans_overlap >= .5:
            num_matches += 1
        else: missing_answers.append(ans)
    # print(missing_answers)
    return num_matches / len(ground_truth), missing_answers
    
    # formally evaluate doubly eponymous
# print(3)
import traceback,json
# print(3)
doubly_eponymous_path = '/fs/class-projects/spring2021/cmsc723/c723g001/squad_project/doubly-eponymous/doubly_eponymous_easy.json_gender_balanced'
# bert_output_path = 'squad_balanced_output/test_predictions.json'
bert_output_path = 'base_balanced_output/test_predictions.json'

cnt = 0
ans = 0
from collections import Counter

all_answers = []
all_missing_answers = []
# missing_answer_counter = Counter()
# answer_counter = Counter()
av_acc = 0
num_examples = 256
num_examples_no_error = 0

with open(doubly_eponymous_path) as g:
    with open(bert_output_path) as b:
        # print(json.load(b))
        b_data = json.load(b)
        de_data = json.loads(g.readline())
        for i,example in enumerate(de_data):
            try:
                # model_ans = answer_question(example['context'], example['question'])
                model_ans = b_data[f'{i+1}']
                # print(model_ans)
                ground_truth = example['options']
                # print(ground_truth)
    #             print(f"question: {example['question']}, \n model answer: {model_ans}, \n ground truth: {example['options']}")
                acc, missing_answers = evaluate_doubly_eponymous(model_ans, ground_truth)
                all_missing_answers += missing_answers
                all_answers += ground_truth
                if acc != 0:
                    ans += 1
                cnt += 1
                print(f"question: {example['question']}, \n model answer: {model_ans}, \n ground truth: {example['options']}\n accuracy: {acc}")


                # raise Exception
            except Exception as e:

                raise Exception
                # pass
                print(e.__traceback__)
                traceback.print_exc()
                print(traceback.format_exc())
                

print(cnt)
print(ans)

gender_counter = Counter()
nationality_counter = Counter()

gender_counter_missing = Counter()
nationality_counter_missing = Counter()

for ans in all_missing_answers:
    gender_counter_missing.update([ans['gender']])
    nationality_counter_missing.update([ans['nationality']])
for ans in all_answers:
    gender_counter.update([ans['gender']])
    nationality_counter.update([ans['nationality']])

print('gender counts: ', gender_counter)
print('missing gender counts: ', gender_counter_missing)

for gender in ['male','female']:
    acc = 1 - (gender_counter_missing[gender] / gender_counter[gender])
    print(f'accuracy for {gender}: {round(acc, 3)}')