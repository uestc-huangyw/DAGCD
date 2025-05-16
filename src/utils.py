from collections import Counter
import re
import string
import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import statistics
# import evaluate
import torch



############################## evaluate function ##############################
## evaluate open-book QA
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def evaluate_em(results, gold_ans_key, pred_ans_key):
    em_score = 0
    correct, wrong = [], []

    for res in results:
        ground_truths = res[gold_ans_key]
        prediction = res[pred_ans_key]
        em = max(exact_match_score(prediction, gt) for gt in ground_truths)
        em_score += em
        if em:
            correct.append(res)
        else:
            wrong.append(res)

    em_score /= len(results)
    return round(em_score, 4), correct, wrong


def evaluate_f1(results, gold_ans_key, pred_ans_key):
    f1_total = 0
    correct, wrong = [], []

    for res in results:
        ground_truths = res[gold_ans_key]
        prediction = res[pred_ans_key]
        f1 = max(f1_score(prediction, gt) for gt in ground_truths)
        f1_total += f1
        if f1 >= 0.5:
            correct.append(res)
        else:
            wrong.append(res)

    mean_f1 = f1_total / len(results)
    return round(mean_f1, 4), correct, wrong


## evaluate summarization
def evaluate_summary(res, ans1='gold_ans', ans2='pred_ans'):
    access_token = ''
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", padding="max_length", truncation=True)
    factkb = AutoModelForSequenceClassification.from_pretrained("bunsenfeng/FactKB", num_labels=2, token=access_token)
    all_fact_score = []
    all_gold, all_pred, all_document = [], [], []

    for d in res:
        document = d['context']
        pred_ans = d[ans2]

        # fact score
        input = [[document, pred_ans]]
        tokens = tokenizer(input, return_tensors="pt", padding="max_length", truncation=True)
        result = torch.softmax(factkb(**tokens).logits, dim = 1)
        fact_score = result[0][1].item()
        all_fact_score.append(fact_score)
        all_pred.append(pred_ans)
        all_document.append(document)
        all_gold.append(d[ans1])


    print("fact_score: ", statistics.mean(all_fact_score))

    # rouge
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=all_pred, references=all_gold)
    print("rouge results: ", results)

    # bert score
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=all_pred, references=all_document, lang="en")
    print("bertscore: ")
    for k, v in results.items():
        if k in ["precision", "recall", "f1"]:
            print(f"{k}: {statistics.mean(v):.4f}")


############################## save and load data ##############################

def load_parquet_data(dataset_path: str):
    data = pd.read_parquet(dataset_path)
    return data.to_dict(orient="records")

def save_as_parquet(path_to_save: str, data_to_save):
    make_dir(path_to_save)
    df = pd.DataFrame(data_to_save)
    df.to_parquet(path_to_save, index=False)
    print(f'file saved in {path_to_save}')

def make_dir(path: str):
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

############################## download model ##############################

def download_model(hf_token, model_name, save_path=''):
    if os.path.exists(save_path):
        return
    if not hf_token: print('please set your huggingface token')
    make_dir(save_path)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, use_auth_token=hf_token)

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

    print(f'{model_name} saved in {save_path}')

