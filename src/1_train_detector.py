import spacy
import utils
from argparse import ArgumentParser
import random
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch
from torch.nn import functional as F
import os


class Model:
    def __init__(self, model_name: str):
        self.model, self.tokenizer = self.load_model_and_tokenizer(model_name)
        self.device = self.model.device


    def load_model_and_tokenizer(self, model_name):
        print(f"Loaded models: {model_name}")

        model_path = f'../models/{model_name}'

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto'
        ).eval()

        config = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = min(2048, config.max_position_embeddings)

        return model, tokenizer


    def print_distribution_topk(self, probs, topn):
        top_values, top_indices = torch.topk(probs, topn)
        top_values = top_values.squeeze(0)
        top_indices = top_indices.squeeze(0)
        # ids to tokens
        for topk, (val, ids) in enumerate(zip(top_values, top_indices)):
            ids = ids.item()
            token = self.tokenizer.decode([ids])
            print(f'rank{topk + 1},  value:{val}, token:{token}, ids:{ids}')


    def get_memory_ans(self, datas):
        ''' generate answers without context '''
        pre_datas = []
        for i, data in enumerate(tqdm(datas[:])):
            question = data['question']
            context = data['context']
            gold_ans = list(data['gold_ans'])

            tgt_len = max(len(self.tokenizer.encode(ans, add_special_tokens=False)) for ans in gold_ans)
            # get memory ans
            memory_ans = self.generate(question=question, gen_max_length=tgt_len)
            if memory_ans is not None:
                new_data = {"idx": i, "subset": data['subset'], "question": question, "context": context,
                            "gold_ans": gold_ans, "memory_ans": memory_ans}
                pre_datas.append(new_data)
        return pre_datas


    def get_context_ans(self, datas):
        ''' generate answers with context '''
        pre_datas = []
        for i, data in enumerate(tqdm(datas[:])):
            question = data['question']
            context = data['context']
            gold_ans = list(data['gold_ans'])

            tgt_len = max(len(self.tokenizer.encode(ans, add_special_tokens=False)) for ans in gold_ans)
            # get context ans
            context_ans = self.generate(question=question, context=context, gen_max_length=tgt_len)
            if context_ans is not None:
                new_data = {"idx": i, "subset": data['subset'], "question": question, "context": context,
                            "gold_ans": gold_ans, "memory_ans": data['memory_ans'], "context_ans": context_ans}
                pre_datas.append(new_data)
        return pre_datas


    def generate(self, question: str, context: str = None, gen_max_length: int = 10):
        if context: prompt = f'Given the following information:{context}\nAnswer the following question based on the given information with one or few words: {question}\nAnswer:'
        else: prompt = f'Answer the following question based on your internal knowledge with one or few words: {question}\nAnswer:'

        tokenized_inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length - gen_max_length).to(self.device)
        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']

        with torch.no_grad():
            generated_tokens = []
            for cur_len in range(gen_max_length):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
                next_token_logits = outputs.logits[:, -1, :]
                token_level_probability_distribution = F.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(token_level_probability_distribution, dim=-1)

                if next_token == self.tokenizer.eos_token_id:
                    break

                tokens_to_add = next_token
                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(0)], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.tensor([[1]], device=self.device)], dim=-1)
                generated_tokens.append(tokens_to_add)

        if len(generated_tokens) > 0:
            generated_tokens = torch.cat(generated_tokens)
            generation = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return generation
        else:
            return None


    def locate_context_position(self, offset_mapping, prompt, context):
        """
        Locate the start and end token indices of the context string within input_ids based on offset_mapping and the context string in the prompt.
        Parameters:
            offset_mapping: A list or tensor containing the character start and end positions
                            of each token in the prompt, formatted as
                            [(start1, end1), (start2, end2), ...].
            prompt: The original full prompt string.
            context: The substring to be located.

        Returns:
            (start_token_idx, end_token_idx) - The start and end token indices of the context within input_ids.
            If the context cannot be found, returns (None, None).
        """

        # Locate the character position of context in the prompt
        start_char = prompt.find(context)
        if start_char == -1:
            return None, None  # If target string not fuond
        end_char = start_char + len(context)

        token_starts = offset_mapping[:, 0]
        token_ends = offset_mapping[:, 1]

        valid_tokens = (token_ends > start_char) & (token_starts < end_char)
        token_indices = np.nonzero(valid_tokens)[0]

        # return the context starting and ending token indices
        return token_indices[0], token_indices[-1]



    def get_unique_attn(self, raw_attention):
        # "In the case of multiple attention values for the same token_ids,
        # we use the average to represent the attention of that token."

        unique_context_ids, inverse_indices = torch.unique(self.context_ids, return_inverse=True)
        unique_context_len = len(unique_context_ids)
        unique_attention = torch.zeros((raw_attention.shape[0], unique_context_len), device=self.device)  # [num_dist, unique_len]

        for idx in range(unique_context_len):
            mask = (inverse_indices == idx)
            avg = raw_attention[:, mask].mean(dim=1)
            unique_attention[:, idx] = avg

        return unique_context_ids, unique_attention


    def tokenize_and_get_ids(self, entities):
        return [
            self.tokenizer(entity, return_tensors="pt", add_special_tokens=False)['input_ids'].squeeze(0).to(self.device)
            for entity in entities
        ]


    def get_rank_in_distribution(self, tensor):
        ''' Obtain the ranking of each token_ids in the current token-level probability distribution. '''
        sorted_indices = torch.argsort(-tensor, dim=1)
        ranks = torch.argsort(sorted_indices, dim=1) + 1.0
        return ranks


    def attention_ratio_vector(
            self,
            question,
            context=None,
            max_length=10,
            gold_ans=None,
            question_entities=None,
            context_entities=None
    ):
        prompt = f'Given the following information:{context}\nAnswer the following question based on the given information with one or few words: {question}\nAnswer:'
        tokenized_inputs = self.tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt").to(self.device)
        input_ids = tokenized_inputs['input_ids']  # [batch_size, token_lens]
        attention_mask = tokenized_inputs['attention_mask']
        offset_mapping = tokenized_inputs['offset_mapping'][0]
        if torch.is_tensor(offset_mapping):
            offset_mapping = offset_mapping.cpu().numpy()
        else:
            offset_mapping = np.array(offset_mapping)

        if input_ids.shape[-1] > 2048: return None, None

        context_start, context_end = self.locate_context_position(offset_mapping, prompt, context)
        self.context_ids = input_ids.squeeze(0)[context_start: context_end + 1]

        # Tokenize gold answer
        gold_ans_ids = self.tokenizer(gold_ans, return_tensors="pt", add_special_tokens=False)['input_ids'].squeeze(0).to(self.device)

        # get question and context entities ids
        question_entities_ids = self.tokenize_and_get_ids(question_entities)
        context_entities_ids = self.tokenize_and_get_ids(context_entities)

        # get non-entities ids
        excluded_ids = set(gold_ans_ids.tolist())
        for entity_ids in question_entities_ids + context_entities_ids:
            excluded_ids.update(entity_ids.tolist())
        excluded_ids_tensor = torch.tensor(list(excluded_ids), device=self.device)
        other_tokens_ids = input_ids[~torch.isin(input_ids, excluded_ids_tensor)]

        output = {}

        with torch.no_grad():
            generated_tokens = []
            idx = 0
            for cur_len in range(max_length):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    return_dict=True,
                )
                next_token_logits = outputs.logits[:, -1, :]
                generation_dist = F.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(generation_dist, dim=-1)


                if next_token == self.tokenizer.eos_token_id:
                    break
                tokens_to_add = next_token
                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(0)], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.tensor([[1]], device=self.device)], dim=-1)
                generated_tokens.append(tokens_to_add)

                if (idx >= len(gold_ans_ids)) or (0 < idx < len(gold_ans_ids) and next_token.item() != gold_ans_ids[idx].item()):
                    break
                elif next_token.item() == gold_ans_ids[idx].item():
                    attentions = outputs.attentions  # num_layers * (batch_size, num_heads, seq_length, seq_length)
                    num_layer, num_head = len(attentions), attentions[0].shape[1]
                    att_last_row = torch.cat([layer[:, :, -1, context_start:context_end + 1] for layer in attentions], dim=1)
                    raw_attention_val = att_last_row.view(num_layer * num_head, -1)  # (num_layer * num_head, context_len)
                    rank = self.get_rank_in_distribution(generation_dist).squeeze(0)


                    # get unique attention
                    unique_context_ids, unique_attention_val = self.get_unique_attn(raw_attention_val)  # [num_layer * num_head, len(unique_context_ids)]
                    # calculate attention ratio
                    unique_context_attention_sum = unique_attention_val.sum(dim=1, keepdim=True).clamp(min=1e-8)
                    attention_ratio = unique_attention_val / unique_context_attention_sum
                    attention_ratio = attention_ratio.transpose(0,1)  # [len(unique_context_ids), num_layer * num_head]
                    attention_ratio = torch.nan_to_num(attention_ratio, nan=0.0)


                    # 1. Positive samples: gold answer tokens
                    gold_mask = unique_context_ids == gold_ans_ids[idx]
                    gold_ans_ratio = None
                    if gold_mask.any():
                        gold_ans_ratio = attention_ratio[gold_mask].cpu().numpy()


                    # 2. Negative samples: context entity tokens (Entities in context that are not related to the answer)
                    entities_ratio = []
                    entities_rank = []
                    for entity_ids in context_entities_ids:
                        mask = unique_context_ids == entity_ids[0]
                        if mask.any():
                            entities_ratio.append(attention_ratio[mask].cpu().numpy())
                            entities_rank.append(rank[entity_ids[0]].item())


                    # 3. Negative samples: non-entity tokens (Function words or symbols)
                    non_entity_ratio = []
                    non_entity_rank = []
                    for ids in other_tokens_ids:
                        mask = unique_context_ids == ids
                        if mask.any():
                            non_entity_ratio.append(attention_ratio[mask].cpu().numpy())
                            non_entity_rank.append(rank[ids].item())


                    output['positive_ratio'] = gold_ans_ratio
                    output['negative_entity_ratio'] = np.concatenate(entities_ratio, axis=0) if entities_ratio else np.array(entities_ratio)
                    output['negative_entity_rank'] = np.array(entities_rank)
                    output['negative_non_entity_ratio'] = np.concatenate(non_entity_ratio, axis=0) if non_entity_ratio else np.array(non_entity_ratio)
                    output['negative_non_entity_rank'] = np.array(non_entity_rank)
                    idx += 1

        return torch.cat(generated_tokens), output




def sample_data(sample_num, train_data_dir='../datasets/train/'):
    '''
    :param sample_num: total data to sample
    :param train_data_dir: sample from train dataset
    :return:
    '''
    save_path = f'../datasets/sample_data-{sample_num}.parquet'
    if os.path.exists(save_path): return

    # you can sample training data from each subset
    # num = int(sample_num / 6)
    # data_list = ['HotpotQA', 'NaturalQuestionsShort', 'NewsQA', 'SearchQA', 'SQuAD', 'TriviaQA-web']

    # or sample training data from only one subset
    num = sample_num
    data_list = ['HotpotQA']
    full_data = []

    for name in data_list:
        path = os.path.join(train_data_dir, f'{name}.parquet')
        data = utils.load_parquet_data(path)
        sampled_data = random.sample(data, num)
        subset_data = []
        for d in sampled_data:
            question = d['question']
            context = d['context']
            gold_ans = list(d['answers'])
            tmp = {"subset": f"{name}-train", "question": question, "context": context, "gold_ans": gold_ans}

            subset_data.append(tmp)

        full_data += subset_data

    utils.save_as_parquet(path_to_save=save_path, data_to_save=full_data)




def get_classification_datas(sample_num, model_name):
    '''
    :param sample_num: total data to sample
    :param model_name: the name of the llm
    :return:
    '''
    def ner(data):
        nlp = spacy.load("en_core_web_sm")

        for entry in data:
            # entities in golden answer
            text = entry.get('gold_ans', '')
            text = ''.join(list(text))
            doc = nlp(text)
            entities = [ent.text for ent in doc.ents]

            # entities in question
            text1 = entry.get('question', '')
            doc1 = nlp(text1)
            entities1 = [ent.text for ent in doc1.ents]
            entry['question_entities'] = entities1

            # entities in context
            text2 = entry.get('context', '')
            doc2 = nlp(text2)
            entities2 = [ent.text for ent in doc2.ents]
            entities2 = set(entities2) - set(entities1) - set(entities)
            entities2 = list(entities2)
            entry['context_entities'] = entities2

        return data

    if os.path.exists(f'../datasets/{model_name}/feature_vector.parquet'): return

    try:
        spacy.load("en_core_web_sm")
    except OSError:
        # If the model is not found, handle the error
        print('Model "en_core_web_sm" not found. Downloading now...')
        from spacy.cli import download
        download("en_core_web_sm")
        spacy.load("en_core_web_sm")
        print('Model "en_core_web_sm" downloaded and loaded successfully!')

    model = Model(model_name=model_name)

    # step 1: filter data (Answers incorrectly without context but answers correctly with context.)
    if os.path.exists(f'../datasets/{model_name}/sample_data-{sample_num}-ner.parquet'):
        datas = utils.load_parquet_data(f'../datasets/{model_name}/sample_data-{sample_num}-ner.parquet')
    else:
        train_datas = utils.load_parquet_data(f'../datasets/sample_data-{sample_num}.parquet')

        # answer incorrectly without context
        memory_ans = model.get_memory_ans(train_datas)
        _, _, wrong_memory_ans = utils.evaluate_em(memory_ans, 'gold_ans', 'memory_ans')

        # answer correctly after connecting to the context
        context_ans = model.get_context_ans(wrong_memory_ans)
        _, correct, _ = utils.evaluate_em(context_ans, 'gold_ans', 'context_ans')
        datas = ner(correct)  # obtain entities from context
        utils.save_as_parquet(path_to_save=f'../datasets/{model_name}/sample_data-{sample_num}-ner.parquet', data_to_save=datas)
        datas = utils.load_parquet_data(f'../datasets/{model_name}/sample_data-{sample_num}-ner.parquet')


    # step 2: sample positive and negative datas from data (answer correctly after connecting to the context)
    feature_vectors = []
    pos_lable = np.array([[1]])

    for data in tqdm(datas):
        question = data['question']
        context = data['context']
        gold_ans = data['context_ans']
        question_entities = data['question_entities']
        context_entities = data['context_entities']

        if question_entities.size == 0 or not gold_ans:
            continue

        generated_tokens, output = model.attention_ratio_vector(
            question=question,
            context=context,
            gold_ans=gold_ans,
            question_entities=question_entities,
            context_entities=context_entities,
        )

        if (generated_tokens is None) or (not output): continue


        # 1. Positive sample
        if output['positive_ratio'] is not None:
            ratio_v = output['positive_ratio']
            row = np.concatenate([ratio_v, pos_lable], axis=1)
            feature_vectors.append(row)

        # 2. Negative sample (entities)
        neg_entity_rank = output['negative_entity_rank']
        neg_entity_ratio = output['negative_entity_ratio']
        if len(neg_entity_rank) > 0:
            entity_mask = neg_entity_rank > 10  # tokens with (rank > 10) are more likely to be non-utilized tokens
            neg_entity_ratio = neg_entity_ratio[entity_mask]
            labels = np.zeros((neg_entity_ratio.shape[0], 1))
            row = np.concatenate([neg_entity_ratio, labels], axis=1)
            feature_vectors.append(row)


        # 3. Negative sample (non-entities)
        neg_non_entity_rank = output['negative_non_entity_rank']
        neg_non_entity_ratio = output['negative_non_entity_ratio']
        if len(neg_non_entity_rank) > 0:
            non_entity_mask = neg_non_entity_rank > 10
            neg_non_entity_ratio = neg_non_entity_ratio[non_entity_mask]
            labels = np.zeros((neg_non_entity_ratio.shape[0], 1))
            row = np.concatenate([neg_non_entity_ratio, labels], axis=1)
            feature_vectors.append(row)


    df = pd.DataFrame(np.concatenate(feature_vectors, axis=0))  # [num_samples, num_features + 1]
    df.to_parquet(f'../datasets/{model_name}/feature_vector.parquet', index=False)




def train_detector(model_name, topk=10, train_data_size=100):
    '''
    :param model_name: the name of the llm
    :param topk: how many top features used for training
    :param train_data_size: training data size (half positive samples and half negative samples)
    :return:
    '''

    def balance_data(X, y, random_state, sample_size):
        ''' Sample positive and negative examples in a 1:1 ratio '''
        X_minority = X[y == 1]
        y_minority = y[y == 1]
        X_majority = X[y == 0]
        y_majority = y[y == 0]

        X_majority_downsampled, y_majority_downsampled = resample(
            X_majority, y_majority,
            replace=False,
            n_samples=sample_size,
            random_state=random_state
        )

        X_balanced = np.vstack((X_majority_downsampled, X_minority))
        y_balanced = np.hstack((y_majority_downsampled, y_minority))

        nan_rows = np.isnan(X_balanced).any(axis=1)
        return X_balanced[~nan_rows], y_balanced[~nan_rows]


    def train_and_select_best(X, y, param_grid, n_splits, topk_features=None):
        ''' Use k-fold cross-validation and save the best model. '''
        all_models = []
        fold_accuracies = []

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if topk_features is not None:
                X_train = X_train[:, topk_features]
                X_test = X_test[:, topk_features]

            # Use GridSearchCV for hyperparameter search.
            grid_search = GridSearchCV(LogisticRegression(max_iter=1000, penalty='l2'), param_grid, cv=3, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            # Obtain the best model for the current fold.
            best_model_in_fold = grid_search.best_estimator_
            all_models.append(best_model_in_fold)

            # Validate the performance of the current model on the test set.
            y_pred = best_model_in_fold.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            fold_accuracies.append(accuracy)
            print(f"Fold {fold} ACC: {accuracy:.4f}")

        # Calculate the overall performance of each fold's model (based on the validation sets of all folds).
        model_average_accuracies = []
        for fold_idx, model in enumerate(all_models):
            validation_accuracies = []
            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if topk_features is not None:
                    X_test = X_test[:, topk_features]

                y_pred = model.predict(X_test)
                validation_accuracies.append(accuracy_score(y_test, y_pred))

            model_average_accuracies.append(np.mean(validation_accuracies))

        # Select the model with the best overall average performance.
        best_model_index = np.argmax(model_average_accuracies)
        best_model = all_models[best_model_index]

        print(f"Best Model Selected from Fold {best_model_index + 1} with Average ACC: {model_average_accuracies[best_model_index]:.4f}")
        return best_model


    data = pd.read_parquet(f'../datasets/{model_name}/feature_vector.parquet')
    X = data.iloc[:, :-1].to_numpy()  # features
    y = data.iloc[:, -1].to_numpy()  # labels

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}


    print("\n===== Training with Full Features =====")
    X_balanced, y_balanced = balance_data(X, y, random_state=42, sample_size=train_data_size//2)
    best_full_model = train_and_select_best(X_balanced, y_balanced, param_grid, n_splits=5)

    coefficients = best_full_model.coef_[0]
    topk_indices = np.argsort(np.abs(coefficients))[-topk:]


    print("\n===== Training with Top-K Features =====")
    X_balanced_topk, y_balanced_topk = balance_data(X, y, random_state=43, sample_size=train_data_size//2)
    best_topk_model = train_and_select_best(X_balanced_topk, y_balanced_topk, param_grid, n_splits=5, topk_features=topk_indices)

    model_name = model_name.lower()

    if not os.path.exists('../detector/'):
        os.makedirs('../detector/', exist_ok=True)

    topk_model_path = f'../detector/{model_name}-top{topk}-size{train_data_size}-model.pkl'
    joblib.dump(best_topk_model, topk_model_path)
    print(f"Saved Top-{topk} model to {topk_model_path}")

    final_coefficients = best_topk_model.coef_[0].tolist()
    data_to_save = {
        'topk_indices': list(topk_indices),
        'weights': [i / sum(final_coefficients) for i in final_coefficients]
    }
    para_filename = f'../detector/{model_name}-top{topk}-size{train_data_size}-para.pkl'
    with open(para_filename, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"Top-{topk} features and weights saved in {para_filename}")



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--sample', type=int, default=6000)
    parser.add_argument('--hf_token', type=str, default='', help='set your huggingface token here')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf', help="model's huggingface_id")
    args = parser.parse_args()

    # step 1: sample data from train set
    print('step 1: sample data from train set')
    sample_data(args.sample)


    # step 2: download model
    print('\n\nstep 2: download model')
    model_name = os.path.basename(args.model).lower()
    utils.download_model(hf_token=args.hf_token, model_name=args.model, save_path=f'../models/{model_name}')


    # step3: get classification data (data used for training context utilization detector)
    print('\n\nstep3: get classification data')
    get_classification_datas(sample_num=args.sample, model_name=model_name)


    # step4: train context utilization detector
    print('\n\nstep4: train context utilization detector')
    train_detector(model_name, topk=10, train_data_size=2000)