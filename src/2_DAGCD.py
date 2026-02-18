import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import joblib
import logging
from datetime import datetime
from argparse import ArgumentParser
import utils
import os
import math

def set_logger(args):
    current_time = datetime.now()
    log_folder = f"../logs/{args.model}/{args.data}"
    if not os.path.exists(log_folder):
        utils.make_dir(log_folder)
    log_filename = os.path.join(log_folder, f"{args.model}-{args.data}-{current_time.strftime('%Y_%m_%d-%H_%M_%S')}.txt")
    utils.make_dir(log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
        # handlers = [logging.FileHandler(log_filename)]
    )

    logger = logging.getLogger()
    # logger.setLevel(logging.CRITICAL)
    logger.setLevel(logging.INFO)

    return logger



def set_seed(random_seed=2025):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)



class AttentionExtractor:
    '''Extract the attention maps of the specific attention heads.'''
    def __init__(self, model, topk_head_index):
        self.model = model
        self.device = self.model.device
        self.topk_head_index = topk_head_index  # [[layer_idx, head_idx], ...]
        self.layer_idx, self.head_idx = None, None
        self.attentions = {(layer, head): None for layer,head in topk_head_index}

    def register_hooks(self):
        for layer_idx, head_idx in self.topk_head_index:
            layer = self.model.model.layers[layer_idx]
            if hasattr(layer, 'self_attn'):
                def hook_wrapper(layer_idx=layer_idx, head_idx=head_idx):
                    def attention_hook(module, input, output):
                        attention_scores = output[1]
                        attention_map = attention_scores[0, head_idx, :, :]
                        self.attentions[(layer_idx, head_idx)] = attention_map.to(self.device)

                    return attention_hook
                layer.self_attn.register_forward_hook(hook_wrapper())
            else:
                print(f"Warning: 'self_attn' attribute not found in model {model}.\n"
                      f"(The error indicates that in model {model}, the name of self-attention is not 'self_attn'. Please check and modify it.)")
                exit()

    def get_attention(self):
        topk_attention = torch.stack([self.attentions[(layer_idx, head_idx)] for layer_idx, head_idx in self.topk_head_index])
        return topk_attention  # [topk, seq, seq]


class Model:
    def __init__(self, args):
        self.top_rank = args.top_rank
        self.model, self.tokenizer = self.load_model_and_tokenizer(args.model)
        self.device = self.model.device
        self.set_up(args.model, args.topk, args.size)
        self.context_ids = None


    def load_model_and_tokenizer(self, model_name):
        print(f"Loaded models: {model_name}")
        model_path = f'../models/{model_name}'

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto'
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        config = AutoConfig.from_pretrained(model_path)
        self.vocab_size = config.vocab_size
        self.max_length = min(2048, config.max_position_embeddings)
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads

        return model, tokenizer


    def set_up(self, model_name, topk, size):
        ''' some initial settings '''
        model_name_lower = model_name.lower()

        # end_token_ids:
        # Represents some non-semantic characters that will be output before or after the answer tokens, such as '\n' or <|eot_id|>.
        model_config = {
            'llama-2': {
                'end_token_ids': [13, 29871],  # '\n', ' '
            },
            'mistral': {
                'end_token_ids': [28705],  # ' '
            },
            'llama-3': {
                'end_token_ids': [13, 198, 220, 128009],  # '.', '\n', ' ', '<|eot_id|>'
            }
        # If you want to use DAGCD on models from other families,
        # you need to first determine the corresponding end_token_ids, which are generally the IDs corresponding to '\n' or <|eot_id|>.
        }

        for key, config in model_config.items():
            if key in model_name_lower:
                self.end_token_ids = config['end_token_ids']
                break
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        path_prefix = f'../detector/{model_name}'
        self.detector = joblib.load(f'{path_prefix}-top{topk}-size{size}-model.pkl')

        with open(f'{path_prefix}-top{topk}-size{size}-para.pkl', 'rb') as f:
            data = pickle.load(f)
            self.topk_head_weights = torch.tensor(data['weights'], device=self.device).view(-1, 1)
            self.topk_head_index = []
            for index in data['topk_indices']:
                layer_idx = index // self.num_heads
                head_idx = index % self.num_heads
                self.topk_head_index.append([layer_idx, head_idx])
            self.topk_heads = data['topk_indices']

        # scaling factor
        self.alpha = 4 if 'chat' in model_name_lower or 'instruct' in model_name_lower else 2

        # Initialize the AttentionExtractor and register hooks
        self.attention_extractor = AttentionExtractor(self.model, self.topk_head_index)
        self.attention_extractor.register_hooks()


    def print_distribution_topk(self, probs, topn):
        ''' Print the information corresponding to the top <topn> tokens in the distribution <probs>. '''
        top_values, top_indices = torch.topk(probs, topn)
        top_values = top_values.squeeze(0)
        top_indices = top_indices.squeeze(0)
        # ids to tokens
        for topk, (val, ids) in enumerate(zip(top_values, top_indices)):
            ids = ids.item()
            token = self.tokenizer.decode([ids])
            logger.info(f'rank{topk + 1},  value:{val}, token:{token}, ids:{ids}')


    def get_context_token_rank(self, original_probs, context_token_ids):
        '''
        return: The ranking of all context tokens in the original distribution.
        '''
        sorted_indices = torch.argsort(original_probs, descending=True)
        ranks = torch.nonzero(sorted_indices == context_token_ids.unsqueeze(1), as_tuple=True)[1] + 1
        return ranks.to(self.device)


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
        '''
        In the case of multiple attention values for the same token_ids,
        we use the average to represent the attention of that token.
        '''
        unique_context_ids, inverse_indices = torch.unique(self.context_ids, return_inverse=True)
        unique_context_len = len(unique_context_ids)
        unique_attention = torch.zeros((raw_attention.shape[0], unique_context_len), device=self.device)  # [num_dist, unique_len]

        for idx in range(unique_context_len):
            mask = (inverse_indices == idx)
            avg = raw_attention[:, mask].mean(dim=1)
            unique_attention[:, idx] = avg

        return unique_context_ids, unique_attention


    def get_utilization_distribution(self, topk_attentions, context_start_idx, context_end_idx, token_level_probability_distribution):
        """
        Locate the start and end token indices of the context string within token_ids
        based on offset_mapping and the context string in the prompt.

        Parameters:
            topk_attentions: topk_heads' attention map
            context_start_idx: The starting position of context in input_ids
            context_end_idx: The ending position of context in input_ids
            token_level_probability_distribution: Original distribution

        Returns:
            utilization_distribution
        """

        # step1: Context Utilization Detection
        # 1.1 Feature Data Collection
        sequence_attention = topk_attentions[:, -1, :]  # [topk_head, seq_len]
        context_attention = sequence_attention[:, context_start_idx: context_end_idx + 1]  # [topk_head, context_len]
        unique_context_ids, unique_context_attention = self.get_unique_attn(context_attention)

        # 1.2 Normalize attention across the unique context tokens
        unique_context_attention_sum = unique_context_attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        unique_context_attention_ratio = unique_context_attention / unique_context_attention_sum  # [len(unique_context_ids), topk_head]

        # 1.3 detect utilized-tokens
        data = unique_context_attention_ratio.permute(1, 0).cpu().numpy()
        data = np.nan_to_num(data, nan=0.0)
        mask = torch.tensor(self.detector.predict(data), device=self.device, dtype=torch.float32)

        # 1.4 Top-Rank Constraint
        ranks = self.get_context_token_rank(token_level_probability_distribution, unique_context_ids)
        mask[ranks > self.top_rank] = 0

        # step2: calculate utilization distribution
        utilization_distribution = torch.zeros(self.vocab_size, device=self.device)
        if mask.sum() > 0:
            informative_head = (unique_context_attention_ratio * self.topk_head_weights).sum(dim=0) * mask
            informative_head_sum = informative_head.sum().clamp_min(1e-8)
            informative_head = informative_head / informative_head_sum
            utilization_distribution[unique_context_ids] = informative_head.float()

        logger.info('***** Utilization Distribution *****')
        self.print_distribution_topk(utilization_distribution, topn=10)

        return utilization_distribution.unsqueeze(0)


    def normalized_entropy(self, dist, eps=1e-12):
        ''' return normalized entropy of distribution <dist> '''
        dist = dist.float().clamp_min(eps)
        entropy = -(dist * dist.log()).sum(dim=-1)
        return (entropy / math.log(self.vocab_size)).item()


    def generate(self,question: str, context: str, gen_max_length: int = 10, use_dagcd=True):
        '''
        Parameters:
            question: question
            context: context
            gen_max_length: Max new token length
            use_dagcd: Whether to use DAGCD
        Returns:
            generated token ids
        '''
        prompt = f'Given the following information:{context}\nAnswer the following question based on the given information with one or few words: {question}\nAnswer:'
        tokenized_inputs = self.tokenizer(prompt,
                                          return_tensors="pt",
                                          return_offsets_mapping=True,
                                          truncation=True,
                                          max_length = self.max_length - gen_max_length
                                          ).to(self.device)
        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']
        offset_mapping = tokenized_inputs['offset_mapping'][0]
        if torch.is_tensor(offset_mapping):
            offset_mapping = offset_mapping.cpu().numpy()
        else:
            offset_mapping = np.array(offset_mapping)

        context_start, context_end = self.locate_context_position(offset_mapping, prompt, context)
        self.context_ids = input_ids.squeeze(0)[context_start: context_end + 1]

        # generate ans
        with torch.no_grad():
            generated_tokens = []
            for cur_len in range(gen_max_length):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, return_dict=True, use_cache=True)
                next_token_logits = outputs.logits[:, -1, :]
                token_level_probability_distribution = F.softmax(next_token_logits.float(), dim=-1)
                next_token = torch.argmax(token_level_probability_distribution, dim=-1)

                ##################################  DAGCD  #######################################
                if use_dagcd:
                    logger.info('\n\n***** Original Distribution *****')
                    self.print_distribution_topk(token_level_probability_distribution, topn=10)
                    topk_attentions = self.attention_extractor.get_attention()  # topk_heads' attention map

                    if gen_max_length == 1 and next_token in self.end_token_ids:
                        utilization_distribution = self.get_utilization_distribution(
                            topk_attentions,
                            context_start,
                            context_end,
                            token_level_probability_distribution.squeeze()
                        )
                        next_token = torch.argmax(utilization_distribution, dim=-1)

                        logger.info('***** Final Distribution *****')
                        self.print_distribution_topk(utilization_distribution, topn=10)
                        logger.info('\n')

                    elif (next_token not in self.end_token_ids):
                        utilization_distribution = self.get_utilization_distribution(
                            topk_attentions,
                            context_start,
                            context_end,
                            token_level_probability_distribution.squeeze()
                        )
                        weight = self.alpha * self.normalized_entropy(token_level_probability_distribution)
                        final_distribution = token_level_probability_distribution + weight * utilization_distribution
                        next_token = torch.argmax(final_distribution, dim=-1)

                        logger.info('***** Final Distribution *****')
                        self.print_distribution_topk(final_distribution, topn=10)
                        logger.info('\n')
                ##################################  DAGCD  n#######################################

                if next_token == self.tokenizer.eos_token_id:
                    break

                tokens_to_add = next_token
                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(0)], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.tensor([[1]], device=self.device)], dim=-1)
                generated_tokens.append(tokens_to_add)

        if len(generated_tokens) == 0:
            return None
        else:
            return torch.cat(generated_tokens)


if __name__ == '__main__':
    set_seed(2025)
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='llama-2-7b-hf')
    parser.add_argument('--data', type=str, default='HotpotQA')
    parser.add_argument('--topk', type=int, default=10, help='top-k features')
    parser.add_argument('--top_rank', type=int, default=10, help='top rank constraint')
    parser.add_argument('--size', type=int, default=2000, help='LR-train-size')
    args = parser.parse_args()

    logger = set_logger(args)

    model = Model(args)
    data_path = f'../datasets/validation/{args.data}.parquet'
    datas = utils.load_parquet_data(data_path)

    results = []

    greedy = False  # Whether to generate greedy answers

    for i, data in enumerate(tqdm(datas)):
        question = data['question']
        context = data['context']
        gold_ans = data['answers']

        tgt_len = max(len(model.tokenizer.encode(ans, add_special_tokens=False)) for ans in gold_ans)

        # Greedy answer
        if greedy:
            greedy_tokens = model.generate(
                question=question,
                context=context,
                gen_max_length=tgt_len,
                use_dagcd=False
            )
            if greedy_tokens is None: continue
            greedy_ans = model.tokenizer.decode(greedy_tokens, skip_special_tokens=True)

        # DAGCD answers
        dagcd_tokens = model.generate(
            question=question,
            context=context,
            gen_max_length=tgt_len,
            use_dagcd=True
        )
        if dagcd_tokens is None: continue
        pred_ans = model.tokenizer.decode(dagcd_tokens, skip_special_tokens=True)

        if greedy:
            res = {'context': context, 'question': question, 'gold_ans': gold_ans, 'greedy_ans': greedy_ans, 'pred_ans': pred_ans}
            logger.info(f"Q:{question}\n"
                        f"Golden Answer: {gold_ans}\n"
                        f"Greedy Answer: {greedy_ans}\n"
                        f"DAGCD Answer: {pred_ans}")
        else:
            res = {'context': context, 'question': question, 'gold_ans': gold_ans, 'pred_ans': pred_ans}
            logger.info(f"Q:{question}\n"
                        f"Golden Answer: {gold_ans}\n"
                        f"DAGCD Answer: {pred_ans}")

        logger.info('*-'*60)
        logger.info('\n\n')

        results.append(res)

    save_path = f'../results/{args.model}/{args.data}-{args.size}.jsonl'
    utils.save_as_parquet(save_path, results)


    ######################################### evaluate #########################################
    # filter token length > 2048 (because truncation)
    new_data = []
    for d in results:
        prompt = f'Given the following information:{d["context"]}\nAnswer the following question based on the given information with one or few words: {d["question"]}\nAnswer:'
        if len(model.tokenizer.encode(prompt)) <= 2048: new_data.append(d)
    results = new_data


    ############# greedy results #############
    if greedy:
        greedy_f1, _, _ = utils.evaluate_f1(results, gold_ans='gold_ans', pred_ans='greedy_ans')
        greedy_em, _, _ = utils.evaluate_em(results, gold_ans='gold_ans', pred_ans='greedy_ans')

        logger.info('*' * 30)
        logger.info('----- Greedy performance -----')
        logger.info(f'EM: {greedy_em * 100}%')
        logger.info(f'F1: {greedy_f1 * 100}%\n\n')


    ############# dagcd results #############
    dagcd_f1, _, _ = utils.evaluate_f1(results, gold_ans='gold_ans', pred_ans='pred_ans')
    dagcd_em, _, _ = utils.evaluate_em(results, gold_ans='gold_ans', pred_ans='pred_ans')
    logger.info('----- DAGCD performance -----')
    logger.info(f'EM: {dagcd_em * 100}%')
    logger.info(f'F1: {dagcd_f1 * 100}%')
    logger.info('*' * 30)

    print('----- DAGCD performance -----')
    print(f'EM: {dagcd_em * 100}%')
    print(f'F1: {dagcd_f1 * 100}%')

    with open(f'../results/{args.model}/{args.data}-top{args.topk}-size{args.size}.txt', 'w') as file:
        if greedy:
            file.write('----- Greedy performance -----\n')
            file.write(f'EM: {greedy_em * 100}%\n')
            file.write(f'F1: {greedy_f1 * 100}%\n\n')

        file.write('----- DAGCD performance -----\n')
        file.write(f'EM: {dagcd_em * 100}%\n')
        file.write(f'F1: {dagcd_f1 * 100}%')
