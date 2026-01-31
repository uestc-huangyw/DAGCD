import os
import pandas as pd
from datasets import load_dataset


def download_qa_data(save_dir: str):
    '''
    download question answering datasets (from MrQA)
    :param save_path: path to save datasets
    :return:
    '''
    def remove_tags_from_ctx(example):
        """
        remove [PAR], [DOC], [TLE], [SEP]
        """
        context = example['context']
        context = context.replace("[PAR]", "\n")
        context = context.replace("[DOC]", "")
        context = context.replace("[TLE]", "")
        context = context.replace("[SEP]", "")
        context = context.strip()
        example['context'] = context
        return example

    os.makedirs(save_dir, exist_ok=True)
    data_split: tuple = ('train', 'validation')

    print("Downloading Question Answering dataset...")
    for split in data_split:
        full_data = load_dataset("mrqa", split=split)
        split_subset_names = list(set(full_data['subset']))
        for subset in split_subset_names:
            subset_data = full_data.filter(lambda ex: ex['subset'] == subset)
            subset_data = subset_data.map(remove_tags_from_ctx)
            subset_split_path = os.path.join(save_dir, split, f"{subset}.parquet")
            subset_data.to_parquet(subset_split_path)
    
    dataset = load_dataset("pminervini/NQ-Swap")['dev']
    data = []
    for d in dataset:
        question = d['question']+'?'
        context = d['sub_context']
        answers = d['sub_answer']
        t = {'subset':'NQ-swap','question':question, 'context':context, 'answers':answers}
        data.append(t)

    df = pd.DataFrame(data)
    df.to_parquet(os.path.join(save_dir, "validation/NQ-swap.parquet"), index=False)


def download_summary_data(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    # abisee/cnn_dailymail test set
    print("Downloading abisee/cnn_dailymail test dataset...")
    cnn_dm_dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
    cnn_dm_dataset = cnn_dm_dataset.map(lambda x: {"gold_ans": x["highlights"], **{k: v for k, v in x.items() if k != "highlights"}})
    cnn_dm_dataset = cnn_dm_dataset.map(lambda x: {"document": x["article"], **{k: v for k, v in x.items() if k != "article"}})
    cnn_dm_save_path = os.path.join(save_dir, "cnn_dm.jsonl")
    cnn_dm_dataset.to_parquet(cnn_dm_save_path)
    print(f"cnn_dailymail save in: {cnn_dm_save_path}")



if __name__ == '__main__':
    download_qa_data(save_dir='../datasets/')
    download_summary_data(save_dir='../datasets/')

