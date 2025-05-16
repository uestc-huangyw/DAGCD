import utils

if __name__ == '__main__':
    data = utils.load_parquet_data('<result path>')

    # evaluate Greedy
    print('------------------Greedy------------------')
    greedy_em, _, _ = utils.evaluate_em(data, 'gold_ans', 'context_ans')
    greedy_f1, _, _ = utils.evaluate_f1(data, 'gold_ans', 'context_ans')

    print(f"EM: {greedy_em * 100}")
    print(f"F1: {greedy_f1 * 100}\n")

    # evaluate DAGCD
    print('------------------DAGCD------------------')
    ours_em, _, _ = utils.evaluate_em(data, 'gold_ans', 'pred_ans')
    ours_f1, _, _ = utils.evaluate_f1(data, 'gold_ans', 'pred_ans')


    print(f"EM: {ours_em * 100}")
    print(f"F1: {ours_f1 * 100}\n")