import os
import numpy as np
from data_loader import Data
from model import Model
from collections import defaultdict as ddict
import json
import time

def save_to_json(file_name, file_dict):
    file = open(file_name, 'w', encoding='utf-8')
    file.write(json.dumps(file_dict))
    file.close()
    return

def get_metrics(pred_list, gt_list):
    metrics = ['HITS', 'MRR']
    result = {metric: 0.0 for metric in metrics}
    for gt in gt_list:
        if gt in pred_list:
            index = pred_list.index(gt)
            result['MRR'] = max(result['MRR'], 1/(index+1))
            result['HITS'] = 1 
    return result

def get_candidate_voter_list(model, data, submit_path, top_k=5):
    print('Start inference...')
    submission = []
    test_iter = iter(data.data_iter['test'])
    for step, (triples, trp_ids) in enumerate(test_iter):
        sub, rel, obj = (
            triples[:, 0],
            triples[:, 1],
            triples[:, 2],
        )
        triples = triples.cpu().tolist()
        ids = trp_ids.cpu().tolist()
        for (triple, triple_id) in zip(triples, ids):
            s, r, _ = triple
            candidate_voter_list = model.get_pred(s, r, 0)
            candidate_voter_list = [data.id2ent[k] for k in candidate_voter_list]
            submission.append({
                'triple_id': '{:04d}'.format(triple_id[0]),
                'candidate_voter_list': candidate_voter_list
            })
    save_to_json(submit_path, submission)

def evaluate(model, data, top_k=5):
    print('Start evaluate...')
    results = ddict(list)
    all_triples, all_preds = [], []
    test_iter = iter(data.data_iter['valid'])
    for step, (triples, trp_ids) in enumerate(test_iter):
        sub, rel, obj = (
            triples[:, 0],
            triples[:, 1],
            triples[:, 2],
        )
        all_triples += triples.cpu().tolist()

    for triple in all_triples:
        s, r, _ = triple
        gt_set = data.sr2o['valid'][(s, r)]
        gt_list = np.array(list(gt_set), dtype=np.int64)

        pred_list = model.get_pred(s, r, 1)

        result = get_metrics(pred_list, gt_list)
        for k,v in result.items():
            results[k].append(v)

    results = {k: np.mean(v)  for k,v in results.items()}
    return results


if __name__ == "__main__":
    # Hypermeter Setting
    pw = 30     # 正向关系权重阈值
    nw = 15     # 正向关系权重阈值
    tc = 100    # 防止时间过大，导致权重溢出
    
    t1 = time.time()
    print("Create dataloader for evaluate...")
    data1 = Data('./data', 10, 2000, 4000)  # 验证用
    model1 = Model(data1, pw, nw, tc) 

    t2 = time.time()
    print("Create dataloader for inference...")
    data = Data('./data', 10, 2000, 0)  # 测试用
    model = Model(data, pw, nw, tc)

    t3 = time.time()
    # 验证：在复赛数据集上进行验证
    val_results = evaluate(model1, data1, top_k=5)
    print("验证结果为: ", val_results)

    t4 = time.time()
    # 测试：生成预测的结果
    submit_path = "{}/final_submission.json".format("./output")
    get_candidate_voter_list(model, data, submit_path, top_k=5)
    print("Submission file has been saved to: {}.".format(submit_path))

    t5 = time.time()

    print("生成验证dataloader所用时间：", t2 - t1)
    print("生成测试dataloader所用时间：", t3 - t2)
    print("计算验证结果所用时间：", t4 - t3)
    print("生成测试json所用时间：", t5 - t4)