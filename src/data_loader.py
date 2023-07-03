from collections import defaultdict as ddict
import numpy as np
import pandas as pd
import torch
from ordered_set import OrderedSet
from torch.utils.data import DataLoader, Dataset

class TestDataset(Dataset):
    def __init__(self, sr2o, triple2idx=None):
        self.sr2o = sr2o
        self.triples, self.ids = [], []
        for (s, r), o_list in self.sr2o.items():
            self.triples.append([s, r, -1])
            if triple2idx is None:
                self.ids.append(0)
            else:
                self.ids.append([triple2idx[(s, r)]])

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return torch.LongTensor(self.triples[idx]), torch.LongTensor(self.ids[idx])

    @staticmethod
    def collate_fn(data):
        triples = []
        ids = []
        for triple, idx in data:
            triples.append(triple)
            ids.append(idx)
        triples = torch.stack(triples, dim=0)
        trp_ids = torch.stack(ids, dim=0)
        return triples, trp_ids

class Data(object):
    def __init__(self, data_dir, num_workers, batch_size, K):
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size

        ent_set, rel_set = OrderedSet(), OrderedSet()

        # 场景是关系
        event_info = pd.read_json("{}/event_info.json".format(self.data_dir)) 
        rel_set.update(event_info['event_id'].tolist())
        print('Number of events: {}'.format(len(rel_set)))

        # 用户是实体
        user_info = pd.read_json("{}/user_info.json".format(self.data_dir))
        ent_set.update(user_info['user_id'].tolist())
        print('Number of users: {}'.format(len(ent_set)))

        # 标识（字符串） --> id（int）；
        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}

        # id（int） --> 标识（字符串）
        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.num_ent = len(self.ent2id)
        self.num_rel = len(self.rel2id)

        self.sr2o = dict()
        for split in ['train', 'valid', 'test']:
            self.sr2o[split] = ddict(set)
        

        self.data = dict()  # 正向邀请关系
        self.data1 = dict() # 反向被邀请关系

        for i in range(0, self.num_ent):
            self.data[i] = ddict(list)
            self.data1[i] = ddict(list)

        # 原始训练集（40个场景）
        df = pd.read_json("{}/source_event_preliminary_train_info.json".format(self.data_dir))
        records = df.to_dict('records')
        print('原始训练集的条数：', len(records))
        for line in records:
            sub, rel, obj = line['inviter_id'], line['event_id'], line['voter_id']
            ds = str(line['ds'])
            sub_id, rel_id, obj_id = (
                self.ent2id[sub],
                self.rel2id[rel],
                self.ent2id[obj],
            )
            self.data[sub_id][obj_id].append((rel_id, ds))
            self.data1[obj_id][sub_id].append((rel_id, ds))
            self.sr2o['train'][(sub_id, rel_id)].add(obj_id)
        
        # 初赛目标训练集（8个场景）
        if K == 0:
            df = pd.read_json("{}/target_event_preliminary_train_info.json".format(self.data_dir))
            records = df.to_dict('records')
            print('初赛目标训练集的条数：', len(records))
            for line in records:
                sub, rel, obj = line['inviter_id'], line['event_id'], line['voter_id']
                ds = str(line['ds'])
                sub_id, rel_id, obj_id = (
                    self.ent2id[sub],
                    self.rel2id[rel],
                    self.ent2id[obj],
                )
                self.data[sub_id][obj_id].append((rel_id, ds))
                self.data1[obj_id][sub_id].append((rel_id, ds))
                self.sr2o['train'][(sub_id, rel_id)].add(obj_id)
        
        # 复赛目标训练集（8个场景）{45: 1874, 29: 1908, 40: 3033, 3: 1779, 42: 1939, 48: 397, 6: 1413, 33: 722} (s,r) 存在重复
        few_shot_valid_cnt = ddict(int)
        df = pd.read_json("{}/target_event_final_train_info.json".format(self.data_dir))
        records = df.to_dict('records')
        aset = set()
        print('复赛目标训练集的条数：', len(records))
        for line in records:
            sub, rel, obj = line['inviter_id'], line['event_id'], line['voter_id']
            ds = str(line['ds'])
            sub_id, rel_id, obj_id = (
                self.ent2id[sub],
                self.rel2id[rel],
                self.ent2id[obj],
            )
            if few_shot_valid_cnt[rel_id] < K:
                few_shot_valid_cnt[rel_id] += 1
                c = (sub_id, rel_id)
                if c not in aset:   # 使(s,r) 不重复，与测试集一致
                    aset.add(c)
                    self.sr2o['valid'][c].add(obj_id)
            else:
                if K == 0:
                    self.data[sub_id][obj_id].append((rel_id, ds))
                    self.data1[obj_id][sub_id].append((rel_id, ds))
                    self.sr2o['train'][(sub_id, rel_id)].add(obj_id)

        # 复赛目标测试集 {42: 1250, 6: 1250, 3: 1250, 29: 1250, 45: 1250, 40: 1250, 33: 1250, 48: 1250} 、 (s,r)不重复
        self.triple2idx = dict()
        df = pd.read_json("{}/target_event_final_test_info.json".format(self.data_dir))
        records = df.to_dict('records')
        for line in records:
            triple_id = int(line['triple_id'])
            sub, rel = line['inviter_id'], line['event_id']
            sub_id, rel_id = self.ent2id[sub], self.rel2id[rel]
            self.sr2o['test'][(sub_id, rel_id)] = set()
            self.triple2idx[(sub_id, rel_id)] = triple_id
        
        def get_test_data_loader(split, batch_size, shuffle=False):
            triple2idx = None if split == 'valid' else self.triple2idx
            return DataLoader(
                TestDataset(
                    self.sr2o[split], triple2idx
                ),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.num_workers),
                collate_fn=TestDataset.collate_fn,
                pin_memory=False
            )

        # valid/test dataloaders
        self.data_iter = {
            "valid": get_test_data_loader("valid", 1024),
            "test": get_test_data_loader("test", 1024),
            }

if __name__ == "__main__":
    data = Data('./data', 10, 2000, 4000)
