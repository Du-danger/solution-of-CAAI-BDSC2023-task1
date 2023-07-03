import datetime
import numpy as np
import heapq

class Model:
    def __init__(self, data, pw, nw, tc, date_str_init = "20211101"):
        self.date_init = datetime.datetime.strptime(date_str_init, '%Y%m%d').date()
        self.data = data
        self.pw = pw
        self.nw = nw
        self.tc = tc
    
    def get_time(self, date_str):
        date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
        dura = (date - self.date_init).days
        return dura
    
    def get_pred(self, s, r, valid = 0, top_k = 5):
        '''
        input:
            s: 邀请者
            r: 关系场景
            valid: 是否为验证 1:验证, 0:测试
            top_k: 默认为5
        output:
            返回大小为 top_k 的 list, 表示预测的候选列表
        '''
        
        d = self.data.data      # 正向邀请关系
        d1 = self.data.data1    # 反向被邀请关系
        
        pre_dict = dict()   # 字典：保存可能是被邀请的用户的得分

        # -------------------一阶邻居建模----------------------
        # 正向关系建模
        max_v = -1
        max_ds = 0
        for a in d[s]:
            if a not in pre_dict:
                pre_dict[a] = 0.0
            l = 0
            for b in d[s][a]:
                ds = self.get_time(b[1])
                l += ds/(self.tc)
                if ds > max_ds:
                    max_ds = ds
                    max_v = a
            if l > self.pw:
                pre_dict[a] += self.pw
            else:
                pre_dict[a] += l
        if max_v != -1:     # 根据最近时间交互进行额外加权
            pre_dict[max_v] += 2

        # 反向关系建模
        max_v1 = -1
        max_ds1 = 0
        for a in d1[s]:
            if a not in pre_dict:
                pre_dict[a] = 0.0
            l = 0
            for b in d1[s][a]:
                ds = self.get_time(b[1])
                l += ds/(2 * self.tc)
                if ds > max_ds1:
                    max_ds1 = ds
                    max_v1 = a
            if l > self.nw:
                pre_dict[a] += self.nw
            else:
                pre_dict[a] += l
        
        if max_v1 != -1:    # 根据最近时间交互进行额外加权
            pre_dict[max_v1] += 0.5
        
        # ---------------------二阶邻居建模----------------------
        pre_dict1 = pre_dict.copy()
        for i in pre_dict1:
            k = 0.00001
            # 正向关系
            for a in d[i]:
                if a not in pre_dict1 and a != s:
                    if a not in pre_dict:
                        pre_dict[a] = 0.0
                    l = len(d[i][a])
                    if l > self.pw:
                        pre_dict[a] += self.pw * k
                    else:
                        pre_dict[a] += l * k
            # 反向关系
            for a in d1[i]:
                if a not in pre_dict1 and a != s:
                    if a not in pre_dict:
                        pre_dict[a] = 0.0
                    l = len(d1[i][a])
                    if l > self.nw:
                        pre_dict[a] += self.nw  * k
                    else:
                        pre_dict[a] += l * k
        
        # 去除在该场景下已邀请过的用户，防止重复邀请，不参与排名
        train_set = self.data.sr2o['train'][(s, r)]
        if valid == 0:
            train_set.update(self.data.sr2o['valid'][(s, r)])
        train_list = list(train_set)
        for i in train_list:
            if i in pre_dict:
                pre_dict.pop(i, None)
        
        # 排序，获得最终结果
        candidate_voter_dict = heapq.nlargest(top_k, pre_dict.items(), key=lambda kv:kv[1])
        candidate_voter_list = [k for k,v in candidate_voter_dict]

        return candidate_voter_list


        