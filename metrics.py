import numpy as np
def DCG(row, k=0):
    '''计算一行DCG'''
    if k > 0:
        return np.sum(np.divide(np.exp2(row[:k]) - 1, np.log2(np.arange(3, len(row[:k]) + 3))))
    else:
        return np.sum(np.divide(np.exp2(row) - 1, np.log2(np.arange(3, len(row) + 3))))
def nDCG(m_pred, m_rate, num):
    '''计算nDCG
       输入:预测评分矩阵，数据集评分矩阵,序列长度
       返回:nDCG
    '''
    ndcgs = []
    # 每一行计算ndcg
    for idx, row in enumerate(m_rate):
        # 获取该行非零位置
        col_n0 = np.array(np.nonzero(row)).flatten()
        if(len(col_n0) == 0):
            continue
        dic = dict(zip(m_pred[idx][col_n0], col_n0))  # 预测值值为健，列号为值
        sort_key = sorted(list(dic.keys()), reverse=True)
        sort_col = [dic[i] for i in sort_key]

        # DCG
        dcg = DCG(row[sort_col], k=num)

        # iDCG
        sort_rate = sorted(row[col_n0], reverse=True)
        idcg = DCG(sort_rate, k=num)

        # ndcg
        ndcgs.append(dcg / idcg)

    return ndcgs
