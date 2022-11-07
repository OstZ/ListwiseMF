import numpy as np
from metrics import nDCG
class ListMF(object):
    def __init__(self,num_user,num_item,num_dim,**kwargs):
        self.num_user = num_user
        self.num_item = num_item
        self.num_dim = num_dim
        self.U = self._initialize(num_user,num_dim)
        self.V = self._initialize(num_item,num_dim)
        print("Shape of U features:{}".format(self.U.shape))
        print("Shape of V features:{}".format(self.V.shape))
    def _sigmoid(self,x):
        '''1/(1+exp(-x))'''
        return 1 / (1 + np.exp(-x))

    def _sigmoid_prime(self,x):
        '''the derivative of sigmoid'''
        temp = self._sigmoid(x)
        return temp * (1 - temp)

    def _initialize(self,row, col):
        '''随机初始化特征矩阵,row行数，col列数
           返回随机初始化的特征矩阵
        '''
        #uniform_distribution
        return np.random.rand(row,col)

    def _PL_Top1_Loss(self,mat_u,mat_v,m_rate,l):
        '''
               给定两特征矩阵mat_u,mat_v，及目标矩阵mat_r和正则化系数l
               返回loss
            '''
        # 计算预测矩阵
        m_pred = np.dot(mat_u, mat_v.transpose())

        # 定义loss
        loss = np.float64(0.0)

        # 计算top_1 probability cross entropy
        for i in (np.arange(len(m_rate))):
            den_pred = np.float64(0.0)
            den_rate = np.float64(0.0)

            col_n0 = np.array(np.nonzero(m_rate[i])).flatten()
            den_pred = np.exp(self._sigmoid(m_pred[i][col_n0])).sum()
            den_rate = np.exp(m_rate[i][col_n0]).sum()

            # 行内top_1 概率交叉熵
            for j in col_n0:
                # if m_pred[i,j] != 0:
                # if m_pred[i, j] and m_obj[i, j] != 0:
                loss -= np.exp(m_rate[i, int(j)]) / den_rate \
                        * np.log(np.exp(self._sigmoid(m_pred[i, int(j)])) / den_pred)
                # print(loss)

        # 计算F范数
        loss += l * np.square(np.linalg.norm(mat_u)) / 2
        loss += l * np.square(np.linalg.norm(mat_v)) / 2

        return loss

    def _compute_gradient_ui(self,row_id, mat_u, mat_v, m_rate, l):
        '''给定Ui,V,评分矩阵对用行,正则化系数,计算Ui的梯度
           返回梯度
        '''
        row_u = mat_u[row_id]
        obj_i = m_rate[row_id]
        partial = np.zeros(len(row_u))
        #nonzero cols
        col_n0 = np.array(np.nonzero(obj_i)).flatten()
        den_pred = np.float64(0.0)
        den_rate = np.float64(0.0)
        mask_rows = mat_v[col_n0]
        pred_select_js = np.dot(row_u, mask_rows.transpose())
        obj_select_js = obj_i[col_n0]
        den_pred = np.exp(self._sigmoid(pred_select_js)).sum()
        den_rate = np.exp(obj_select_js).sum()

        # 计算left
        for i in range(len(pred_select_js)):
            partial += (np.exp(self._sigmoid(pred_select_js[i])) / den_pred - np.exp(obj_i[i]) / den_rate) \
                       * self._sigmoid_prime(pred_select_js[i]) * mat_v[i]
        # 计算right
        partial += l*row_u

        # in_bracket = (pred_select_js/den_pred - obj_select_js/den_rate)
        # out_braket = np.matmul(np.diag(self._sigmoid_prime(pred_select_js.T)),mask_rows)
        # partial = np.dot(in_bracket, out_braket) + l * row_u



        return partial

    def _compute_gradient_vj(self,row_id, mat_u, mat_v, m_rate, l):
        '''
           给定V中某一行的索引，U，V,评分矩阵，正则化系数
           计算并返回梯度
        '''
        partial = np.zeros(len(mat_v[row_id]))
        # 找到row_v在m_obj中对应列，选择这一列不为零的行号
        row_n0 = np.array(np.nonzero(m_rate.transpose()[row_id])).flatten()
        for i in row_n0:
            den_pred = np.float64(0.0)
            den_rate = np.float64(0.0)

            # m_obj中第i行非零位置
            col_n0 = np.array(np.nonzero(m_rate[i])).flatten()
            den_pred = np.exp(self._sigmoid(np.dot(mat_u[i], mat_v[col_n0].transpose()))).sum()
            den_rate = np.exp(m_rate[i][col_n0]).sum()

            # 计算left
            partial += (np.exp(self._sigmoid(np.dot(mat_u[i], mat_v[row_id].transpose()))) / den_pred - np.exp(
                m_rate[i][row_id]) / den_rate) \
                       * self._sigmoid_prime(np.dot(mat_u[i], mat_v[row_id].transpose())) * mat_u[i]
        # 计算right
        partial += l * mat_v[row_id]

        return partial

    def train(self,train,l,lr):

        #update U,V
        tmp_u = self.U
        tmp_v = self.V
        for i in np.arange(self.num_user):
            tmp_u[i] -= lr * self._compute_gradient_ui(i, self.U, self.V, train, l)
        for j in np.arange(self.num_item):
            tmp_v[j] -= lr * self._compute_gradient_vj(j, self.U, self.V, train, l)
        self.U = tmp_u
        self.V = tmp_v

    def eval(self,mat,l):
        loss = self._PL_Top1_Loss(self.U, self.V, mat, l)  # train loss
        ndcgs = nDCG(np.dot(self.U, self.V.transpose()), mat, 10)
        ndcg = sum(ndcgs) / len(ndcgs)  # ndcg in training set
        return loss,ndcg


