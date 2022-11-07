import numpy as np
from models import ListMF
from utils import create_traintest_split
from metrics import nDCG
num_u,num_i,train,test = create_traintest_split('ml-100k',10,20)
train = np.array(train.todense())
test = np.array(test.todense())
model = ListMF(num_u,num_i,5)
for i in range(2000):
    model.train(train,0.01,0.01)
    loss,ndcg = model.eval(test,0.01)
    print("epoch:{}  loss_train:{:.6f}  ndcg:{:.6f}".format(i, loss, ndcg,))
