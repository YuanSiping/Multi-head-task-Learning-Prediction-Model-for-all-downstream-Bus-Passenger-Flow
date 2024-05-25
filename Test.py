# %%
# normal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time,random
from tqdm import tqdm
import os

# 日志的配置
os.environ['GLOG_logtostderr'] = '0'
os.environ['GLOG_v'] = '2'
os.environ['GLOG_log_dir'] = '/log2'

# %%
# mindspore
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import Model
from mindspore.dataset import vision, transforms
import mindspore.ops.operations as P

config = ms.get_log_config()
print(config)
# %%
# 设置随机种子，确保每次运行结果一致
np.random.seed(1)
ms.set_seed(1)
np.random.seed(1)

# %%
# 读取数据
#读取十堰一个月的车程时间数据
data_train = []
data_test = []
for i in range(10):
    filename_train = 'passenger_data_line5_stop{}.csv'.format(i+1)
    filename_test = 'test_passenger_data_line5_stop{}.csv'.format(i+1)

    filepath_train = './data/test_line5_csv/' + filename_train
    filepath_test = './data/test_line5_csv/' + filename_test

    d_train = pd.read_csv(filepath_train, encoding='GB2312')
    d_test = pd.read_csv(filepath_test, encoding='GB2312')

    data_train.append(np.array(d_train))
    data_test.append(np.array(d_test))

X_train = []
Y_train = []

X_test = []
Y_test = []

# 定义需要提取的列索引
columns_to_select = [0,1,2,3,4,5,6,7,8,9,10]
for i in range(10):
    
    X_train.append(data_train[i][:,columns_to_select])
    Y_train.append(data_train[i][:, 11])

    X_test.append(data_test[i][:,columns_to_select])
    Y_test.append(data_test[i][:, 11])


X_train = np.concatenate(X_train, axis=1, dtype=np.float32)
Y_train = np.array(Y_train, dtype=np.float32).T

X_test = np.concatenate(X_test, axis=1, dtype=np.float32)
Y_test = np.array(Y_test, dtype=np.float32).T


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# %%
# 整理测试集，使得(x-1)*x+1行至x*10行是stop x
X_test2 = np.empty((100, 110), dtype=np.float32)
Y_test2 = np.empty((100, 10), dtype=np.float32)

for i in range(0, X_test.shape[1]-11+1, 11):    # 决定站点，i//11代表站点几
    for j in range(0, X_test.shape[0]-10+1, 10):
        Xrow = np.ndarray.flatten(X_test[j:j+10, i:i+11]).astype(np.float32)
        Yrow = np.ndarray.flatten(Y_test[j:j+10, i//11]).astype(np.float32)
        X_test2[(i//11)*10+(j//10)] = Xrow
        Y_test2[(i//11)*10+(j//10)] = Yrow


# print(Y_test[0:10])
# print(Y_test2[90])

# %%
# Iterator as input source
class IterableDataset():
    # 可迭代数据集
    # 可以通过迭代的方式逐步获取数据样本
    def __init__(self, data, label):
        '''init the class object to hold the data'''
        self.data = data
        self.label = label

        self.start = 0
        self.end = len(data)
    def __iter__(self):
        self.start = 0
        self.end = len(self.data)
        return self

    def __next__(self):
        if self.start >= self.end:
            raise StopIteration
        data = self.data[self.start]
        label = self.label[self.start]
        self.start += 1
        return data, label
    
    
def datapipe(dataset, batch_size):
    dataset = dataset.batch(batch_size)
    return dataset    
    
train_dataset = IterableDataset(X_train, Y_train)
train_dataset = ds.GeneratorDataset(train_dataset, column_names=["data", "label"])
train_dataset = datapipe(train_dataset, 10)

test_dataset = IterableDataset(X_test2, Y_test2)
test_dataset = ds.GeneratorDataset(test_dataset, column_names=["data", "label"])
test_dataset = datapipe(test_dataset, 10)

# %%
i = 0
for data, label in train_dataset.create_tuple_iterator():
    print(f"Shape of image [N, C, H, W]: {data.shape} {data.dtype}")
    print(f"Shape of label: {label.shape} {label.dtype}")
    # print(label)
    break
    

# %%
class ML(nn.Cell):
    def __init__(self):
        super(ML, self).__init__()
        embed_dim = 11
        num_heads = 11
        # for 1
        self.m11 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f11 = nn.LSTM(input_size=11, hidden_size=128, num_layers=1, batch_first=True)
        self.ac11 = nn.Tanh()
        self.d11 = nn.Dropout(p=0.1)
        self.f12 = nn.Dense(128, 128, activation='relu')  # 线性层
        self.d12 = nn.Dropout(p=0.1)
        # for 2
        self.m21 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f21 = nn.LSTM(input_size=11, hidden_size=128, num_layers=1, batch_first=True)
        self.ac21 = nn.Tanh()
        self.d21 = nn.Dropout(p=0.1)
        self.f22 = nn.Dense(128, 128, activation='relu')  # 线性层
        self.d22 = nn.Dropout(p=0.1)
        # for 3
        self.m31 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f31 = nn.LSTM(input_size=11, hidden_size=128, num_layers=1, batch_first=True)
        self.ac31 = nn.Tanh()
        self.d31 = nn.Dropout(p=0.1)
        self.f32 = nn.Dense(128, 128, activation='relu')  # 线性层
        self.d32 = nn.Dropout(p=0.1)
        # for 4
        self.m41 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f41 = nn.LSTM(input_size=11, hidden_size=128, num_layers=1, batch_first=True)
        self.ac41 = nn.Tanh()
        self.d41 = nn.Dropout(p=0.1)
        self.f42 = nn.Dense(128, 128, activation='relu')  # 线性层
        self.d42 = nn.Dropout(p=0.1)
        # for 5
        self.m51 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f51 = nn.LSTM(input_size=11, hidden_size=128, num_layers=1, batch_first=True)
        self.ac51 = nn.Tanh()
        self.d51 = nn.Dropout(p=0.1)
        self.f52 = nn.Dense(128, 128, activation='relu')  # 线性层
        self.d52 = nn.Dropout(p=0.1)
        # for 6
        self.m61 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f61 = nn.LSTM(input_size=11, hidden_size=128, num_layers=1, batch_first=True)
        self.ac61 = nn.Tanh()
        self.d61 = nn.Dropout(p=0.1)
        self.f62 = nn.Dense(128, 128, activation='relu')  # 线性层
        self.d62 = nn.Dropout(p=0.1)
        # for 7
        self.m71 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f71 = nn.LSTM(input_size=11, hidden_size=128, num_layers=1, batch_first=True)
        self.ac71 = nn.Tanh()
        self.d71 = nn.Dropout(p=0.1)
        self.f72 = nn.Dense(128, 128, activation='relu')  # 线性层
        self.d72 = nn.Dropout(p=0.1)
        # for 8
        self.m81 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f81 = nn.LSTM(input_size=11, hidden_size=128, num_layers=1, batch_first=True)
        self.ac81 = nn.Tanh()
        self.d81 = nn.Dropout(p=0.1)
        self.f82 = nn.Dense(128, 128, activation='relu')  # 线性层
        self.d82 = nn.Dropout(p=0.1)
        # for 9
        self.m91 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f91 = nn.LSTM(input_size=11, hidden_size=128, num_layers=1, batch_first=True)
        self.ac91 = nn.Tanh()
        self.d91 = nn.Dropout(p=0.1)
        self.f92 = nn.Dense(128, 128, activation='relu')  # 线性层
        self.d92 = nn.Dropout(p=0.1)
        # for 10
        self.m101 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f101 = nn.LSTM(input_size=11, hidden_size=128, num_layers=1, batch_first=True)
        self.ac101 = nn.Tanh()
        self.d101 = nn.Dropout(p=0.1)
        self.f102 = nn.Dense(128, 128, activation='relu')  # 线性层
        self.d102 = nn.Dropout(p=0.1)

        #shared layer
        # 此处和源代码不同
        self.f2 = nn.Dense(128, 11, activation='relu')  # 线性层
        self.d2 = nn.Dropout(p=0.1)
        self.m3 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f3 = nn.Dense(11, 128, activation='relu')
        self.d3 = nn.Dropout(p=0.1)
        self.f4 = nn.Dense(128, 11, activation='relu')
        self.d4 = nn.Dropout(p=0.1)
        #target layer
        #for 1
        # 此处和源代码不同，因为embed_dim的限制，与调用输入的x的维度是一样的
        self.m12 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f13 = nn.Dense(11, 128, activation='relu')
        self.d13 = nn.Dropout(p=0.1)
        self.f14 = nn.Dense(128, 1)
        #for 2
        self.m22 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f23 = nn.Dense(11, 128, activation='relu')
        self.d23 = nn.Dropout(p=0.1)
        self.f24 = nn.Dense(128, 1)
        #for 3
        self.m32 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f33 = nn.Dense(11, 128, activation='relu')
        self.d33 = nn.Dropout(p=0.1)
        self.f34 = nn.Dense(128, 1)
        #for 4
        self.m42 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f43 = nn.Dense(11, 128, activation='relu')
        self.d43 = nn.Dropout(p=0.1)
        self.f44 = nn.Dense(128, 1)
        #for 5
        self.m52 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f53 = nn.Dense(11, 128, activation='relu')
        self.d53 = nn.Dropout(p=0.1)
        self.f54 = nn.Dense(128, 1)
        #for 6
        self.m62 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f63 = nn.Dense(11, 128, activation='relu')
        self.d63 = nn.Dropout(p=0.1)
        self.f64 = nn.Dense(128, 1)
        #for 7
        self.m72 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f73 = nn.Dense(11, 128, activation='relu')
        self.d73 = nn.Dropout(p=0.1)
        self.f74 = nn.Dense(128, 1)
        #for 8
        self.m82 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f83 = nn.Dense(11, 128, activation='relu')
        self.d83 = nn.Dropout(p=0.1)
        self.f84 = nn.Dense(128, 1)
        #for 9
        self.m92 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f93 = nn.Dense(11, 128, activation='relu')
        self.d93 = nn.Dropout(p=0.1)
        self.f94 = nn.Dense(128, 1)
        #for 10
        self.m102 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first=True)
        self.f103 = nn.Dense(11, 128, activation='relu')
        self.d103 = nn.Dropout(p=0.1)
        self.f104 = nn.Dense(128, 1)
        
    # @ms.jit  # 使用ms.jit装饰器，使被装饰的函数以静态图模式运行    
    def construct(self, x):
        # print(x.shape)
        # in 1
        x1 = x[:,0:11][:, np.newaxis]
        x1,_ = self.m11(x1, x1, x1)
        x1,_ = self.f11(x1)
        x1 = self.ac11(x1)
        x1 = self.d11(x1)
        x1 = self.f12(x1)
        x1 = self.d12(x1)

        # in 2
        x2 = x[:,11:22][:, np.newaxis]
        x2,_ = self.m21(x2, x2, x2)
        x2,_ = self.f21(x2)
        x2 = self.ac21(x2)
        x2 = self.d21(x2)
        x2 = self.f22(x2)
        x2 = self.d22(x2)

        # in 3
        x3 = x[:,22:33][:, np.newaxis]
        x3,_ = self.m31(x3, x3, x3)
        x3,_ = self.f31(x3)
        x3 = self.ac31(x3)
        x3 = self.d31(x3)
        x3 = self.f32(x3)
        x3 = self.d32(x3)

        # in 4
        x4 = x[:,33:44][:, np.newaxis]
        x4,_ = self.m41(x4, x4, x4)
        x4,_ = self.f41(x4)
        x4 = self.ac41(x4)
        x4 = self.d41(x4)
        x4 = self.f42(x4)
        x4 = self.d42(x4)
        
        # in 5
        x5 = x[:,44:55][:, np.newaxis]
        x5,_ = self.m51(x5, x5, x5)
        x5,_ = self.f51(x5)
        x5 = self.ac51(x5)
        x5 = self.d51(x5)
        x5 = self.f52(x5)
        x5 = self.d52(x5)

        # in 6
        x6 = x[:,55:66][:, np.newaxis]
        x6,_ = self.m61(x6, x6, x6)
        x6,_ = self.f61(x6)
        x6 = self.ac61(x6)
        x6 = self.d61(x6)
        x6 = self.f62(x6)
        x6 = self.d62(x6)

        # in 7
        x7 = x[:,66:77][:, np.newaxis]
        x7,_ = self.m71(x7, x7, x7)
        x7,_ = self.f71(x7)
        x7 = self.ac71(x7)
        x7 = self.d71(x7)
        x7 = self.f72(x7)
        x7 = self.d72(x7)

        # in 8
        x8 = x[:,77:88][:, np.newaxis]
        x8,_ = self.m81(x8, x8, x8)
        x8,_ = self.f81(x8)
        x8 = self.ac81(x8)
        x8 = self.d81(x8)
        x8 = self.f82(x8)
        x8 = self.d82(x8)

        # in 9
        x9 = x[:,88:99][:, np.newaxis]
        x9,_ = self.m91(x9, x9, x9)
        x9,_ = self.f91(x9)
        x9 = self.ac91(x9)
        x9 = self.d91(x9)
        x9 = self.f92(x9)
        x9 = self.d92(x9)

        # in 10
        x10 = x[:,99:110][:, np.newaxis]
        x10,_ = self.m101(x10, x10, x10)
        x10,_ = self.f101(x10)
        x10 = self.ac101(x10)
        x10 = self.d101(x10)
        x10 = self.f102(x10)
        x10 = self.d102(x10)


        #shared layer

        x1 = self.f2(x1)
        x1 = self.d2(x1)
        x1,_ = self.m3(x1, x1, x1)
        x1 = self.f3(x1)
        x1 = self.d3(x1)
        x1 = self.f4(x1)
        x1 = self.d4(x1)

        x2 = self.f2(x2)
        x2 = self.d2(x2)
        x2,_ = self.m3(x2, x2, x2)
        x2 = self.f3(x2)
        x2 = self.d3(x2)
        x2 = self.f4(x2)
        x2 = self.d4(x2)

        x3 = self.f2(x3)
        x3 = self.d2(x3)
        x3,_ = self.m3(x3, x3, x3)
        x3 = self.f3(x3)
        x3 = self.d3(x3)
        x3 = self.f4(x3)
        x3 = self.d4(x3)
                
        x4 = self.f2(x4)
        x4 = self.d2(x4)
        x4,_ = self.m3(x4, x4, x4)
        x4 = self.f3(x4)
        x4 = self.d3(x4)
        x4 = self.f4(x4)
        x4 = self.d4(x4)

        x5 = self.f2(x5)
        x5 = self.d2(x5)
        x5,_ = self.m3(x5, x5, x5)
        x5 = self.f3(x5)
        x5 = self.d3(x5)
        x5 = self.f4(x5)
        x5 = self.d4(x5)

        x6 = self.f2(x6)
        x6 = self.d2(x6)
        x6,_ = self.m3(x6, x6, x6)
        x6 = self.f3(x6)
        x6 = self.d3(x6)
        x6 = self.f4(x6)
        x6 = self.d4(x6)

        
        x7 = self.f2(x7)
        x7 = self.d2(x7)
        x7,_ = self.m3(x7, x7, x7)
        x7 = self.f3(x7)
        x7 = self.d3(x7)
        x7 = self.f4(x7)
        x7 = self.d4(x7)

        x8 = self.f2(x8)
        x8 = self.d2(x8)
        x8,_ = self.m3(x8, x8, x8)
        x8 = self.f3(x8)
        x8 = self.d3(x8)
        x8 = self.f4(x8)
        x8 = self.d4(x8)
        
        x9 = self.f2(x9)
        x9 = self.d2(x9)
        x9,_ = self.m3(x9, x9, x9)
        x9 = self.f3(x9)
        x9 = self.d3(x9)
        x9 = self.f4(x9)
        x9 = self.d4(x9)

        
        x10 = self.f2(x10)
        x10 = self.d2(x10)
        x10,_ = self.m3(x10, x10, x10)
        x10 = self.f3(x10)
        x10 = self.d3(x10)
        x10 = self.f4(x10)
        x10 = self.d4(x10)


        #out1
        x1,_ = self.m12(x1, x1, x1)
        out1 = self.f13(x1)
        out1 = self.d13(out1)
        out1 = self.f14(out1)
        # out2
        x2,_ = self.m22(x2, x2, x2)
        out2 = self.f23(x2)
        out2 = self.d23(out2)
        out2 = self.f24(out2)
        # out3
        x3,_ = self.m32(x3, x3, x3)
        out3 = self.f33(x3)
        out3 = self.d33(out3)
        out3 = self.f34(out3)
        # out4
        x4,_ = self.m42(x4, x4, x4)
        out4 = self.f43(x4)
        out4 = self.d43(out4)
        out4 = self.f44(out4)
        # out5
        x5,_ = self.m52(x5, x5, x5)
        out5 = self.f53(x5)
        out5 = self.d53(out5)
        out5 = self.f54(out5)
        # out6
        x6,_ = self.m62(x6, x6, x6)
        out6 = self.f63(x6)
        out6 = self.d63(out6)
        out6 = self.f64(out6)
        # out7
        x7,_ = self.m72(x7, x7, x7)
        out7 = self.f73(x7)
        out7 = self.d73(out7)
        out7 = self.f74(out7)
        # out8
        x8,_ = self.m82(x8, x8, x8)
        out8 = self.f83(x8)
        out8 = self.d83(out8)
        out8 = self.f84(out8)
        # out9
        x9,_ = self.m92(x9, x9, x9)
        out9 = self.f93(x9)
        out9 = self.d93(out9)
        out9 = self.f94(out9)
        # out10
        x10,_ = self.m102(x10, x10, x10)
        out10 = self.f103(x10)
        out10 = self.d103(out10)
        out10 = self.f104(out10)

        # return out1, out2, out3, out4, out5, out6, out7, out8, out9, out10
    
        output2 = P.Stack(axis=3)([out1, out2, out3, out4, out5, out6, out7, out8, out9, out10])
        # print(out1)
        # print(output.shape)
        # print(output2.T.shape)
        # print(output2.shape)
        return output2
        

# %%
# 测试用
model = ML()
param_dict = ms.load_checkpoint("model.ckpt")   # 模型的路径
param_not_load, _ = ms.load_param_into_net(model, param_dict)
print(param_not_load)



# %%
import mindspore.nn as nn
import mindspore.ops as ops

# 引入测试的参数
criterion = nn.MSELoss()
squared = ops.Square()
sqrt = ops.Sqrt()

# %%

# 进行测试
model.set_train(False)
for batch, (data, label) in enumerate(test_dataset.create_tuple_iterator()):
    # print(data.shape)
    # print(label)
    pred = model(data)
    mse1 = criterion(pred, label)
    rmse1 = sqrt(mse1)
    # rmse2 = sqrt(mse2)
    plt.plot(np.array(pred.flatten()))
    plt.plot(np.array(label.flatten()))
    plt.xlabel('Data Index')
    plt.ylabel('Value')
    plt.legend(['Predicted', 'Actual Value'])
    plt.title(f'Comparison between Predicted and Actual Values : (Stop{batch+1})')
    filename = './pic/分布实验/' + f"Stop_{batch+1}.png"
    plt.savefig(filename)
    plt.show()
    
    


