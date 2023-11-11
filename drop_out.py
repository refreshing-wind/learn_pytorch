import numpy as np
import torch

class Test(torch.nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.linear1 = torch.nn.Linear(2,3)
        self.linear2 = torch.nn.Linear(3,4)
        self.batch_norm = torch.nn.BatchNorm2d(4)
        self.drop_out = torch.nn.Dropout(0.5)
test_module = Test()
for p in test_module.named_parameters():
    print(p)
# dropout中没有可训练的参数
def train(rate, x, w1, b1, w2, b2):
    layer1 = np.maximum(0, np.dot(w1, x) + b1)
    mask1 = np.random.binomial(1, 1-rate, layer1.shape)
    layer1 = layer1*mask1
    layer2 = np.maximum(0, np.dot(w2, layer1) +b2)
    mask2 = np.random.binomial(1, 1-rate, layer2.shape)
    layer2 = layer2*mask2
    return layer2
def test(rate, x, w1, b1, w2, b2):
    layer1 = np.maximum(0, np.dot(w1, x)+b1)
    layer1 = layer1*(1-rate)
    layer2 = np.maximum(0, np.dot(w2, layer1)+b2)
    layer2 = layer2(1-rate)
    return layer2

def another_train(rate, x, w1, b1, w2, b2):
    layer1 = np.maximum(0, np.dot(w1, x)+b1)
    mask1 = np.random.binomial(1, 1-rate, layer1.shape)
    layer1 = layer1*mask1
    layer1 = layer1/(1-rate)
    layer2 = np.maximum(0, np.dot(w2, layer1)+b2)
    mask1 = np.random.binomial(1, 1-rate, layer2.shape)
    layer2 = layer2*mask1
    layer2 = layer2/(1-rate)

    return layer2

def another_test(x, w1, b1, w2, b2):
    layer1 = np.maximum(0, np.dot(w1, x)+b1)
    layer2 = np.maximum(0, np.dot(w2, layer1)+b2)
    return layer2


def train_r_drop(rate, x, w1, b1, w2, b2):
    x = torch.cat([x, x], 0)
    layer1 = np.maximum(0, np.dot(w1, x)+b1)
    mask1 = np.random.binomial(1, 1-rate, layer1.shape)
    layer1 = layer1*mask1
    layer2 = np.maximum(0, np.dot(w2, layer1)+b2)
    mask2 = np.random.binomial(1, 1-rate, layer2.shape)
    layer2 = layer2*mask2
    # nllloss首先需要初始化
    nllloss = torch.nn.NLLLoss() # 可选参数中有 reduction='mean', 'sum', 默认mean
    logits = torch.log(torch.softmax(layer2, dim=-1))
    logits1, logits2 = logits[:batch_size, :], logits[batch_size:, :]
    nll1 = nllloss(logits1, label)
    nll2 = nllloss(logits2, label)
    kl = torch.nn.KLDivLoss(reduction="batchmean")
    kl_loss = kl(logits1, logits2)
    loss = nll1+nll2+kl_loss

    return loss

def another_test(x, w1, b1, w2, b2):
    layer1 = np.maximum(0, np.dot(w1, x)+b1)
    layer2 = np.maximum(0, np.dot(w2, layer1)+b2)
    return layer2
