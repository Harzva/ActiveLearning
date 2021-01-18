import torch.nn as nn
assert True,'ssss'

input_size = 28 * 28   
hidden_size = 500   
num_classes = 10    


# 三层神经网络
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到影藏层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 影藏层到输出层
        self.dropout = nn.Dropout(p=0.5)  # dropout训练
        self.input_size=input_size
        self.num_classes=num_classes
        self.hidden_size=hidden_size

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
DataParallel=True

model = NeuralNet(input_size, hidden_size, num_classes)
print('model.fc1',model.fc1)
print('model.input_size',model.input_size)
print(type(model))
# print(model.parameters)
# print(model.state_dict)
print(model.input_size)
print(model.num_classes)
# print(model.module)
# print(dir(model))
print('*'*60,'model\n',model)
model = nn.DataParallel(model , device_ids=[0,1])
print('*'*60,'model.module\n',model.module)
print(model.module.num_classes)
print('DataParallel after model.module.fc1',model.module.fc1)
print('DataParallel after model.module.input_size',model.module.input_size)
# print(model)
# model.train()
# model.eval()
