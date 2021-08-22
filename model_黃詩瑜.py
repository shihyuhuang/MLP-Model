#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data = open("/home/U111syhuang/HW4/training_data.txt",'r')
train_data_input = np.empty((10000,4))
train_data_golden_output = np.empty((10000,1))

#print(train_data_input)
#print(train_data_input[0])
#print(train_data_golden_output)
#print(train_data_golden_output[0])

lines = train_data.readlines()
for line in range(20000):
    if line % 2 == 0:
        #print(lines[line].rstrip('\n').split('   '))
        train_data_input[line>>1] = list(map(float,lines[line].rstrip('\n').split(' ')))
    else:
        #print(lines[line].rstrip('\n'))  
        train_data_golden_output[line>>1] = float(lines[line].rstrip('\n'))

print(train_data_input)
print(train_data_golden_output)



idx = np.arange(10000)
np.random.shuffle(idx)
train_idx = idx[:9900]
val_idx = idx[9900:]

x_train, y_train = train_data_input[train_idx], train_data_golden_output[train_idx]
x_val, y_val = train_data_input[val_idx], train_data_golden_output[val_idx]
print(x_train)
print(y_train)
x_train_tensor = torch.from_numpy(x_train)
y_train_tensor = torch.from_numpy(y_train)
x_val_tensor = torch.from_numpy(x_val)
y_val_tensor = torch.from_numpy(y_val)

class myDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, file_num):
        self.x = x_tensor
        self.y = y_tensor
        self.file_num = file_num
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])
    
    def __len__(self):
        return self.file_num
    
training_data = myDataset(x_train_tensor,y_train_tensor,9900)
val_data = myDataset(x_val_tensor,y_val_tensor,100)
print(training_data[0])


# In[2]:


test_data = open("/home/U111syhuang/HW4/test_data_in.txt",'r')
test_data_input = np.empty((100,4))
lines = test_data.readlines()
for line in range(100):
    test_data_input[line] = list(map(float,lines[line].rstrip('\n').split(' ')))


print(test_data_input)
test_data_input_tensor = torch.from_numpy(test_data_input)


# In[3]:


class MLP(torch.nn.Module):
  def __init__(self):
    super(MLP,self).__init__()
    self.fc1 = torch.nn.Linear(4,5)
    self.fc2 = torch.nn.Linear(5,3)
    self.fc3 = torch.nn.Linear(3,1)


    
    
  def forward(self, node_1_in):
    node_1_out = self.fc1(node_1_in)
    node_2_in = F.relu(node_1_out)
    node_2_out = self.fc2(node_2_in)
    node_3_in = F.relu(node_2_out)
    node_3_out = self.fc3(node_3_in)
    return node_3_out


torch.manual_seed(7)
model = MLP().to(device)
MSELoss = torch.nn.MSELoss()


# In[14]:


#optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.001, momentum = 0.9)
optimizer = torch.optim.Adagrad(params = model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
train_loader = DataLoader(dataset=training_data, batch_size=100, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=True)

losses =[]



epochs = 10

for epoch in range(epochs):
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        model.train()
        yhat = model(x_batch.float())
        loss = MSELoss(y_batch.float(),yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    losses.append(loss.item())
    print('losses: ', loss.item())
    
for x_batch, y_batch in val_loader:
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)  
    model.eval()
    yhat = model(x_batch.float())
    #print('x: ',x_batch,'y: ',yhat)
    diff = y_batch.float() - yhat
    print(diff)

print('---------------ans---------------') 
with torch.no_grad():
    for x in test_data_input_tensor:
        x = x.to(device)
        model.eval()
        yhat = model(x.float())
        print(yhat.cpu().numpy()[0])
   
#print(model.state_dict())
plt.plot(losses) 
plt.show()


# In[ ]:




