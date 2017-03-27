require 'nngraph'
require 'torch'
require 'nn'   
require 'optim'

-- define inputs
local x1 = torch.rand(5)
local x2 = torch.rand(20)
local x3 = torch.rand(2,15)

-- model implementation, define an MLP
input1 = nn.Identity()()
h11 = nn.Linear(5,10)(input1)
h12 = nn.Linear(5,15)(input1)
h1 = nn.JoinTable(1)({h11,h12})

input2 = nn.Identity()()
h21 = nn.Linear(25,11)(h1)
h22 = nn.Linear(20,12)(input2)
h2 = nn.JoinTable(1)({h21,h22})

input3 = nn.Identity()()
layer3 = nn.Sequential()
layer3:add(nn.SplitTable(1))
layer3:add(nn.ParallelTable():add(nn.Linear(15,9)):add(nn.Linear(15,14)))
h31, h32 = layer3(input3):split(2)
h33 = nn.JoinTable(1)({h31,h32})
h3 = nn.JoinTable(1)({h33,h2})

mlp = nn.gModule({input1,input2,input3}, {h1,h2,h3})

-- print the size of your outputs
outputs = mlp:forward({x1,x2,x3})
print(outputs[1]:size())
print(outputs[2]:size())
print(outputs[3]:size())
