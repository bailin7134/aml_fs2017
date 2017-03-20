require 'nngraph'
require 'torch'
require 'nn'   
require 'optim'

-- define inputs
local x1 = torch.rand(5)
local x2 = torch.rand(20)
local x3 = torch.rand(2,15)

-- model implementation, define an MLP

-- print the size of your outputs
outputs = mlp:forward({x1,x2,x3})
print(outputs[1]:size())
print(outputs[2]:size())
print(outputs[3]:size())