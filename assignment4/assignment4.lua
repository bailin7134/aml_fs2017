require 'nngraph'
require 'torch'
require 'nn'   
require 'optim'

-- define labels
local label1 = torch.rand(1,3,64,64)
local label2 = torch.rand(1,3,64,64)
-- TODO
local function createResBlock()
	-- local resBlock = nn.Sequential()
	-- TODO implement resblock in this function
	-- do not implement this separately
	local cat = nn.Sequential()
	cat:add(nn.SpatialConvolution(32, 32, 3, 3,1,1,1,1))
	cat:add(nn.ReLU(true))
	cat:add(nn.SpatialConvolution(32, 32, 3, 3,1,1,1,1))
	-- concate block
	local conBlock = nn.ConcatTable()
	conBlock:add(nn.Identity())
	conBlock:add(cat)
	--local conResult = conBlock:forward(input)
	-- add block
	--local addBlock = nn.CAddTable()({conBlock})
	local resBlock = nn.Sequential()
	resBlock:add(conBlock)
	resBlock:add(nn.CAddTable())
	if arg[1] == "debug" then
		print(resBlock)
	end
	return resBlock
end
local model = nn.ParallelTable()
local L1Net = nn.Sequential()
local L2Net = nn.Sequential()
local conv1 = nn.Sequential()
local conv4 = nn.Sequential()
-- Define conv1 and conv4 layers
conv1:add(nn.SpatialConvolution(3, 32, 3, 3,1,1,1,1))
conv4:add(nn.SpatialConvolution(32, 3, 3, 3,1,1,1,1))

-- TODO add shared conv1 layer to L1Net and L2Net
L1Net:add(conv1)
L2Net = L1Net:clone('weight', 'bias', 'gradWeight', 'gradBias')
-- TODO add ResBlock to L1Net and L2Net
L1Net:add(createResBlock())
L2Net:add(createResBlock())
-- TODO add shared conv4 layer to L1Net and L2Net
L1Net:add(conv4)
L2Net = L1Net:clone('weight', 'bias', 'gradWeight', 'gradBias')
-- TODO add L1Net and L2Net to model
model:add(L1Net)
model:add(L2Net)
if arg[1] == "debug" then
	print(model)
end

-- define criterion
criterion1 = nn.MSECriterion()
criterion2 = nn.AbsCriterion()
if model then
   parameters,gradParameters = model:getParameters()
end
print '==> configuring optimizer'
optimMethod = optim.adam
local parameters, gradParameters = model:getParameters()
feval = function(x)
  model:zeroGradParameters()
  local inputs1 = torch.rand(1,3,64,64)
  local inputs2 = torch.rand(1,3,64,64)
  outputs = model:forward({inputs1, inputs2}) 
  err1 = criterion1:forward(outputs[1], label1)
  err2 = criterion2:forward(outputs[2], label2)

  local gradOutputs1   = criterion1:backward(outputs[1], label1)
  local gradOutputs2   = criterion2:backward(outputs[2], label2)

  model:backward({inputs1, inputs2}, {gradOutputs1, gradOutputs2})
  err = err1 + err2
  return err, gradParameters
end
-- run 10 iterations
for iii=1,10 do
	optim.adam(feval, parameters, optimState)
	print(err)
end
-- Test whether parameters are shared, don't remove this part
local m1 = nn.Sequential()
m1:add(model.modules[1].modules[1])
local m2 = nn.Sequential()
m2:add(model.modules[2].modules[1])
parameters1,gradParameters1 = m1:getParameters()
parameters2,gradParameters2 = m2:getParameters()
local diff = parameters1 - parameters2
print(diff:min())
print(diff:max())
