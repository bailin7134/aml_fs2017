require 'nngraph'
require 'torch'
require 'nn'   
require 'optim'

-- define labels
local label1 = torch.rand(1,3,64,64)
local label2 = torch.rand(1,3,64,64)

input1=torch.rand(1,32,64,64):double()

input = nn.Identity()()
cat = nn.Sequential()
cat:add(nn.SpatialConvolution(32, 32, 3, 3,1,1,1,1))
cat:add(nn.ReLU(true))
cat:add(nn.SpatialConvolution(32, 32, 3, 3,1,1,1,1))
--cat:forward(input)
--print(cat.output:size())

conBlock = nn.ConcatTable()
conBlock:add(nn.Identity())
conBlock:add(cat)
conResult = conBlock:forward(input1)

addBlock = nn.CAddTable()
resBlock = addBlock:forward(conResult)

--pred = conBlock:forward(input1)
print(resBlock:size())
--test = nn.CAddTable()(pred)
--print(test:size())
--for i, k in ipairs(pred) do print(i, k) end
--output = resBlock(input)
--mlp = nn.gModule({input,input}, output)
--resBlock:forward(inputDouble)
-- print(resBlock.output:size())
--resBlock:add(nn.CAddTable(true))
--resBlock:forward(inputDouble)

local function createResBlock(input)
	-- local resBlock = nn.Sequential()
	-- TODO implement resblock in this function
	-- do not implement this separately
	local cat = nn.Sequential()
	cat:add(nn.SpatialConvolution(1, 1, 3, 3,1,1,1,1))
	cat:add(nn.ReLU(true))
	cat:add(nn.SpatialConvolution(1, 1, 3, 3,1,1,1,1))
	-- concate block
	local conBlock = nn.ConcatTable()
	conBlock:add(nn.Identity())
	conBlock:add(cat)
	conBlock:add(nn.CAddTable())
	--local conResult = conBlock:forward(input)
	-- add block
	--local addBlock = nn.CAddTable()({conBlock})
	print(conBlock)
	--local resBlock = addBlock:forward(conResult)
	return conBlock
end

input2=torch.rand(1,1,3,3):double()
local cat = nn.Sequential()
cat:add(nn.SpatialConvolution(1, 1, 3, 3,1,1,1,1))
cat:add(nn.ReLU(true))
cat:add(nn.SpatialConvolution(1, 1, 3, 3,1,1,1,1))
local conBlock = nn.ConcatTable()
conBlock:add(nn.Identity())
conBlock:add(cat)
conBlock:add(nn.CAddTable())
--local conResult = conBlock:forward(input)
-- add block
--local addBlock = nn.CAddTable()({conBlock})
print(conBlock)
--local resBlock = addBlock:forward(conResult)
conBlock:forward(input2)
print(conBlock.output)
