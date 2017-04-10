
require 'hdf5'
require 'nn'
require 'torch'

train_file = hdf5.open('train.h5', 'r')
train_data = train_file:read('data')
train_feats = train_data:all()
train_labels = train_file:read('label')
train_labels = train_labels:all()
n = train_feats:size(1)
p = train_feats:size(4)
train_feats = torch.reshape(train_feats, n,p)

test_file = hdf5.open('test.h5', 'r')
test_data = test_file:read('data')
test_feats = test_data:all()
test_labels = test_file:read('label')
test_labels = test_labels:all()
n = test_feats:size(1)
p = test_feats:size(4)
test_feats = torch.reshape(test_feats, n,p)

print('data is loaded successfully!')

-- model, logistic regression
local model = nn.Sequential()
model:add(nn.Reshape(p))
model:add(nn.Linear(p,2))
model:add(nn.LogSoftMax())
-- preprocess
model:double()
-- criterion

-- training
x, dl_dx = model:getParameters()
model:zeroGradParameters()

sumError = 0
learning_rate = 0.0001
-- adapt to softmax
train_labels = train_labels + 1
test_labels = test_labels + 1
for iter = 1, n do
	-- generate random numbers
	idx = math.random(1,n)

    -- feature and target extraction
    local features = torch.DoubleTensor(p):fill(0)
    features[{}] = train_feats[{idx,{}}]
    local target = torch.DoubleTensor(1):fill(0)
    target[{}] = train_labels[idx]

	local criterion = nn.ClassNLLCriterion()
	local prediction = model:forward(features)
	local err = criterion:forward(prediction, target)
	model:zeroGradParameters()
	local gradient = criterion:backward(prediction, target)
	model:backward(features, gradient)
	model:updateParameters(learning_rate)
end

-- test
-- threshold
corretNo = 0
for idx = 1, n do
	-- test cases
    local features = torch.DoubleTensor(p):fill(0)
    features[{}] = test_feats[{idx,{}}]
    local target = torch.DoubleTensor(1):fill(0)
    target = test_labels[idx]
	-- prediction
	local output = model:forward(features)
	if output[1] > output[2] then
		prediction = 1
	else
		prediction = 2
	end

	if prediction==target[1] then
		corretNo = corretNo + 1
	end

end
print(string.format('Accuracy is: %.2f', corretNo/n))

