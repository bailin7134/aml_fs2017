
require 'hdf5'
require 'gnuplot'

function sigmoid(z)
    -- 1/1+exp(-z)
    g = torch.FloatTensor(1,1):fill(1):cdiv(1.0+torch.exp(-z))
    return g
end

function train_logistic_sgd(data, labels)

    -- learning rate
    local alpha = torch.FloatTensor(1,1):fill(0.00005)

    local p = data:size(1)  -- size of the features
    local n = data:size(2)  -- number of train samples
    local theta = torch.FloatTensor(1,p):fill(0)  -- initialize the value of theta

    for k = 1, n, 1 do
        i = math.random(1,n)
        gradJ = data[{{}, {i}}] * (labels[i] - sigmoid(theta * data[{{}, {i}}])):view(1, 1)
        theta = theta + gradJ * alpha:view(1, 1)
    end
    return theta

end



train_file = hdf5.open('train.h5', 'r')
train_data = train_file:read('data')
train_feats = train_data:all()
train_labels = train_file:read('label')
train_labels = train_labels:all()
n = train_feats:size(1)
p = train_feats:size(4)
train_feats = torch.reshape(train_feats, n,p)

data = torch.cat(train_feats, torch.FloatTensor(n,1):fill(1),2):t()

theta = train_logistic_sgd(data, train_labels)

test_file = hdf5.open('test.h5', 'r')
test_data = test_file:read('data')
test_feats = test_data:all()
test_labels = test_file:read('label')
test_labels = test_labels:all()
n = test_feats:size(1)
p = test_feats:size(4)
test_feats = torch.reshape(test_feats, n,p)

scores = theta*(torch.cat(test_feats, torch.FloatTensor(n,1):fill(1),2):t())
y, idx = torch.sort(scores, 2, true)
sorted_ytest = test_labels:index(1,idx:reshape(n))
positives=sorted_ytest:gt(0):reshape(n)
precision = torch.cdiv(torch.cumsum(positives:float()), torch.FloatTensor(n,1):range(1,n))
recall  = torch.cdiv(torch.cumsum(positives:float()), torch.FloatTensor(n,1):fill(positives:sum()))
ap=0;
for t=0,.99,.099 do
    a = precision[recall:gt(t)]
    if a:dim()==0 then
	p = 0
   else
	p = torch.max(a)	
   end
   ap=ap+p/11;
end
gnuplot.pngfigure('precision_recall.png')
gnuplot.plot(precision, recall)
gnuplot.xlabel('recall')
gnuplot.ylabel('precision')
gnuplot.plotflush()

tpr = recall
fpr = torch.cdiv(recall, precision)-recall
gnuplot.pngfigure('ROC.png')
gnuplot.plot(fpr, tpr)
gnuplot.xlabel('false positive rate')
gnuplot.ylabel('true positive rate')
gnuplot.plotflush()
print(string.format('average precision: %.2f', ap*100))




