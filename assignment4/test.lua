require"nn" 
local i2w = {"ball","is","red","the"} 
local w2i = {ball = 1, is = 2, red = 3, the = 4} 
local ngram = 3 
local nhid = 10 
local nproj = (ngram-1) * nhid
-- Bengio et al. 2003 for 3-grams: 
local feedforward = nn.Sequential() 
feedforward:add(nn.LookupTable(#i2w,nhid)) 
feedforward:add(nn.View(nproj)) 
feedforward:add(nn.Sigmoid()) 
feedforward:add(nn.Linear(nproj,nhid)) 
feedforward:add(nn.Sigmoid()) 
feedforward:add(nn.Linear(nhid, #i2w))

if arg[1] == "debug" then
	print(feedforward)
end
