require 'nn'
require 'optim'   
require 'ClassCRFCriterion'
require 'LinearCRF'
require 'SGD'

torch.manualSeed(1)	-- fix the seed of the random number generator for reproducing the results

local lambda = 1e-3  -- the lambda value in Eq (4)

opt = {}
opt.maxIter = 50			-- max number of iterations in BFGS
--opt.maxIter = 5
opt.nCorrection = 10 	-- number of previous gradients used to approximate the Hessian
opt.learningRate = 1e-1		-- fixed step size used in the line search of BFGS

-- Specify the filenames of the training and test data
local train_fname = '../data/train_torch.dat'
local test_fname = '../data/test_torch.dat'

capWord = 100	-- only use the first 100 words. Otherwise it takes too long to train.
-- uncomment the next line if you do not want to cap #words
-- capWord = math.huge  

optimState = {
    learningRate = opt.learningRate, -- used if 'lineSearch' is not specified
    maxIter = opt.maxIter,
    verbose = true		-- print a bit more information from the solver
   }
optimMethod = sgd	-- use sgd as the optimization solver

----------------------------------------------------------------------
local train = torch.load(train_fname, 'ascii')
data = train.data			
nFea = train.nFea			-- number of features (129)
nState = train.nState	-- number of possible labels (26)
label = train.label		-- array recording the label of all letters (1-26)
wLen = train.wLen			-- array recording the number of letters in each word
cumwLen = torch.cumsum(wLen)	-- cumulative sum of wLen

local test = torch.load(test_fname, 'ascii')
tdata = test.data
assert(nFea == test.nFea)
assert(nState == test.nState)
tlabel = test.label		-- array recording the label of all letters (1-26)
twLen = test.wLen			-- array recording the number of letters in each word
tcumwLen = torch.cumsum(twLen)	-- cumulative sum of wLen

-- max #letter throughout all words
max_word_len = torch.max(torch.DoubleTensor{wLen:max(), twLen:max()}) 

-- Cap the number of words used for training at capWord
nWord = (#train.wLen)[1]
tnWord = (#test.wLen)[1]
if nWord > capWord then nWord = capWord  end


print('==> training in progress with ' .. nWord .. ' words')

-- Construct the network that represents CRF
model = nn.Sequential()
model:add(nn.LinearCRF(nState, nFea, data))
criterion = nn.ClassCRFCriterion(nState, torch.max(twLen))
parameters,gradParameters = model:getParameters()

-- Construct the network that represents CRF for the test set
tmodel = nn.Sequential()
tmodel:add(nn.LinearCRF(nState, nFea, tdata))
tcriterion = nn.ClassCRFCriterion(nState, torch.max(twLen))
tparameters,tgradParameters = tmodel:getParameters()

-- Function that computes the objective value and its gradient
local function feval(x)

	if parameters ~= x then
		parameters:copy(x)
	end

	local f = 0
	word_order = torch.randperm(nWord)      -- Permute the word

	-- evaluate function by enumerating all words
	for i = 1,nWord do
	  index = word_order[i]  -- get the index of 
	  -- estimate f
	  first = i > 1 and cumwLen[i-1]+1 or 1	--index of the first letter in the word
	  last = cumwLen[i]											--index of the last letter in the word
		local input = torch.linspace(first, last, last-first+1)		
		local output = model:forward(input)
		local target = label:sub(first, last)
	  f = f + criterion:forward(output, target)
	
	  -- estimate df/dW
	  local df_do = criterion:backward(output, target)
	  model:backward(input, df_do)		  
	end
	
	-- normalize gradients and f(X), add regularization
	gradParameters:div(nWord):add(lambda*x)
	f = f/nWord + lambda*torch.pow(x:norm(), 2)/2
	
	-- return f and df/dX
	return f,gradParameters
end

-- set model to training mode (for modules that differ in training and testing, like Dropout)
model:training()


--- Allocate some persistent memory for MAP inference
--- Functions dp_argmax and find_MAP are for MAP inference
argmax = torch.DoubleTensor(max_word_len, nState)
labels = torch.DoubleTensor(max_word_len)

local function dp_argmax(i, nodePot, edgePot)
	if i == 1 then
		res = nodePot[1]
	else
		res = dp_argmax(i-1, nodePot, edgePot)
		res, argmax[i] = torch.max(edgePot + torch.repeatTensor(res:resize(nState,1), 1, nState), 1)	 	  	
	 	res:add(nodePot[i])	 	  	
	end
	return res
end


local function find_MAP(nodePot, edgePot)	
	nLetter = nodePot:size(1)
	max_val, labels[nLetter] = torch.max(dp_argmax(nLetter, nodePot, edgePot), 2)
	for i=nLetter-1,1,-1 do
		labels[i] = argmax[i+1][labels[i+1]]
  end
	return labels:sub(1,nLetter)
end

-- Monitor is used to print training/test error for each intermediate solution x
-- The BFGS solver will call it with the current x furnished.
-- Here we just show an example on how to compute the most likely decoding sequence
-- Only the computation of training error is shown here.
-- You need to fill in the code to compute the test error 
--   See the online tutorial for computing the test error:
--    https://web.archive.org/web/20151116000833/http://code.madbits.com/wiki/doku.php?id=tutorial_supervised_5_test
local function Monitor(x)

	te_word_err = -1		-- dummy, to be set by your code
	te_let_err = -1			-- dummy, to be set by your code

	local lError = 0
	local wError = 0
	parameters:copy(x)


	for i = 1,nWord do  -- loop over training words	  
		first = i > 1 and cumwLen[i-1]+1 or 1		--index of the first letter in the word
		last = cumwLen[i]							--index of the last letter in the word
		lenWord = last-first+1
		local input = torch.linspace(first, last, lenWord)		
		local output = model:forward(input)
		local target = label:sub(first, last)

		-- Call the MAP routine to fine the most likely output as our prediction
		pred_word = find_MAP(output.outNode, output.wEdge)

		-- Compare with the ground truth
		local wFail = false
		for j = 1,lenWord do
			if pred_word[j] ~= target[j] then
				lError = lError + 1
				wFail = true
			end
		end
		if wFail then wError = wError + 1 end
	end	
	tr_word_err = wError * 100.0 / nWord
	tr_let_err = lError * 100.0 / cumwLen[nWord]

--  Estimate the test error

	local tlError = 0
	local twError = 0

	tparameters:copy(x)
	for i = 1,tnWord do  -- loop over testing words	  
		local tfirst = i > 1 and tcumwLen[i-1]+1 or 1			--index of the first letter in the word
		local tlast = tcumwLen[i]								--index of the last letter in the word
		local tlenWord = tlast-tfirst+1
		local tinput = torch.linspace(tfirst, tlast, tlenWord)
		local toutput = tmodel:forward(tinput)
		local ttarget = tlabel:sub(tfirst, tlast)

		-- Call the MAP routine to fine the most likely output as our prediction
		tpred_word = find_MAP(toutput.outNode, toutput.wEdge)

		-- Compare with the ground truth
		local twFail = false
		for j = 1,tlenWord do
			if tpred_word[j] ~= ttarget[j] then
				tlError = tlError + 1
				twFail = true
			end
		end
		if twFail then twError = twError + 1 end
	end	
	te_word_err = twError * 100.0 / tnWord
	te_let_err = tlError * 100.0 / tcumwLen[tnWord]

	-- Different from print(), io.write does not append the newline.
	io.write(string.format("%.2f %.2f %.2f %.2f ", tr_let_err, tr_word_err, te_let_err, te_word_err))
end


-- Set the monitor for the solver.
-- If commented out, then the training/test errors won't be printed/computed.
optimState.monitor = Monitor


-- Really start the optimization
print('iter fn.val gap time train_lett_err train_word_err test_lett_err test_word_err')

x,fx = optimMethod(feval, parameters, optimState)
