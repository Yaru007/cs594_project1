require 'nn'
require 'optim'   
require 'ClassCRFCriterion'
require 'LinearCRF'
require 'sgd'
require 'sag_nus'

cmd = torch.CmdLine()
cmd:option('-lambda',1e-3,'regularization parameter')
cmd:option('-optim', 'LBFGS' ,'optimization method')
cmd:option('-iter',50,'number of maximal iterations')
cmd:option('-lr',1e-1,'learning rate')
cmd:option('-b',100,'number of words in gradient update')
cmd:option('-n',100,'number of words used for training')
-- parse input params
params = cmd:parse(arg)
-- print(params)
	
if params.optim == 'sgd' then
	optimMethod = sgd
elseif params.optim == 'sag' then
	optimMethod = sag_nus
else
	-- default
	optimMethod = optim.lbfgs		-- use LBFGS as the optimization solver
end

-- optimMethod = sgd
-- optimMethod = sag_nus

local lambda = params.lambda -- 1e-3  -- the lambda value in Eq (4)

torch.manualSeed(1)	-- fix the seed of the random number generator for reproducing the results

opt = {}
opt.maxIter = params.iter			-- max number of iterations in BFGS
opt.nCorrection = 10 		-- number of previous gradients used to approximate the Hessian
opt.learningRate = params.lr	-- fixed step size used in the line search of BFGS

-- Specify the filenames of the training and test data
local train_fname = '../data/train_torch.dat'
local test_fname = '../data/test_torch.dat'

capWord = params.n	-- only use the first 100 words. Otherwise it takes too long to train.
-- uncomment the next line if you do not want to cap #words
-- capWord = math.huge  

optimState = {
	learningRate = opt.learningRate, -- used if 'lineSearch' is not specified
	maxIter = opt.maxIter,
	nCorrection = opt.nCorrection,
	verbose = true,			-- print a bit more information from the solver
	lambda = params.lambda
	-- lineSearch = optim.lswolfe,  -- if we use this then the function might 
	-- be evaluated multiple times at each iteration
}

----------------------------------------------------------------------
local train = torch.load(train_fname, 'ascii')
data = train.data			
nFea = train.nFea				-- number of features (129)
nState = train.nState			-- number of possible labels (26)
label = train.label				-- array recording the label of all letters (1-26)
wLen = train.wLen				-- array recording the number of letters in each word
cumwLen = torch.cumsum(wLen)	-- cumulative sum of wLen

local test = torch.load(test_fname, 'ascii')
tdata = test.data
assert(nFea == test.nFea)
assert(nState == test.nState)
tlabel = test.label				-- array recording the label of all letters (1-26)
twLen = test.wLen				-- array recording the number of letters in each word
tcumwLen = torch.cumsum(twLen)	-- cumulative sum of wLen

-- max #letter throughout all words
max_word_len = torch.max(torch.DoubleTensor{wLen:max(), twLen:max()}) 

-- Cap the number of words used for training at capWord
nWord = (#train.wLen)[1]
-- if capWord < 0, train on all the words
if capWord > 0 and nWord > capWord then nWord = capWord  end
optimState.nWord = nWord
optimState.fevalIntervel = params.b			-- compute objective function after the number of updates
optimState.L = torch.ones(nWord)			-- Lipschitz constant for each training example
optimState.sampleRecord = torch.zeros(nWord)			-- record of training example if it is sampled before
optimState.backtrackingSkip = torch.zeros(nWord)			-- number of sampled word consecutively avoids backtracking
optimState.lineSearchToSkip = torch.Tensor(nWord):fill(-1)			-- number of next samples that line search can be skipped

print('==> training in progress with ' .. nWord .. ' words')

-- Construct the network that represents CRF
model = nn.Sequential()
model:add(nn.LinearCRF(nState, nFea, data))
criterion = nn.ClassCRFCriterion(nState, torch.max(wLen))
parameters,gradParameters = model:getParameters()

tmodel = nn.Sequential()
tmodel:add(nn.LinearCRF(nState, nFea, tdata))
tcriterion = nn.ClassCRFCriterion(nState, torch.max(twLen))
tparameters,tgradParameters = tmodel:getParameters()

-- Function that computes the objective value and its gradient
local function feval(x)
	parameters:copy(x)
	gradParameters:zero()	
	local f = 0

	-- evaluate function by enumerating all words
	for i = 1,nWord do
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

-- get gradient and object value of ith word
local function singleEval(x, idx)
	parameters:copy(x)
	gradParameters:zero()	

	-- out of scope
	if idx < 1 or idx > nWord then
		return nil
	end
    
	-- estimate f
	first = idx > 1 and cumwLen[idx-1]+1 or 1	--index of the first letter in the word
	last = cumwLen[idx]											--index of the last letter in the word
	local input = torch.linspace(first, last, last-first+1)	
	local output = model:forward(input)
	local target = label:sub(first, last)
	local f = criterion:forward(output, target)

	-- estimate df/dW
	local df_do = criterion:backward(output, target)
	model:backward(input, df_do)		

	-- gradients and f(X), add regularization
	-- gradParameters:add(lambda*x)
	-- f = f + lambda*torch.pow(x:norm(), 2)/2

	-- return df/dX
	return f, gradParameters
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

	--[[
	Your code to compute the test errors
	--]]
	local tlError = 0
	local twError = 0
	local tnWord = (#test.wLen)[1]

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
	io.write(string.format("%.2f %.2f %.2f %.2f", tr_let_err, tr_word_err, te_let_err, te_word_err))
end

-- sample one training example uniformly and return its f and gradient
local function uniformSample(x)
	local idx = math.random(nWord)
	-- print('sample ', idx)
	return idx, singleEval(x, idx)
end

-- input:	x:	weight
--			idx:	sample id
--			Li: 	old Lipschitz constant
--			fi:		objective function value without regulation term
--			gi: 	gradient without regulation term
-- output: 	Li: Lipschitz constant for each training example
local function Llinesearch(x,idx,Li,fi,gi)
    local x_p = torch.add(x, -1/Li, gi)
    local f_p,g_p = singleEval(x_p,idx)
    local g_norm = torch.pow(gi:norm(), 2)
	--print('--------------------')
	local count = 0
    while f_p >= (fi - 1/2/Li*g_norm) do
    	Li = 2*Li
     	x_p:add(-1/Li, gi)
        f_p,g_p = singleEval(x_p,idx)
        -- print(idx, Li, f_p, fi, (fi - 1/2/Li*g_norm))
    end
    -- restore the old w parameter of the model
    parameters:copy(x)
    return Li
end

-- input x: w in CRF problem
local function nonUniformSample(x, L)
	local randi = math.random()
	local rand_idx = 0
	-- first time or half probability
	if L:sum() == L:size(1) or randi > 0.5 then
		rand_idx = math.random(nWord)
	else
		-- get the weight of L
		local p = torch.div(L, L:sum())
		-- sample by weight
		rand_idx = torch.multinomial(p,1,true)[1]
	end
	
	-- print(rand_idx)
	local fi, gi = singleEval(x, rand_idx)
	return rand_idx, fi, gi
end

-- Set the monitor for the solver.
-- If commented out, then the training/test errors won't be printed/computed.
optimState.monitor = Monitor
if params.optim == 'sgd' then
	optimState.sampler = uniformSample
end

if params.optim == 'sag' then
	optimState.sampler = nonUniformSample
	optimState.lineSearch = Llinesearch
end

-- Really start the optimization
 print('iter fn.val gap time feval.num train_lett_err train_word_err test_lett_err test_word_err')
 x,fx = optimMethod(feval, parameters, optimState)
-- for i=1,#fx do print(i,fx[i]); end	-- secrete
