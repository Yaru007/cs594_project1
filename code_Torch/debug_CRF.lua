-- Use optim.checkgrad to check the overall gradient is computed correctly

require 'nn'
require 'optim'
require 'ClassCRFCriterion'
require 'LinearCRF'

torch.manualSeed(1)

local lambda = 1e-3

train = torch.load('../data/debug_torch.dat', 'ascii')

data = train.data
nFea = train.nFea
nState = train.nState
label = train.label		--array recording the label of all letters (1-26)
wLen = train.wLen			--array recording the number of letters in each word
cumwLen = torch.cumsum(wLen)
nWord = (#wLen)[1]

model = nn.Sequential()
model:add(nn.LinearCRF(nState, nFea, data))
criterion = nn.ClassCRFCriterion(nState, torch.max(wLen))
parameters,gradParameters = model:getParameters()

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

x = torch.randn((nFea+nState)*nState)
eps = 1e-6
diff,dC,dC_est = optim.checkgrad(feval, x, eps)
io.write('diff = ')
print(diff)
