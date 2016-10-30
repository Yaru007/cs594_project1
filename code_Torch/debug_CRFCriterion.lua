-- Test the criterion file ClassCRFCriterion using checkgrad

require 'nn'
require 'optim'
require 'ClassCRFCriterion'

torch.manualSeed(1)

nState = 4
target = torch.DoubleTensor{1, 2, 4, 3}
nNode = (#target)[1]

criterion = nn.ClassCRFCriterion(nState, nNode)

-- Here x must be a vector representing the output of the LinearCRF layer 
-- (i.e. input of ClassCRFCriterion)
function obj(x)
	
	-- first reshape the leading nNode*nState elements of x into a matrix sized nNode-by-nState
	-- No physical memory copy is done by set()
	-- This is the "outNode" part of the output in LinearCRF:updateOutput
	outNode = torch.DoubleTensor():set(x:storage(), 1, torch.LongStorage{nNode, nState})
	
	-- Next reshape the remaining elements of x into a matrix sized nState-by-nState
	-- This is the wEdge part of the output in LinearCRF:updateOutput
	wEdge = torch.DoubleTensor():set(x:storage(), nNode*nState+1, torch.LongStorage{nState, nState})
	
	local input = {outNode=outNode, wEdge=wEdge}
	f = criterion:forward(input, target)
	g = criterion:backward(input, target)
	
	-- The gradient of ClassCRFCriterion has two parts: gNode and gEdge
	-- See the comment above ClassCRFCriterion:updateGradInput
	g = torch.cat(g.gNode, g.gEdge, 1):resizeAs(x)
	return f,g
end

x = torch.randn((nNode+nState)*nState)
eps = 1e-6
diff,dC,dC_est = optim.checkgrad(obj, x, eps)

print('diff = ' .. diff)
