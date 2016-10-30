-- Test the module of LinearCRF

require 'nn'
require 'optim'
require 'LinearCRF'

torch.manualSeed(1)


local train = torch.load('../data/debug_torch.dat','ascii')

data = train.data
local nFea = train.nFea
local nState = train.nState

model = nn.Sequential()
model:add(nn.LinearCRF(nState, nFea, data))

parameters,gradParameters = model:getParameters()
print (parameter, gradParameter)

-- Take all the five letters pretending that they form a word
local input = torch.DoubleTensor{1,2,3,4,5}

-- Since the output of this layer is not a scalar,
-- we multiply its output with two constant weights gNode and gNode
-- so that we do finally get a scalar for using checkgrad
gNode = torch.randn((#input)[1], nState)
gEdge = torch.randn(nState, nState)
grad_from_CRFCriterion = {gNode=gNode, gEdge=gEdge}

-- Here x is the weight of the LinearCRF layer
function obj(x)
	parameters:copy(x)
	gradParameters:zero()
	
	output = model:forward(input)
  local f = torch.dot(output.outNode, gNode) + torch.dot(output.wEdge, gEdge)
  
  -- estimate df/dW
  model:backward(input, grad_from_CRFCriterion)
	return f, gradParameters
end

-- x is a vectorization of all weights in the LinearCRF layer
x = torch.randn((nFea+nState)*nState)
eps = 1e-6
diff,dC,dC_est = optim.checkgrad(obj, x, eps)

print('diff = ' .. diff)
