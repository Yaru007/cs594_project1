---First layer model that performs linear transformation
local LinearCRF, parent = torch.class('nn.LinearCRF', 'nn.Module')

function LinearCRF:__init(nState, nFea, data)
	parent.__init(self)
	self.nFea = nFea
	self.nState = nState
	self.weight = torch.Tensor(nFea+nState, nState)
	self.gradWeight = self.weight:clone()
	self.data = data
	self:reset()
end

function LinearCRF:reset()	
  self.weight:zero()
  return self
end

-- compute W*x, where x is a sparse vector
local function spprod(self, W, x)
	res = torch.zeros(self.nState)
	for j = 1, x[1]:size(1) do
		idx = x[1][j]
		val = x[2][j]
		w = W[idx]
		for k = 1,self.nState do
			res[k] = res[k] + w[k]*val
		end
	end
	return res
end


-- Compute the output X^t * (w_1, ..., w_26) and copy T
-- input is an integer array specifying the id of the letters (belonging to the current word)
function LinearCRF:updateOutput(input)  
	local nState = self.nState
	local nFea = self.nFea
	
    local wNode = self.weight:sub(1, nFea)
	local wEdge = self.weight:sub(nFea+1, nFea+nState)

	local nLetter = (#input)[1]
	outNode = torch.DoubleTensor(nLetter, nState)	-- you may pre-allocate persistent space for it
	for i = 1,nLetter do
		outNode[i]:copy(spprod(self, wNode, self.data[input[i]][2]))
	end

  return {outNode=outNode, wEdge=wEdge}
end


-- The gradient with respect to the input.
-- No need to implement because it is never needed.
function LinearCRF:updateGradInput(input, gradOutput)
	return self.gradInput
end


-- Compute the gradient with respect to the weights in this layer
-- 	input: is an array specifying the letter id (same as LinearCRF:updateOutput)
-- 	gradOutput: the gradient given by ClassCRFCriterion:updateGradInput
--   		It is a table {gNode, gEdge}  with two parts:
--  		1. gradOutput.gNode is the C^t matrix, sized #letter-by-nState
--  		2. gradOutput.gEdge is the gradient in T, sized nState-by-nState
-- Output:
--   No explicit output. Record the gradient in self.gradWeight.
--   It is the vectorization of the concatenation of two matrices
--   One is sized #letter-by-nState, and the other is nState-by-nState
function LinearCRF:accGradParameters(input, gradOutput)
-- gradient of logp respect to T

   local C_train = gradOutput.gNode
   local T_train = gradOutput.gEdge


-- gradient of logp respect to w in (13)
   local nLetter = (#input)[1]
   local nFea = self.nFea
   local nState = self.nState

   local G_train = torch.zeros(nFea,nState)

   for j = 1, nState do
       for k = 1, nLetter do
           local x = self.data[input[k]][2]
           for i = 1, x[1]:size(1) do
               ind = x[1][i]
               val = x[2][i]
               G_train[ind][j] = G_train[ind][j] + val*C_train[k][j]
           end
       end
   end



-- gradient of logp respect to T is the T_train
   self.gradWeight:sub(1,nFea):add(G_train)
   self.gradWeight:sub(nFea+1,nFea+nState):add(T_train)
end


-- we do not need to accumulate parameters when sharing
LinearCRF.sharedAccUpdateGradParameters = LinearCRF.accUpdateGradParameters


function LinearCRF:clearState()   
   return parent.clearState(self)
end

function LinearCRF:__tostring__()
  return torch.type(self) ..
      string.format('(#fea = %d, #state = %d)', self.nFea, self.nState) ..
      (self.bias == nil and ' without bias' or '')
end
