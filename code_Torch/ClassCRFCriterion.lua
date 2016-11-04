--- Output layer of CRF
local ClassCRFCriterion, parent = torch.class('nn.ClassCRFCriterion', 'nn.Criterion')


function ClassCRFCriterion:__init(nState, maxLen)
	parent.__init(self)
	if nState == nil then
		error('Constructor of ClassCRFCriterion must have at least 1 argument')
	end  
	self.maxLen = maxLen or 100		--max number of letters in all words
	self.nState = nState 			-- number of possible labels (26)
	self.left = torch.zeros(self.maxLen, nState)
	self.right = self.left:clone()  
	self.edge_marg = torch.DoubleTensor(self.maxLen, nState, nState)
	self.node_marg = self.left:clone() 
end


function ClassCRFCriterion:__len()
	return 0
end

-- input: a table which is the union of X^t * (w_1, ..., w_26) and T
-- target: a vector of length m (#letter in the current word). 
--         Each encodes the label of the letter in [1,26]
-- Output: a real number: log p(target | X^t)
function ClassCRFCriterion:updateOutput(input, target)

	local nNode = (#target)[1]	
	local nState = self.nState
	local left = self.left:sub(1, nNode)
	local right = self.right:sub(1, nNode)
	local node_marg = self.node_marg:sub(1,nNode)
	if nNode > 1 then
		edge_marg = self.edge_marg:sub(1,nNode-1)
	end
	local v = torch.DoubleTensor(1, nState)	

	local nodePot = input.outNode
	local edgePot = input.wEdge

	-- Dynamic programming for marginal probability inference starts
	left[1]:zero()
	right[nNode]:zero()
	for i=2, nNode do
		v = (left[i-1] + nodePot[i-1]):resize(nState, 1)
		tmp = edgePot + torch.repeatTensor(v, 1, nState)
		max_tmp = torch.max(tmp, 1)
		left[i]:copy(torch.sum((tmp - torch.repeatTensor(max_tmp, nState, 1)):exp(), 1):log() + max_tmp)
	end

	for i=nNode-1,1,-1 do
		tmp = edgePot + torch.repeatTensor(right[i+1] + nodePot[i+1], nState, 1)
		max_tmp = torch.max(tmp, 2):resize(nState, 1)
		right[i]:copy(torch.sum((tmp - torch.repeatTensor(max_tmp, 1, nState)):exp(), 2):log() + max_tmp)
	end   

	left:add(nodePot)
	right:add(nodePot)
	node_marg:copy(left + right - nodePot)
	local t = node_marg[1]:max()
	local logZ = torch.log((node_marg[1] - t):exp():sum()) + t;

	node_marg:add(torch.repeatTensor(-torch.max(node_marg, 2):resize(nNode,1), 1, nState)):exp()
	node_marg:cdiv(torch.repeatTensor(torch.sum(node_marg,2):resize(nNode,1), 1, nState))

	local f = nodePot[nNode][target[nNode]]
	for i=1,nNode-1 do
		V = edgePot + torch.repeatTensor(left[i]:resize(nState,1), 1, nState) + torch.repeatTensor(right[i+1], nState, 1)
		V:add(-torch.max(V)):exp()
		edge_marg[i]:copy(V / torch.sum(V))
		f = f + nodePot[i][target[i]] + edgePot[target[i]][target[i+1]]	 	
	end

	--	print('log(Z): ' .. logZ)
	--	print('node marginals = ')
	--	print(node_marg)	
	--	print('edge marginals = ')
	--	print(edge_marg)

	self.output = logZ - f
	return self.output
end


-- Compute the gradient with respect to the input
-- input: a table which is the union of X^t * (w_1, ..., w_26) and T
-- The returned gradient needs to be a table with two parts:
--   self.gradInput should be represented as {gNode, gEdge} where
--  1. gNode is the C^t matrix, sized #letter-by-nState
--  2. gEdge is the gradient in T, sized nState-by-nState
function ClassCRFCriterion:updateGradInput(input, target)

	--[[
	Your implementation here
	--]]
	-- self.edge_marg is #letter-by-nState-by-nState, need to accumulate for each letter
	local nNode = (#target)[1]
	local nState = self.nState
	local node_marg = self.node_marg:sub(1,nNode)
	if nNode > 1 then
		local edge_marg = self.edge_marg:sub(1,nNode-1)
	end

	local gEdge = torch.zeros(nState, nState)

	-- subtract indicator
	for i = 1, nNode do
		node_marg[i][target[i]] = node_marg[i][target[i]]-1;
	end

	for i=1,nNode-1 do
		edge_marg[i][target[i]][target[i+1]] = edge_marg[i][target[i]][target[i+1]] - 1;
		gEdge:add(edge_marg[i])
	end
	-- self.node_marg is already computed
	self.gradInput = {gNode = node_marg, gEdge = gEdge}

	return self.gradInput 
end
