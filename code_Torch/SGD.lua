--[[ A plain implementation of SGD
ARGS:
- `opfunc` : a function that takes a single input (X), the point
of a evaluation, and returns f(X) and df/dX
- `x`      : the initial point
- `config` : a table with configuration parameters for the optimizer
- `config.learningRate`      : learning rate
- `state`  : a table describing the state of the optimizer; after each
call the state is modified
- `state.evalCounter`        : evaluation counter (optional: 0, by default)
RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update
]]
function sgd(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local learningRate = config.learningRate or 1e-3
   local monitor = optimState.monitor
   local maxIter = tonumber(config.maxIter) or 20
   local maxEval = tonumber(config.maxEval) or maxIter*1.25
   local tolFun = config.tolFun or 1e-5
   local tolX = config.tolX or 1e-9
   local isverbose = config.verbose or false
   local sample = config.sampler
   local fevalIntervel = config.fevalIntervel
   local lambda = config.lambda

   state.updateCount = state.updateCount or 0
   state.funcEval = state.funcEval or 0
   state.nIter = state.nIter or 0
   
   local start_time = sys.clock()

   -- import some functions
   local abs = math.abs
   local fmod = math.fmod

   if isverbose then
      verbose = function(...) print('<vanilla SGD> ', ...) end
   else
      verbose = function() end
   end

   -- (1) evaluate f(x) and df/dx
   local fx,g = opfunc(x)
   local f_old = fx

   -- check optimality of initial point
   state.tmp1 = state.tmp1 or g.new(g:size()):zero(); 
   local tmp1 = state.tmp1
   tmp1:copy(g):abs()
   gtol = tmp1:sum()
   if tmp1:sum() <= tolFun then
      -- optimality condition below tolFun
      verbose('optimality condition below tolFun')
      return x,{fx}
   end

   local f_old = state.f_old
   local nIter = 0
   local x_old = torch.zeros(x:size())

   while nIter < maxIter do
      if fmod(state.updateCount, fevalIntervel) == 0 then
         f_old = fx
         fx,g = opfunc(x)

         nIter = nIter + 1
         state.nIter = state.nIter + 1
         state.funcEval = state.funcEval + 1
         io.write(string.format("%d %.4f %.4f %.3f %d ", nIter-1, fx, gtol, sys.clock()-start_time, state.funcEval))
         if monitor then monitor(x) end
         print('')
      end
      
      i, fi,gi = sample(x)

      -- parameter update with single or individual learning rates
      if learningRate then
         gi:add(-lambda*x_old)
         x_old = x
         state.updateCount = state.updateCount + 1
         x:add(-learningRate, gi)
      end

      ------------------------------------------------------------
      -- check conditions
      ------------------------------------------------------------
      if nIter == maxIter then
         -- no use to run tests
         verbose('reached max number of iterations')
         break
      end
      
      tmp1:copy(g):abs()
      gtol = tmp1:sum()
      if tmp1:sum() <= tolFun then
         -- check optimality
         verbose('optimality condition below tolFun')
         break
      end

      if fmod(state.funcEval, fevalIntervel) == 0 and abs(fx-f_old) < tolX then
         -- function value changing less than tolX
         verbose('function value changing less than tolX')
         break
      end

   end

   -- return x*, f(x) before optimization
   return x,{fx}
end