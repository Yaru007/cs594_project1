--[[ A plain implementation of SGD
ARGS:
- `opfunc` : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- `x`      : the initial point
- `config` : a table with configuration parameters for the optimizer
- `config.learningRate`      : learning rate
- `config.learningRateDecay` : learning rate decay
- `config.weightDecay`       : weight decay
- `config.weightDecays`      : vector of individual weight decays
- `config.momentum`          : momentum
- `config.dampening`         : dampening for momentum
- `config.nesterov`          : enables Nesterov momentum
- `config.learningRates`     : vector of individual learning rates
- `state`  : a table describing the state of the optimizer; after each
             call the state is modified
- `state.evalCounter`        : evaluation counter (optional: 0, by default)
RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update
(Clement Farabet, 2012)
]]
function sgd(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local isverbose = config.verbose or false
   local tolFun = config.tolFun or 1e-5
   local maxIter = tonumber(config.maxIter) or 20
   local tolX = config.tolX or 1e-9



   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter
   local monitor = optimState.monitor

   local start_time = sys.clock()
   local monitor = optimState.monitor
   
   local verbose
   if isverbose then
      verbose = function(...) print('<optim.lbfgs> ', ...) end
   else
      verbose = function() end
   end

   -- import some functions
   local abs = math.abs
   local min = math.min

   -- evaluate initial f(x) and df/dx
   f,g = opfunc(x)
   x:add(-lr,g)

   state.evalCounter = 1
   local f_hist = {f}
  
   -- check optimality of initial point
   state.tmp1 = state.tmp1 or g.new(g:size()):zero(); local tmp1 = state.tmp1
   tmp1:copy(g):abs()
   gtol = tmp1:sum()

   if tmp1:sum() <= tolFun then
      -- optimality condition below tolFun
      verbose('optimality condition below tolFun')
      return x,f_hist
   end

   local nIter = 0
   while nIter < maxIter do
      f_old = f
      nIter = nIter + 1
      local f,g = opfunc(x)
      x:add(-lr,g)
      table.insert(f_hist, f) 
      state.evalCounter = state.evalCounter + 1

      tmp1:copy(g):abs()
      gtol = tmp1:sum()
      
      io.write(string.format("%d %.4f %.4f %d ", nIter-1, f, gtol, sys.clock()-start_time))
      if monitor then monitor(x) end
      print('')


       ------------------------------------------------------------
      -- check conditions
      ------------------------------------------------------------
      if nIter == maxIter then
         -- no use to run tests
         verbose('reached max number of iterations')
         break
      end


      if tmp1:sum() <= tolFun then
         -- check optimality
         verbose('optimality condition below tolFun')
         break
      end

      if abs(f-f_old) < tolX then
         -- function value changing less than tolX
         verbose('function value changing less than tolX')
         break
      end
   end

   return x,f_hist
end

