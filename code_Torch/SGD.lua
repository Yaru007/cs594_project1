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

   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter
   local monitor = optimState.monitor
   local f,gradParameters = opfunc(x)

   -- parameter update with designated lr
   x:add(-lr,gradParameters)
   state.evalCounter = state.evalCounter + 1

   return x,f
end

