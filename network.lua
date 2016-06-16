require 'torch'
require 'nn'
require 'optim'
local chars1 = torch.load('chars1')
local fullset = chars1.trainset
fullset.size = 2860
 sfullset = {}
local sdata = {}
local slabel = {}
--testset = mnist.testdataset()
local sh = torch.randperm(fullset.size)
for i=1,fullset.size do
    sdata[i]=fullset.data[sh[i]]
    slabel[i] = fullset.label[sh[i]]
end
sfullset.data = sdata
sfullset.label=slabel
sfullset.size = fullset.size

qwerty = {}
abcdef = {}
qwerty1 = {}
abcdef1 = {}
for z=1,2500 do
    qwerty[z]=sfullset.data[z]:double()
    abcdef[z]=sfullset.label[z]
    end
for z=2501,2860 do
    qwerty1[z-2500]=sfullset.data[z]:double()
    abcdef1[z-2500]=sfullset.label[z]
    end


trainset = {
    size = 2500,
    data = qwerty,
    label = abcdef
}

validationset = {
    size = 360,
    data = qwerty1,
    label = abcdef1
    }

model = nn.Sequential()
model:add(nn.Reshape(1,28,28))
model:add(nn.SpatialConvolution(1,20,5,5))
model:add(nn.SpatialMaxPooling(2,2))

model:add(nn.SpatialConvolution(20,40,5,5))
model:add(nn.SpatialMaxPooling(2,2))

model:add(nn.Reshape(40*4*4))
model:add(nn.Linear(40*4*4,100))
model:add(nn.Tanh())
model:add(nn.Dropout(0.5))

model:add(nn.Linear(100, 26))
model:add(nn.LogSoftMax())

--print("ok1")
criterion = nn.ClassNLLCriterion()

sgd_params = {
   learningRate = 8*(1e-3),
   learningRateDecay = 1e-6
   --weightDecay = 1e-3,
   --momentum = 1e-4
}
    la={['a']=1,['b']=2,['c']=3,['d']=4,['e']=5,['f']=6,['g']=7,['h']=8,['i']=9,['j']=10,['k']=11,['l']=12,['m']=13,['n']=14,['o']=15,['p']=16,['q']=17,['r']=18,['s']=19,['t']=20,['u']=21,['v']=22,['w']=23,['x']=24,['y']=25,['z']=26}
x, dl_dx = model:getParameters()
--print("ok2")
step = function(batch_size)
    local current_loss = 0
    local count = 0
    local shuffle = torch.randperm(trainset.size)
    batch_size = batch_size or 200
    --la={['a']=1,['b']=2,['c']=3,['d']=4,['e']=5,['f']=6,['g']=7,['h']=8,['i']=9,['j']=10,['k']=11,['l']=12,['m']=13,['n']=14,['o']=15,['p']=16,['q']=17,['r']=18,['s']=19,['t']=20,['u']=21,['v']=22,['w']=23,['x']=24,['y']=25,['z']=26}

    for t = 1,trainset.size,batch_size do
        -- setup inputs and targets for this mini-batch
        local size = math.min(t + batch_size, trainset.size) - t
        local inputs = torch.Tensor(size, 28, 28)
        local targets = torch.Tensor(size)
        --print(la[trainset.label['c']])
        for i = 1,size do
            local input = trainset.data[shuffle[i+t]]
            --print(input)
            local target = la[trainset.label[shuffle[i+t]]]
          --  print(target)
            -- if target == 0 then target = 10 end
            inputs[i] = input
            targets[i] = target
        end
        --print("ok4")
        --targets = la[targets]
        local feval = function(x_new)
            -- reset data
            if x ~= x_new then x:copy(x_new) end
            dl_dx:zero()

            -- perform mini-batch gradient descent
            local loss = criterion:forward(model:forward(inputs), targets)
            model:backward(inputs, criterion:backward(model.output, targets))

            return loss, dl_dx
        end
        
        _, fs = optim.sgd(feval, x, sgd_params)
        -- fs is a table containing value of the loss function
        -- (just 1 value for the SGD optimization)
        count = count + 1
        current_loss = current_loss + fs[1]
    end

    -- normalize loss
    return current_loss / count
end
--[[eval = function(dataset, batch_size)
    local count = 0
    local batch_size = batch_size or 200
    
    for i = 1,dataset.size,batch_size do
        local size = math.min(i + batch_size , dataset.size) - i
        --print(size)
        local inputs = torch.Tensor(size,28,28):fill(0)
        local targets = torch.Tensor(size):fill(0):long()
        for q=i,i+size-1 do
        inputs[q-i+1] = dataset.data[q]
        print("label is", trainset.label[q])
        targets[q-i+1] = la[dataset.label[q]:long()
        --print(targets)
        end    
        --print(inputs,targets)
        local outputs = model:forward(inputs)
        local _, indices = torch.max(outputs, 2)
        local guessed_right = indices:eq(targets):sum()
        count = count + guessed_right
    end

    return count / dataset.size
end ]]

eval = function(dataset)
        local size = dataset.size
        local inputs = torch.Tensor(size,28,28):fill(0)
        local targets = torch.Tensor(size):fill(0):long()
        --print(dataset.data)
        for i=1,dataset.size do
          --  local inputs = torch.Tensor(size,28,28):fill(0)
            --local targets = torch.Tensor(size):fill(0):long()
            inputs[i] = dataset.data[i]
        --print("label is", trainset.label[q])
            targets[i] = la[dataset.label[i]]
        end 
        local outputs = model:forward(inputs)
        local _,indices = torch.max(outputs,2)
        local guessed_right = indices:eq(targets):sum()
        return guessed_right / size
   end 

max_iters = 100

do
    local last_accuracy = 0
    local decreasing = 0
    local threshold = 1 -- how many deacreasing epochs we allow
    for i = 1,max_iters do
        local loss = step()
        print(string.format('Epoch: %d Current loss: %4f', i, loss))
        local accuracy = eval(validationset)
        print(string.format('Accuracy on the validation set: %4f', accuracy))
        if accuracy < last_accuracy then
            if decreasing > threshold then break end
            decreasing = decreasing + 1
        else
            decreasing = 0
        end
        last_accuracy = accuracy
    end
end

--testset.data = testset.data:double()

--[[function f(str)
local i = image.load(str,1,'double')
i = image.scale(i,28,28,simple)
i = i*255
out = model:forward(i)
return out
end
function g(str)
local i = image.load(str,1,'double')
i = image.scale(i,28,28,simple)
i = 255-i*255

out = model:forward(i)
return out
end]]--
torch.save('alphaModel.net',model)
