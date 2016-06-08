require 'torch'
require 'nn'
require 'optim'
mnist = require 'mnist'

fullset = mnist.traindataset()
testset = mnist.testdataset()



trainset = {
    size = 50000,
    data = fullset.data[{{1,50000}}]:double(),
    label = fullset.label[{{1,50000}}]
}

validationset = {
    size = 10000,
    data = fullset.data[{{50001,60000}}]:double(),
    label = fullset.label[{{50001,60000}}]
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

model:add(nn.Linear(100, 10))
model:add(nn.LogSoftMax())


criterion = nn.ClassNLLCriterion()

sgd_params = {
   learningRate = 5*(1e-3),
   learningRateDecay = 1e-5
   --weightDecay = 1e-3,
   --momentum = 1e-4
}

x, dl_dx = model:getParameters()

step = function(batch_size)
    local current_loss = 0
    local count = 0
    local shuffle = torch.randperm(trainset.size)
    batch_size = batch_size or 200
    
    for t = 1,trainset.size,batch_size do
        -- setup inputs and targets for this mini-batch
        local size = math.min(t + batch_size - 1, trainset.size) - t
        local inputs = torch.Tensor(size, 28, 28)
        local targets = torch.Tensor(size)
        for i = 1,size do
            local input = trainset.data[shuffle[i+t]]
            local target = trainset.label[shuffle[i+t]]
            -- if target == 0 then target = 10 end
            inputs[i] = input
            targets[i] = target
        end
        targets:add(1)
        
        local feval = function(x_new)
            -- reset data
            if x ~= x_new then x:copy(x_new) end
            dl_dx:zero()
            model:training()
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

eval = function(dataset, batch_size)
    local count = 0
    batch_size = batch_size or 200
    
    for i = 1,dataset.size,batch_size do
        local size = math.min(i + batch_size - 1, dataset.size) - i
        local inputs = dataset.data[{{i,i+size-1}}]
        local targets = dataset.label[{{i,i+size-1}}]:long()
        model:evaluate()
        local outputs = model:forward(inputs)
        local _, indices = torch.max(outputs, 2)
        indices:add(-1)
        local guessed_right = indices:eq(targets):sum()
        count = count + guessed_right
    end

    return count / dataset.size
end

max_iters = 30

do
    local last_accuracy = 0
    local decreasing = 0
    local maxacc=0
    local threshold = 1 -- how many deacreasing epochs we allow
    for i = 1,max_iters do
        local loss = step()
        print(string.format('Epoch: %d Current loss: %4f', i, loss))
        local accuracy = (eval(validationset))*100
        print(string.format('Accuracy on the validation set: %4f ', accuracy))
        if maxacc < accuracy then maxacc=accuracy end 
        if accuracy < last_accuracy then
          if decreasing > threshold then break end
          decreasing = decreasing + 1
        else
            decreasing = 0
        end
        last_accuracy = accuracy
    end
    print(maxacc)
end

testset.data = testset.data:double()



local function centroid(mat)
	local k = torch.Tensor(mat:size(1),mat:size(2))
	for i=1,mat:size(1) do
		k[i]:fill(i)
	end
	local a = torch.cmul(k,mat)
	local X = torch.floor(torch.sum(a) / torch.sum(mat))
	k = k:transpose(1,2)
	for i=1,mat:size(2) do
		k[i]:fill(i)
	end
	k = k:transpose(1,2)
	local a = torch.cmul(k,mat)
	local Y = torch.floor(torch.sum(a) / torch.sum(mat))
	--local m = torch.Tensor({X-1,Y-1,(mat:size(1)-X),(mat:size(2)-Y)})
	--local min = torch.min(m,1):sum()		
	return X,Y
	--(image.crop(mat,X-min,Y-min,X+min-1,Y+min-1))
end

function wbg(img)
			require 'image'
			local i =image.load(img,1,'double')
			i = image.scale(i,20,20,simple)
			i = 255 - 255 * i
			X,Y = centroid(i)
			local c=1
			s = i:storage()
			x = 15 -X
			y = 15-Y
			t = torch.Tensor(28,28):fill(0)
			for p=x,x+19 do
				for q=y,y+19 do
					t[p][q] = s[c]
                    if t[p][q] < 100 then t[p][q] = 0 
                    elseif t[p][q] < 200 then t[p][q] = t[p][q] + 40 end
					c = c+1
				end
			end			
            print(itorch.image(t))
			return model:forward(t)
		end
function bbg(img)
			require 'image'
			local i =image.load(img,1,'double')
			i = image.scale(i,20,20,simple)
			i = 255 * i
			X,Y = centroid(i)
			local c=1
			s = i:storage()
			x = 15 -X
			y = 15-Y
			t = torch.Tensor(28,28):fill(0)
			for p=x,x+19 do
				for q=y,y+19 do
					t[p][q] = s[c]
					if t[p][q] < 100 then t[p][q] = 0 
                    elseif t[p][q] < 200 then t[p][q] = t[p][q] + 40 end
            c = c+1
				end
			end			
     print(itorch.image(t))
			return model:forward(t)
		end

eval1 = function(dataset)
   local count = 0
   for i = 1,dataset.size do
      model:evaluate()  
      local output = model:forward(dataset.data[i])
      local _, index = torch.max(output, 1) -- max index
      index=index-1
        local guessed_right = index:eq(dataset.label[i]):sum()
        count = count + guessed_right
        end

   return count / dataset.size
end
