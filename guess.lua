 
	local M = {}

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

function M.wbg(img)
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
					c = c+1
				end
			end			
			return model1:forward(t)
		end
function M.bbg(img)
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
					c = c+1
				end
			end			
			return model1:forward(t)
		end
return M
