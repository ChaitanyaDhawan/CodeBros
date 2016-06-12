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
	k = 144444441414k:transpose(1,2)
	local a = torch.cmul(k,mat)
	local Y = torch.floor(torch.sum(a) / torch.sum(mat))
	--local m = torch.Tensor({X-1,Y-1,(mat:size(1)-X),(mat:size(2)-Y)})
	--local min = torch.min(m,1):sum()		
	return X,Y
	--(image.crop(mat,X-min,Y-min,X+min-1,Y+min-1))
end

function M.norm(img)
	i = image.load(img,1,'double')
	i = 255 * i
	min = torch.min(i)
	max = torch.max(i)
	i = (i - min)*(255/(max - min))
	i = image.scale(i,20,20,simple)
		--i = 255 - 255 * i
		if i[1][2]>150 then i = 255 -i end
		X,Y = centroid(i)
		local c=1
		s = i:storage()
		x = 15 -X
		y = 15-Y
		t = torch.Tensor(28,28):fill(0)
		--if(s[1]>150) then flag =1
		--else flag =0 end
		
		--if(flag == 1) then s = 255 - s end 
		for p=x,x+19 do
			for q=y,y+19 do
				t[p][q] = s[c]
				if (s[c]<80 ) then t[p][q]=0 
				elseif (s[c]<200) then t[p][q] = t[p][q] + 40 end
				c = c+1
			end
		end
	
	return t
end 
	

function M.f(img)
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
					if (s[c]<100 ) then t[p][q]=0 
				        elseif (s[c]<200) then t[p][q] = t[p][q] + 40 end	
					--if(s[c]>50 and s[c]<200) then
						--t[p][q]= t[p][q]+30
					--end	
					c = c+1
				end
			end	
            _ , out = torch.max(model1:forward(t),1)
			return out-1
		end
function M.g(i)
	--i = image.load(img,1,'double')
	--i = 255 * i
	min = torch.min(i)
	max = torch.max(i)
	i = (i - min)*(255/(max - min))
	i = image.scale(i,20,20,simple)
		--i = 255 - 255 * i
		--if i[1][2]>150 then i = 255 -i end
		X,Y = centroid(i)
		local c=1
		s = i:storage()
		x = 15 -X
		y = 15-Y
		t = torch.Tensor(28,28):fill(0)
		--if(s[1]>150) then flag =1
		--else flag =0 end
		
		--if(flag == 1) then s = 255 - s end 
		for p=x,x+19 do
			for q=y,y+19 do
				t[p][q] = s[c]
				if (s[c]<80 ) then t[p][q]=0 
				elseif (s[c]<200) then t[p][q] = t[p][q] + 40 end
				c = c+1
			end
		end
	
	return torch.floor(t)
end 
	

function M.segment(img)
	p=1
	flag=0
	t = {}
	count = 1
	img=image.load(img,1,'double')
	img = 255 - (255 * img)
	while(p<img:size(2)) do
        ::sos::
		--column boundary
		for j=p,img:size(2) do
			if(j==img:size(2)) then flag=1 end
			if(torch.max(img[{{},j}])>100) then
				if(j==1) then c1=j
				else c1 = j-1 end
                print(c1)
				break
			end
		end
		
		if(flag==1) then break end 
		
		for j=c1+1,img:size(2) do
			p=j
			if(torch.max(img[{{},j}])<100) then
				if(j==img:size(2)) then c2=j
				else c2 = j+1 end
				break
			end
		end					 
	
        
		--row boundary
        
        asd = 1
        cq = 1
        ::sis::
		for i=asd,img:size(1) do
			if(torch.max(img[{i,{c1,c2}}])>100) then
				if(i==1) then r1=i
				else r1 = i-1 end
				break
			end
		end
		for i=r1+1,img:size(1) do
            asd = i
			if(torch.max(img[{i,{c1,c2}}])<100) then
				if(i==img:size(1)) then r2=i
				else r2 = i+1 end
				break
			end
		end
        cq = cq+1
        if(cq>10) then goto lol end
        if((r2-r1)/img:size(1)<0.1) then goto sis end
        ::lol::
		a = {c1-1,r1-1,c2-1,r2-1}
        if((c2-c1)*(r2-r1)/(img:size(2)*img:size(1))<0.001) then goto sos end
		t[count]=a
		count = count +1
	end
    
	for q=1,count-1 do 
		im = image.crop(img,t[q][1],t[q][2],t[q][3],t[q][4])
		min = torch.min(im)
		max = torch.max(im)
		im = (im - min)*(255/(max - min))
		im = image.scale(im,20,20,simple)
			--i = 255 - 255 * i
			--if i[1][2]>150 then i = 255 -i end
			X,Y = centroid(im)
			local c=1
			s = im:storage()
			x = 15 -X
			y = 15-Y
			f = torch.Tensor(28,28):fill(0)
			--if(s[1]>150) then flag =1
			--else flag =0 end
		
			--if(flag == 1) then s = 255 - s end 
			for p=x,x+19 do
				for q=y,y+19 do
					f[p][q] = s[c]
					if (s[c]<80 ) then f[p][q]=0 
					elseif (s[c]<200) then f[p][q] = f[p][q] + 40 end
					c = c+1
				end
			end
		print(itorch.image(torch.floor(f)))
		print(model1:forward(f))
	end
	
	return t
end		

return M
		
		
