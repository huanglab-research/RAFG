import torch

h=[]
x1 = torch.rand(2,4,4)
x2 = torch.rand(2,4,4)
h.append(x1)
h.append(x2)
dic = {}
for batchh in range(0,2):
    s = []
    for column1 in range(0,3):
        for row1 in range(0,3):
            summed_value = h[1][batchh,column1:column1+2,row1:row1+2].sum(dim=(0, 1))
            # print(f"66666新的试一下{summed_value.shape}")
            s.append([column1,row1,summed_value])
        sorted_s = sorted(s, key=lambda x: x[2], reverse=True)
    print(sorted_s)
    dic[batchh] = s
    
print(f"222````````新的试一下{len(dic)}")
# for stratx in range(0,3120):