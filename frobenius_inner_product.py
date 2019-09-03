import torch
import numpy as np
import scipy.sparse
from torch import from_numpy
import operator

# Create your sparse tensor here

# Tensor 1
indexA = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]])
valueA = torch.Tensor([1, 2, 3, 4, 5])

# Tensor 2
indexB = torch.tensor([[0, 2], [1, 0]])
valueB = torch.Tensor([2, 4])

print('Tensor 1')
print(torch.sparse_coo_tensor(indexA, valueA, [3,3]).to_dense())
print('Tensor 2')
print(torch.sparse_coo_tensor(indexB, valueB, [3,2]).to_dense())


# creates triples from a sparse tensor
def create_triple(sparse_coo_tensor):
	indicesA = sparse_coo_tensor.coalesce().indices()
	npindicesA=indicesA.numpy()
	valsA = sparse_coo_tensor.coalesce().values()
	npvalsA=valsA.numpy()

	print('npindices: ', npindicesA)
	print('npvals', npvalsA)

#itertools.izip ???
	triples_a = [[i,v] for i,v in zip(npindicesA.T,npvalsA)]
	# print('triples: ', triples_a) 
	#x:  [[array([0, 1]), 1.0], [array([0, 2]), 2.0], [array([1, 0]), 3.0], [array([2, 0]), 4.0], [array([2, 1]), 5.0]]

	# sort list by first elem of indices
	sorted(triples_a, key=lambda x: (x[0][0], x[0][1]))
	# print('sorted_triples: ', triples_a)
	return triples_a

# takes in 4 tensors and 2 ints
def frobenius_vals(indexA, valueA, indexB, valueB, ydim, xdim):
# def frobenius_vals(tensor1,tensor2):
	triples_a = create_triple(torch.sparse_coo_tensor(indexA, valueA, [ydim,xdim]))
	triples_b = create_triple(torch.sparse_coo_tensor(indexB, valueB, [ydim,xdim]))
	print('triples_a ', triples_a)
	print('triples_b ',triples_b)
	frobenius=[]
	# need two pointer method
	i=0
	j=0
	max_pointer=min(len(triples_a),len(triples_b))
	while (i < len(triples_a) and j < len(triples_b)):
		# print('len triples a: ',len(triples_a))
		# print('len triples b: ',len(triples_b))
		# print('i: ',i)
		# print('j: ',j)
		#checking first elem of triples (first elem is an array of 2 elems)
		if np.array_equal(triples_a[i][0],triples_b[j][0]):
			# can multiply sparse elems
			product = triples_a[i][1]*triples_b[j][1]
			frobenius.append([triples_a[i][0],product])
			print('added')
			i+=1
			j+=1
		# otherwise need to move a pointer
		else:
			# check 1st elems of numpy arrays
			if (operator.eq(triples_a[i][0][0],triples_b[j][0][0])):
				# compare2 = cmp(triples_a[i][0][1],triples_b[j][0][1])
				if (operator.lt(triples_a[i][0][1],triples_b[j][0][1])):
					i+=1
				else:
					j+=1
			else:
				if (operator.lt(triples_a[i][0][0],triples_b[j][0][0])):
					i+=1
				else:
					j+=1
	return frobenius
import time
start=time.time()
frob_array = frobenius_vals(indexA, valueA, indexB, valueB, 3, 3)



ind1=[]
ind2=[]
vals=[]
for x, y in frob_array:
	
	print(x[1])
	ind1.append(x[0])
	ind2.append(x[1])
	vals.append(y)

p=[(x[0], x[1], y) for x, y in frob_array]

# print([(y, x[0], x[1]) for x, y in frob_array])

# NEEEDS MORE WORK HERE
# lsts = [list(a) for a in zip(p[0])]
# print('lsts', [[list(a) for a in zip(x)] for x in p])

# print('p',p)

indexR = torch.tensor([ind1, ind2])

# indexR = torch.tensor([ind1, ind2])
valueR = torch.Tensor(vals)

print(torch.sparse_coo_tensor(indexR, valueR, [3,3]).to_dense())



end=time.time()
print('time: ', end-start)

# compare w group sparse product

# print('frob_array: ',frob_array)
# print('sum: ',sum([x[1] for x in frob_array]))

# import pdb;pdb.set_trace();

# indicesX=[x[0][0][0] for x,y in frob_array]
# indicesY=[x[0][0][1] for x,y in frob_array]

# vals=[y for x,y in frob_array]

# finalIndex = torch.tensor(indicesX,indicesY)
# finalValue = torch.tensor(vals)

# print(torch.sparse_coo_tensor(finalIndex, finalValue, [3,3]).to_dense())


# def cmp(a, b):
#     return (a > b) - (a < b) 
    
# testing purposes

# a = indexA.numpy()
# print('a', a)
# # a [[0 0 1 2 2]
# #   [1 2 0 0 1]]

# b = indexB.numpy()
# print('b: ', b)
# # b:  [[0 2]
# #      [1 0]]

# # assume a, b have same length
# # al=list(zip(a, b))
# # print('al', list(zip(a, b)))

# # [[0 1]
# #  [0 2]
# #  [1 0]
# #  [2 0]
# #  [2 1]]
# print(a.T.tolist().sort(key=lambda x: x[0]))
# print('a',a)

# print(b.T.tolist().sort(key=lambda x: x[0]))
# print('b',b)

# print('sortedA',sortedA)
# print(a)




# pntsa = [a for (a,b) in sorted(zip(a, b), key=operator.itemgetter(0))]
# print(pntsa)
# sorted(zip(ds, pnts), key=operator.itemgetter(0))


# *************************** second test ***************************
# print('Second Test')
# indices1 = torch.LongTensor([[0, 0, 1], [0, 1, 1]]) # numpy.int
# values1 = torch.FloatTensor([2, 3, 4]) # numpy.float32
# sizes = [2, 2]
# print(torch.sparse_coo_tensor(indices1, values1, sizes).to_dense())

# indices2 = torch.LongTensor([[0, 0, 1], [0, 1, 1]]) # numpy.int
# values2 = torch.FloatTensor([2, 3, 4]) # numpy.float32
# print(torch.sparse_coo_tensor(indices2, values2, sizes).to_dense())

# index3, value3 = sparse_sparse_mult(indices1, values1, indices2, values2, 2, 2, 2)

# print(torch.sparse_coo_tensor(index3, value3, [2,2]).to_dense())
