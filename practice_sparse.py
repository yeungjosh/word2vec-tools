import torch
import numpy as np
import scipy.sparse
# import torch
from torch import from_numpy
# from torch_sparse import transpose, to_scipy, from_scipy, coalesce

def spspmm(indexA, valueA, indexB, valueB, m, k, n, coalesced=False):
    """ Matrix product of two sparse tensors
    Args:
        indexA : The index LongTensor of first sparse matrix.
        valueA : The value Tensor of first sparse matrix.
        indexB : The index LongTensor of second sparse matrix.
        valueB : The value Tensor of second sparse matrix.
        m (int): The first dimension of first dense matrix
        k (int): The shared second dimension of first dense matrix and
            first dimension of second dense matrix
        n (int): The second dimension of second dense matrix.

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    # index, value = SpSpMM.apply(indexA, valueA, indexB, valueB, m, k, n)
    index, value = mm(indexA, valueA, indexB, valueB, m, k, n)
    return index.detach(), value

# class SpSpMM(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, indexA, valueA, indexB, valueB, m, k, n):
#         indexC, valueC = mm(indexA, valueA, indexB, valueB, m, k, n)
#         ctx.m, ctx.k, ctx.n = m, k, n
#         ctx.save_for_backward(indexA, valueA, indexB, valueB, indexC)
#         return indexC, valueC


def mm(indexA, valueA, indexB, valueB, m, k, n):
    assert valueA.dtype == valueB.dtype

    # if indexA.is_cuda:
    #     return torch_sparse.spspmm_cuda.spspmm(indexA, valueA, indexB, valueB,
                                               # m, k, n)
    A = to_scipy(indexA, valueA, m, k)
    B = to_scipy(indexB, valueB, k, n)
    C = A.dot(B).tocoo().tocsr().tocoo()  # Force scipy to coalesce
    indexC, valueC = from_scipy(C)
    return indexC.detach(), valueC



import numpy as np
import scipy.sparse
from torch import from_numpy

def to_scipy(index, value, m, n):
    assert not index.is_cuda and not value.is_cuda
    (row, col), data = index.detach(), value.detach()
    return scipy.sparse.coo_matrix((data, (row, col)), (m, n))

def from_scipy(A):
    A = A.tocoo()
    row, col, value = A.row.astype(np.int64), A.col.astype(np.int64), A.data
    row, col, value = from_numpy(row), from_numpy(col), from_numpy(value)
    index = torch.stack([row, col], dim=0)
    return index, value
   # import torch
# from torch_sparse import spspmm

indexA = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]])
valueA = torch.Tensor([1, 2, 3, 4, 5])

indexB = torch.tensor([[0, 2], [1, 0]])
valueB = torch.Tensor([2, 4])

print(torch.sparse_coo_tensor(indexA, valueA, [3,3]).to_dense())

print(torch.sparse_coo_tensor(indexB, valueB, [3,2]).to_dense())

indexC, valueC = spspmm(indexA, valueA, indexB, valueB, 3, 3, 2)

# print(indexC)
# print(valueC)
sizes=[3,2]
print('here we go')
print(torch.sparse_coo_tensor(indexC, valueC, sizes).to_dense())


# *************************** second ***************************
print('second')
indices1 = torch.LongTensor([[0, 0, 1], [0, 1, 1]]) # numpy.int
values1 = torch.FloatTensor([2, 3, 4]) # numpy.float32
sizes = [2, 2]
print(torch.sparse_coo_tensor(indices1, values1, sizes))

indices2 = torch.LongTensor([[0, 0, 1], [0, 1, 1]]) # numpy.int
values2 = torch.FloatTensor([2, 3, 4]) # numpy.float32

indexC, valueC = spspmm(indexA, valueA, indexB, valueB, 3, 3, 2)




