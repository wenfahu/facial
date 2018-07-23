import numpy as np
import faiss
import torch


def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        if x.is_cuda:
            D = torch.cuda.FloatTensor(n, k)
        else:
            D = torch.FloatTensor(n, k)
    else:
        assert D.__class__ in (torch.FloatTensor, torch.cuda.FloatTensor)
        assert D.size() == (n, k)
        assert D.is_contiguous()

    if I is None:
        if x.is_cuda:
            I = torch.cuda.LongTensor(n, k)
        else:
            I = torch.LongTensor(n, k)
    else:
        assert I.__class__ in (torch.LongTensor, torch.cuda.LongTensor)
        assert I.size() == (n, k)
        assert I.is_contiguous()
    torch.cuda.synchronize()
    xptr = x.storage().data_ptr()
    Iptr = I.storage().data_ptr()
    Dptr = D.storage().data_ptr()
    index.search_c(n, faiss.cast_integer_to_float_ptr(xptr),
                   k, faiss.cast_integer_to_float_ptr(Dptr),
                   faiss.cast_integer_to_long_ptr(Iptr))
    torch.cuda.synchronize()
    return D, I
