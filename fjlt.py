import numpy as np
from scipy.fft import dct
from sklearn.random_projection import GaussianRandomProjection
from scipy import sparse

def sjlt(A, k):
    """
    Yin et al. 2020 ESE: Extremely Sparse JL Transform.
    Implemented for scipy.sparse.csc_matrix representation.
    """
    d = A.shape[1]
    h = np.random.choice(d, size=k)
    sigma = np.random.choice([-1, 1], size=d)
    R = sparse.lil_matrix((d, k))
    for j in range(k):
        R[h[j], j] = sigma[h[j]]
    R = R.tocsc()
    A_sjlt = (A.dot(R)).multiply(np.sqrt(d / k))
    return A_sjlt

def fjlt(A, k):
    """
    A variant of FJLT. Embed each row of matrix A with FJLT. See the following resources:
        - The review section (page 3) of https://arxiv.org/abs/1909.04801
        - Page 1 of https://www.sketchingbigdata.org/fall17/lec/lec9.pdf
    
    Note:
        I name it sfd because the matrices are called S(ample), F(ourier transform), D(iagonal).
    """
    A = A.T
    d = A.shape[0]
    sign_vector = np.random.randint(0, 2, size=(d, 1)) * 2 - 1
    idx = np.zeros(k, dtype=int)
    idx[1:] = np.random.choice(d - 1, k - 1, replace=False) + 1
    DA = sign_vector * A
    FDA = dct(DA, axis=0, norm='ortho')
    A_embedded = np.sqrt(d / k) * FDA[idx]
    return A_embedded.T

def gaussian_random_projection(A, k):
    """
    Gaussian random projection from sklearn.
    """
    transformer = GaussianRandomProjection(n_components=k)
    A_embedded = transformer.fit_transform(A)
    return A_embedded
