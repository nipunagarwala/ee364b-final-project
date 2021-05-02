import numpy as np
from sklearn.random_projection import GaussianRandomProjection

def fjlt(A, k):
    """
    A variant of FJLT. See the following resources:
        - The review section (page 3) of https://arxiv.org/abs/1909.04801
        - Page 1 of https://www.sketchingbigdata.org/fall17/lec/lec9.pdf
    
    Note:
        I name it sfd because the matrices are called S(ample), F(ourier transform), D(iagonal).
    """
    d = A.shape[0]
    sign_vector = np.random.randint(0, 2, size=(d, 1)) * 2 - 1
    idx = np.zeros(k, dtype=int)
    idx[1:] = np.random.choice(d - 1, k - 1, replace=False) + 1
    DA = sign_vector * A
    FDA = np.fft.fft(DA, axis=0, norm='ortho')
    A_embedded = np.sqrt(d / k) * FDA[idx]
    return A_embedded

def gaussian_random_projection(A, k):
    """
    Gaussian random projection from sklearn.
    """
    transformer = GaussianRandomProjection(n_components=k)
    A_embedded = transformer.fit_transform(A.T).T
    return A_embedded