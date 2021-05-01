#!/usr/local/bin/python3
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from numpy import linalg as LA
import random
import scipy.linalg
import math
import scipy.io as sio
from ipynb.fs.full.my_functions import fjlt_sfd, gaussian_random_projection


def findSMat(MstarPath, MatlabVarName, AMat=None):
	MStarMat_dict = sio.loadmat(MstarPath, squeeze_me=True)
	# Hardcoded to the 4th entry since that seems to be the name of the key actually having
	# the matrix value
	MStarMat_CSR = MStarMat_dict[MatlabVarName]
	MStarMatDense = MStarMat_CSR.todense()
	Idn = np.identity(MStarMatDense.shape[0])
	SMat = 2*Idn - np.dot(MStarMatDense, AMat) + np.dot(AMat, MStarMatDense)

	return SMat

def findThinQ(SMat, EmbedRow, RndType='JLT'):
	SMatOmegaMatProd = np.zeros((EmbedRow, SMat.shape[1]))
	if RndType == 'Gaussian':
		SMatOmegaMatProd = gaussian_random_projection(SMat, EmbedRow)
	elif RndType == 'JLT':
		SMatOmegaMatProd = fjlt_sfd(SMat, EmbedRow)
	else:
		sys.exit('Random Matrix type is neither JLT nor Gaussian')

	Q,R = linalg.qr(SMatOmegaMatProd)
	MNumRows = SMatOmegaMatProd.shape[0]
	NNumCols = SMatOmegaMatProd.shape[0]
	QFinal = Q
	if MNumRows > NNumCols:
		QFinal = Q[:, np.arange(NNumCols)]

	return QFinal


def findATilde(QMat,AMat):
	Atilde = np.dot(np.dot(QMat.T, AMat), QMat)
	return Atilde


def findEMat(QMat, SMat):
	EMat = np.dot(np.dot(QMat.T, SMat), QMat)
	return EMat


def solveForPSDSymmetricP(EMat, AtildeMat, RNumCol):
	PMat = cp.Variable(RNumCol, RNumCol)
	objective_fn = cp.norm2(EMat - cp.matmul(PMat,AtildeMat) - cp.matmul(AtildeMat, PMat))
	prob = cp.Problem(cp.Minimize(objective_fn))

	prob.solve()
	PMatVal = PMat.value

	return PMatVal



def main():
	findSMat("matrices/Trefethen_64.mat", "tref2")



if __name__ == '__main__':
	main()

