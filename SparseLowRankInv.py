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
import fjlt

'''
FIXME: Use sparse representation to save computation
'''

def findSMat(MstarPath, MatlabVarName, AMat=None):
	MStarMat_dict = sio.loadmat(MstarPath, squeeze_me=True)
	# Hardcoded to the 4th entry since that seems to be the name of the key actually having
	# the matrix value
	MStarMat_CSR = MStarMat_dict[MatlabVarName]
	MStarMatDense = MStarMat_CSR.todense()
	Idn = np.identity(MStarMatDense.shape[0])
	MStarAMatProd = np.dot(MStarMatDense, AMat)
	SMat = 2*Idn - MStarAMatProd - MStarAMatProd.T

	return SMat


def findThinQ(SMat, EmbedRow, RndType='JLT'):
	SMatOmegaMatProd = np.zeros((EmbedRow, SMat.shape[1]))
	if RndType == 'Gaussian':
		SMatOmegaMatProd = fjlt.gaussian_random_projection(SMat, EmbedRow)
	elif RndType == 'JLT':
		SMatOmegaMatProd = fjlt.fjlt(SMat, EmbedRow)
	else:
		sys.exit('Random Matrix type is neither JLT nor Gaussian')

	Q,R = LA.qr(SMatOmegaMatProd)
	MNumRows = SMatOmegaMatProd.shape[0]
	NNumCols = SMatOmegaMatProd.shape[1]
	QFinal = Q
	if MNumRows > NNumCols:
		QFinal = Q[:, np.arange(NNumCols)]

	return QFinal, NNumCols


def findATilde(QMat,AMat):
	Atilde = np.dot(np.dot(QMat.T, AMat), QMat)
	return Atilde


def findEMat(QMat, SMat):
	EMat = np.dot(np.dot(QMat.T, SMat), QMat)
	return EMat


def solveForPSDSymmetricP(EMat, AtildeMat, RNumCol):
	PMat = cp.Variable(RNumCol, RNumCol)
	objective_fn = cp.norm(EMat - cp.matmul(PMat,AtildeMat) - cp.matmul(AtildeMat, PMat),
							p='fro')
	prob = cp.Problem(cp.Minimize(objective_fn),
				[PMat >> 0])

	prob.solve()
	PMatVal = PMat.value

	print(PMat.value)

	return PMatVal


def main():
	mean = np.zeros(64)
	CovMat = np.identity(64)
	AMat_dict = sio.loadmat("matrices/Trefethen_64.mat", squeeze_me=True)
	AMat = AMat_dict['tref2'].todense() #FIXME: Change to sparse representation only
	SMat = findSMat("matrices/Trefethen_SSAI_64.mat", "Mst", AMat)
	QMat, RNumCols = findThinQ(SMat, 24, RndType='JLT')
	AMatTilde = findATilde(QMat, AMat)
	EMat = findEMat(QMat, SMat)

	PVal = solveForPSDSymmetricP(EMat,AMatTilde, RNumCols)



if __name__ == '__main__':
	main()

