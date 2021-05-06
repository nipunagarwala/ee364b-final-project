#!/usr/local/bin/python3
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from numpy import linalg as LA
import random
from scipy.linalg import ldl
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
	MStarMatDense = np.asarray(MStarMat_CSR.todense())
	Idn = np.identity(MStarMatDense.shape[0])
	MStarAMatProd = np.dot(MStarMatDense, AMat)
	SMat = 2*Idn - MStarAMatProd - MStarAMatProd.T

	return SMat, MStarMatDense


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
	PMat = cp.Variable((RNumCol,RNumCol))
	objective_fn = cp.norm(EMat - cp.matmul(PMat,AtildeMat) - cp.matmul(AtildeMat, PMat),
							p='fro')
	prob = cp.Problem(cp.Minimize(objective_fn))

	prob.solve()
	PMatVal = PMat.value

	return PMatVal

def doCholeskyFactAbsEigenVal(PMat):
	print(PMat.shape)
	eigVal, eigVec = np.linalg.eig(PMat)
	eigValAbsDiagMat = np.diag(abs(eigVal))
	PMatPSD = np.dot(np.dot(eigVec, eigValAbsDiagMat), eigVec.T)
	UTildeMat = np.linalg.cholesky(PMatPSD)
	return UTildeMat

def findUMat(QMat, UTildeMat):
	UMat = np.dot(QMat, UTildeMat)
	return UMat


def checkResult(UMat, MStarMatDense, AMat):
	tempProd = MStarMatDense + np.dot(UMat, UMat.T)
	temprod2 = np.dot(tempProd, AMat) 
	finalProd = temprod2+np.transpose(temprod2)- 2*np.identity(MStarMatDense.shape[0])
	finnorm = LA.norm(finalProd,'fro')
	temprod2 = np.dot(MStarMatDense, AMat) 
	finalProd = temprod2 + np.transpose(temprod2)- 2*np.identity(MStarMatDense.shape[0])
	startnorm = LA.norm(finalProd,'fro')

	print("Starting Norm: {0}".format(startnorm))
	print("Final Norm: {0}".format(finnorm))



def main():
	AMatPath = "matrices/Wathen_11041.mat"
	MStarPath = "matrices/Wathen_SSAI_11041.mat"
	NumEmbedRows = 3000
	AMat_dict = sio.loadmat(AMatPath, squeeze_me=True)
	AMat = None
	if "Wathen" in AMatPath:
		AMat = np.asarray(AMat_dict['A'].todense()) #FIXME: Change to sparse representation only
	elif "Trefethen" in AMatPath:
		AMat = np.asarray(AMat_dict['tref2'].todense()) #FIXME: Change to sparse representation only

	SMat, MStarMatDense = findSMat(MStarPath, "Mst", AMat)
	QMat, RNumCols = findThinQ(SMat, NumEmbedRows, RndType='JLT')
	AMatTilde = findATilde(QMat, AMat)
	EMat = findEMat(QMat, SMat)

	PVal = solveForPSDSymmetricP(EMat,AMatTilde, RNumCols)
	UTildeMat = doCholeskyFactAbsEigenVal(PVal)
	UMat = findUMat(QMat, UTildeMat)
	checkResult(UMat, MStarMatDense, AMat)



if __name__ == '__main__':
	main()

