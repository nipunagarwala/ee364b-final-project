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
from scipy import sparse
import fjlt
import time

'''
Global variables to tune and matrices to use.
'''

MAT_REPR_TYPE = "Dense"
# AMatPath = "matrices/Trefethen_4096.mat"
# MStarPath = "matrices/Trefethen_SSAI_4096.mat"
# matPaths = ["matrices/SPLRI_n4033_r2.mat"]
matPaths = [ "matrices/Trefethen_4096.mat",  "matrices/Trefethen_SSAI_4096.mat"]
NUM_EMBED_ROWS = 4

# Options include: Abs, Discard
NEG_EIG_VAL_METHOD = "Discard"


def findSMat(Mstar, AMat):
	if MAT_REPR_TYPE == "Sparse":
		Idn = sparse.csr_matrix(np.identity(Mstar.shape[0]))
	else:
		Idn = np.identity(Mstar.shape[0])

	MStarAMatProd = None
	start_time = time.perf_counter()
	if MAT_REPR_TYPE == "Sparse":
		MStarAMatProd = Mstar*AMat
	else:
		MStarAMatProd = np.dot(Mstar, AMat)

	SMat = 2*Idn - MStarAMatProd - MStarAMatProd.T
	end_time = time.perf_counter()

	SMatCalcTime = (end_time - start_time)
	print("Finished Computing S matrix ...")
	return SMat, SMatCalcTime

def findThinQ(SMat, EmbedRow, RndType='JLT'):
	SMatOmegaMatProd = np.zeros((EmbedRow, SMat.shape[1]))
	start_time = time.perf_counter()
	if RndType == 'Gaussian':
		SMatOmegaMatProd = fjlt.gaussian_random_projection(SMat, EmbedRow)
	elif RndType == 'JLT':
		if MAT_REPR_TYPE == "Sparse":
			SMatOmegaMatProd = fjlt.sjlt(SMat, EmbedRow)
		else:
			SMatOmegaMatProd = fjlt.fjlt(SMat, EmbedRow)
	else:
		sys.exit('Random Matrix type is neither JLT nor Gaussian')

	Q,R = LA.qr(SMatOmegaMatProd)
	end_time = time.perf_counter()
	ThinQCalcTime += (end_time - start_time)
	MNumRows = SMatOmegaMatProd.shape[0]
	NNumCols = SMatOmegaMatProd.shape[1]
	QFinal = Q
	if MNumRows > NNumCols:
		QFinal = Q[:, np.arange(NNumCols)]

	print("Finished Computing QR Factorization ...")
	return QFinal, NNumCols,ThinQCalcTime


def findATilde(QMat,AMat):
	start_time = time.perf_counter()
	Atilde = np.dot(np.dot(QMat.T, AMat), QMat)
	end_time = time.perf_counter()
	AtildeCalcTime += (end_time - start_time)
	return Atilde, AtildeCalcTime


def findEMat(QMat, SMat):
	start_time = time.perf_counter()
	EMat = np.dot(np.dot(QMat.T, SMat), QMat)
	end_time = time.perf_counter()
	EMatCalcTime += (end_time - start_time)
	return EMat, EMatCalcTime


def solveForPSDSymmetricP(EMat, AtildeMat, RNumCol):
	print("Starting CVXPY Solver")
	PMat = cp.Variable((RNumCol,RNumCol), symmetric=True)
	objective_fn = cp.norm(EMat - cp.matmul(PMat,AtildeMat) - cp.matmul(AtildeMat, PMat),
							p='fro')
	prob = cp.Problem(cp.Minimize(objective_fn),
					[PMat >> 0])

	start_time = time.perf_counter()
	prob.solve()
	end_time = time.perf_counter()
	PSDCalcTime += (end_time - start_time)
	PMatVal = PMat.value

	print("Ended CVXPY Solver\n")

	return PMatVal, PSDCalcTime

def doCholeskyFactAbsEigenVal(PMat):
	print("Started Cholesky Factorization\n")

	start_time = time.perf_counter()
	eigVal, eigVec = np.linalg.eig(PMat)
	eigValAbsDiagMat = np.diag(abs(eigVal))
	PMatPSD = np.dot(np.dot(eigVec, eigValAbsDiagMat), eigVec.T)
	UTildeMat = np.linalg.cholesky(PMatPSD)
	end_time = time.perf_counter()
	CholeskyCalcTime += (end_time - start_time)
	return UTildeMat, CholeskyCalcTime


def doCholeskyFactEigenReduction(PMat):
	print("Started Cholesky Factorization\n")

	start_time = time.perf_counter()
	eigVal, eigVec = np.linalg.eig(PMat)
	# Sorting the eigenvalues and eigenvectors because apparently Numpy does not do that
	idx = eigVal.argsort()[::-1]   
	eigVal = eigVal[idx]
	eigVec = eigVec[:,idx]
	end_time = time.perf_counter()
	CholeskyCalcTime += (end_time - start_time)

	# Finding the first < 0 eigenvalue and discarding the negative eigenvalues
	# Also discarding the associated eigenvectors and the associated rows.
	eigValRed = np.where(eigVal > 0, eigVal, 0*eigVal)
	firstZeroEig = np.argwhere(eigValRed <= 0)
	if firstZeroEig.size == 0:
		firstZeroEig = eigVal.shape[0]
	else:
		firstZeroEig = firstZeroEig[0][0]

	reducedEigValDiag = np.diag(eigVal[:firstZeroEig])
	reducedEigVec = eigVec[:firstZeroEig,:firstZeroEig]
	PMatPSD = np.dot(np.dot(reducedEigVec, reducedEigValDiag), reducedEigVec.T)
	UTildeMat = np.linalg.cholesky(PMatPSD)
	return UTildeMat, firstZeroEig, CholeskyCalcTime



def findUMat(QMat, UTildeMat, newNumRows=NUM_EMBED_ROWS):
	# If we need an even further reduction in number of rows due to discarding
	# of eigenvalues, do that with the newNumRows
	start_time = time.perf_counter()
	UMat = np.dot(QMat[:,:newNumRows], UTildeMat)
	end_time = time.perf_counter()
	UMatCalcTime += (end_time - start_time)
	return UMat, UMatCalcTime

def findUMatSmall(QMat, UTildeMat):
	Ushape=UTildeMat.shape # number of rows for U
	UMat = np.dot(QMat[:,0:Ushape[0]], UTildeMat) #take same number of columns for Q
	return UMat


def checkResult(UMat, MStarMatDense, AMat):
	print("Checking awesomenesssss of Result\n")
	tempProd = MStarMatDense + np.dot(UMat, UMat.T)
	temprod2 = np.dot(tempProd, AMat) 
	finalProd = temprod2+np.transpose(temprod2)- 2*np.identity(MStarMatDense.shape[0])
	finnorm = LA.norm(finalProd,'fro')
	temprod2 = np.dot(MStarMatDense, AMat) 
	finalProd = temprod2 + np.transpose(temprod2)- 2*np.identity(MStarMatDense.shape[0])
	startnorm = LA.norm(finalProd,'fro')

	print("Starting Norm: {0}".format(startnorm))
	print("Final Norm: {0}".format(finnorm))


def read_matrices(matPaths):
	AMat, MStar = None, None
	if len(matPaths) == 1:
		data = sio.loadmat(matPaths[0], squeeze_me=True)
		AMat = data['IAr']  # Dense
		MStar = np.asarray(data['A'].todense())   # Dense
	else:
		AMatPath, MStarPath = matPaths
		if "Wathen" in AMatPath:
			if MAT_REPR_TYPE == "Sparse":
				AMat = sio.loadmat(AMatPath, squeeze_me=True)['A']      # Sparse CSC
				MStar = sio.loadmat(MStarPath, squeeze_me=True)['Mst']  # Sparse CSC
			else:
				AMat = sio.loadmat(AMatPath, squeeze_me=True)['A'].todense()      # Sparse CSC
				MStar = sio.loadmat(MStarPath, squeeze_me=True)['Mst'].todense()  # Sparse CSC
		elif "Trefethen" in AMatPath:
			if MAT_REPR_TYPE == "Sparse":
				AMat = sio.loadmat(AMatPath, squeeze_me=True)['tref2']  # Sparse CSC
				MStar = sio.loadmat(MStarPath, squeeze_me=True)['Mst']  # Sparse CSC
			else:
				AMat = sio.loadmat(AMatPath, squeeze_me=True)['tref2'].todense() 
				MStar = sio.loadmat(MStarPath, squeeze_me=True)['Mst'].todense()

	print("Finished reading in the Matrices")
	return AMat, MStar


def main():

	AMat, MStar = read_matrices(matPaths)
	SMat, SMatCalcTime = findSMat(MStar, AMat)
	QMat, RNumCols, ThinQCalcTime = findThinQ(SMat, NUM_EMBED_ROWS, RndType='JLT')
	AMatTilde, AtildeCalcTime = findATilde(QMat, AMat)
	EMat, EMatCalcTime = findEMat(QMat, SMat)

	PVal, PSDCalcTime = solveForPSDSymmetricP(EMat,AMatTilde, RNumCols)
	UTildeMat, newNumRows = None, None
	CholeskyCalcTime = 0
	if NEG_EIG_VAL_METHOD == "Abs":
		UTildeMat, CholeskyCalcTime = doCholeskyFactAbsEigenVal(PVal)
	elif NEG_EIG_VAL_METHOD == "Discard":
		UTildeMat, newNumRows, CholeskyCalcTime = doCholeskyFactEigenReduction(PVal)

	UMat, UMatCalcTime = findUMat(QMat, UTildeMat, newNumRows)
	checkResult(UMat, MStar, AMat)

	totalPerfTime = (SMatCalcTime + ThinQCalcTime + AtildeCalcTime + 
				EMatCalcTime + PSDCalcTime + CholeskyCalcTime)

	print("Total runtime for Approximate Inverse: {0}".format(totalPerfTime))

	start_time = time.perf_counter()
	AMatInv = LA.inv(AMat)
	end_time = time.perf_counter()
	fullInvTime = (end_time - start_time)
	print("Total runtime for Complete Inverse: {0}".format(fullInvTime))

if __name__ == '__main__':
	main()

