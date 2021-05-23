"""
Generate some results for midterm report
"""

import numpy as np
from matplotlib import pyplot as plt
import pickle
from SparseLowRankInv import *

MAT_REPR_TYPE = "Sparse"
SAVE_NAME = 'result/baselines'
NUM_EMBED_ROWS_LIST = np.arange(1, 21)
PROJECTION = 'JLT'
NEG_EIG_VAL_METHOD = 'Abs'

np.random.seed(0)

def calcObjective(UMat, MStarMatDense, AMat):
	if UMat is None:
		temprod2 = np.dot(MStarMatDense, AMat) # Approx Inv A @ A 
	else:
		tempProd = MStarMatDense + np.dot(UMat, UMat.T)
		temprod2 = np.dot(tempProd, AMat) 
	finalProd = temprod2 + np.transpose(temprod2) - 2 * np.identity(MStarMatDense.shape[0])
	obj = LA.norm(finalProd,'fro')
	return obj

def runSparseLowRankInvManyRanks(matPaths, targetRanks):

	print('---Begin finding approximate inverses with many target ranks---')
	print('Matrix name: {}'.format(matPaths))

	# Prepare logging variable and load matrices
	obj_log = np.zeros(len(targetRanks) + 1)  # First value is rank=0 i.e. only MStar as approximate inverse
	AMat, MStar = read_matrices(matPaths)

	SMat = findSMat(MStar, AMat)
	obj_log[0] = calcObjective(None, MStar, AMat)
	print('Starting norm: {}'.format(obj_log[0]))

	# Find the low-rank correction U matrix for each target rank
	for i, rank in enumerate(targetRanks):
		QMat, RNumCols = findThinQ(SMat, rank, RndType=PROJECTION)
		AMatTilde = findATilde(QMat, AMat)
		EMat = findEMat(QMat, SMat)

		PVal = solveForPSDSymmetricP(EMat,AMatTilde, RNumCols)
		UTildeMat, newNumRows = None, None
		if NEG_EIG_VAL_METHOD == "Abs":
			UTildeMat = doCholeskyFactAbsEigenVal(PVal)
		elif NEG_EIG_VAL_METHOD == "Discard":
			UTildeMat, newNumRows = doCholeskyFactEigenReduction(PVal)

		UMat = findUMat(QMat, UTildeMat, newNumRows)
		obj_log[i + 1] = calcObjective(UMat, MStar, AMat)
		print('r = {}, norm: {}'.format(rank, obj_log[i]))

	return obj_log

def runBaselineMatrices(targetRanks, saveName):

	results = {}
	names = ['rank = 2', 'rank = 4', 'rank = 6', 'rank = 8']
	matPaths = [
		'matrices/SPLRI_n4033_r2.mat',
		'matrices/SPLRI_n4033_r4.mat',
		'matrices/SPLRI_n4033_r6.mat',
		'matrices/SPLRI_n4033_r8.mat',
	]
	for name, path in zip(names, matPaths):
		results[name] = runSparseLowRankInvManyRanks([path], targetRanks)

	pickle.dump(results, open('{}.p'.format(saveName), 'wb'))

	plotRanks = np.concatenate(([0], targetRanks))
	for name in names:
		plt.plot(plotRanks, results[name], label=name)
	plt.xlabel('Target rank')
	plt.ylabel(r'$||UU^TA + AUU^T - S||_F$')
	plt.title('Error of Sparse + Low-rank Approximation')
	plt.legend()
	plt.savefig('{}.png'.format(saveName))

def runExperiments():
	# MAT_PATHS = [
	# 	['matrices/Trefethen_64.mat', 'matrices/Trefethen_SSAI_64.mat'],
	# 	['matrices/Trefethen_512.mat', 'matrices/Trefethen_SSAI_512.mat'],
	# 	['matrices/Trefethen_4096.mat', 'matrices/Trefethen_SSAI_4096.mat'],
	# 	['matrices/Trefethen_32768.mat', 'matrices/Trefethen_SSAI_32768.mat'],
	# 	['matrices/Wathen_11041.mat', 'matrices/Wathen_SSAI_11041.mat'],
	# 	['matrices/Wathen_43681.mat', 'matrices/Wathen_SSAI_43681.mat'],
	# 	['matrices/Wathen_146081.mat', 'matrices/Wathen_SSAI_146081.mat']
	# ]
	# SAVE_NAMES = ['result/Trefethen_64', 'result/Trefethen_512',
	# 				'result/Trefethen_4096', 'result/Trefethen_32768',
	# 				'result/Wathen_11041', 'result/Wathen_43681',
	# 				'result/Wathen_146081']

	# Dense-representation-friendly matrices
	MAT_PATHS = [
		['matrices/Trefethen_64.mat', 'matrices/Trefethen_SSAI_64.mat'],
		['matrices/Trefethen_512.mat', 'matrices/Trefethen_SSAI_512.mat'],
		['matrices/Trefethen_4096.mat', 'matrices/Trefethen_SSAI_4096.mat'],
	]
	SAVE_NAMES = ['result/Trefethen_64', 'result/Trefethen_512', 'result/Trefethen_4096']
	TARGET_RANKS = np.arange(2, 65, 2)

	for saveName, matPaths in zip(SAVE_NAMES, MAT_PATHS):
		obj_history = runSparseLowRankInvManyRanks(matPaths, TARGET_RANKS)
		plotRanks = np.concatenate(([0], TARGET_RANKS))
		plt.plot(plotRanks, obj_history)
		plt.xlabel('Target rank')
		plt.ylabel(r'$||UU^TA + AUU^T - S||_F$')
		plt.title('Error of Sparse + Low-rank Approximation')
		plt.savefig('{}.png'.format(saveName))
		plt.clf()

def main():
	# runBaselineMatrices(NUM_EMBED_ROWS_LIST, SAVE_NAME)
	runExperiments()

if __name__ == '__main__':
	main()