"""
Generate some results for midterm report
"""

import numpy as np
from matplotlib import pyplot as plt
import pickle
import time
import SparseLowRankInv as slri

MAT_REPR_TYPE = "Dense"
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
	obj = np.linalg.norm(finalProd,'fro')
	return obj

def runSparseLowRankInvManyRanks(matPaths, targetRanks, completeInverse=False):

	print('---Begin finding approximate inverses with many target ranks---')
	print('Matrix name: {}'.format(matPaths))

	# Prepare logging variable and load matrices
	obj_log = np.zeros(len(targetRanks) + 1)  # First value is rank=0 i.e. only MStar as approximate inverse
	time_log = np.zeros(len(targetRanks))
	AMat, MStar = slri.read_matrices(matPaths)

	SMat, SMatCalcTime = slri.findSMat(MStar, AMat)
	obj_log[0] = calcObjective(None, MStar, AMat)
	print('Starting norm: {}'.format(obj_log[0]))

	# Find the low-rank correction U matrix for each target rank
	for i, rank in enumerate(targetRanks):
		QMat, RNumCols, ThinQCalcTime = slri.findThinQ(SMat, rank, RndType=PROJECTION)
		AMatTilde, AtildeCalcTime = slri.findATilde(QMat, AMat)
		EMat, EMatCalcTime = slri.findEMat(QMat, SMat)

		PVal, PSDCalcTime = slri.solveForPSDSymmetricP(EMat,AMatTilde, RNumCols)
		UTildeMat, newNumRows = None, None
		if NEG_EIG_VAL_METHOD == "Abs":
			UTildeMat, CholeskyCalcTime = slri.doCholeskyFactAbsEigenVal(PVal)
		elif NEG_EIG_VAL_METHOD == "Discard":
			UTildeMat, newNumRows, CholeskyCalcTime = slri.doCholeskyFactEigenReduction(PVal)

		UMat, UMatCalcTime = slri.findUMat(QMat, UTildeMat, newNumRows)
		obj_log[i + 1] = calcObjective(UMat, MStar, AMat)
		time_log[i] = SMatCalcTime + ThinQCalcTime + AtildeCalcTime + EMatCalcTime +\
						PSDCalcTime + CholeskyCalcTime + UMatCalcTime
		print('r = {}, norm: {}'.format(rank, obj_log[i + 1]))
		print('Running time: {}'.format(time_log[i]))

	fullInvTime = None
	if completeInverse:
		start_time = time.perf_counter()
		AMatInv = np.linalg.inv(AMat)
		end_time = time.perf_counter()
		fullInvTime = (end_time - start_time)

	return obj_log, time_log, fullInvTime

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
	plt.xlabel('Target rank, r')
	plt.ylabel(r'$||UU^TA + AUU^T - S||_F$')
	plt.title('Error of Sparse + Low-rank Approximation')
	plt.legend()
	plt.savefig('{}.png'.format(saveName))

def runExperiments(completeInverse=False):
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
	TARGET_RANKS = np.arange(1, 65)

	for saveName, matPaths in zip(SAVE_NAMES, MAT_PATHS):
		obj_history, time_history, fullInvTime = runSparseLowRankInvManyRanks(matPaths, TARGET_RANKS, completeInverse=completeInverse)
		save_data = {
			'obj': obj_history,
			'time': time_history,
			'rank': TARGET_RANKS
		}

		# Save results
		pickle.dump(save_data, open('{}.p'.format(saveName), 'wb'))

		# Plot objective
		plotRanks = np.concatenate(([0], TARGET_RANKS))
		plt.plot(plotRanks, obj_history)
		plt.xlabel('Target rank, r')
		plt.ylabel(r'$||UU^TA + AUU^T - S||_F$')
		plt.title('Error of Sparse + Low-rank Approximation')
		plt.savefig('{}_obj.png'.format(saveName))
		plt.clf()

		# Plot time
		plt.plot(TARGET_RANKS, time_history, label='Approximate inverse')
		if fullInvTime is not None:
			plt.axhline(y=fullInvTime, label='Complete inverse')
		plt.xlabel('Target rank, r')
		plt.ylabel('Time (s)')
		plt.title('Computation time')
		plt.legend()
		plt.savefig('{}_time.png'.format(saveName))
		plt.clf()

def plot_analytical_flop_counts(n, p_frac_list, r_list, savename):

	flop_counts = np.zeros((len(p_frac_list), len(r_list)))
	p_list = (n * p_frac_list).astype(int)

	# Compute flop counts not depending on the sparisty p
	SJLT_count = n * (n + 1) * r_list
	QR_count = 2 * (n - r_list / 3) * (r_list ** 2)
	A_tilde_count = 2 * (n ** 2) * r_list
	E_count = 2 * (n ** 2) * r_list
	cvx_count = 1000 * (r_list ** 3)
	cholesky_count = (r_list ** 3) / 3
	U_count = (n ** 2) * r_list
	count_no_p = SJLT_count + QR_count + A_tilde_count + E_count + cvx_count + \
					cholesky_count + U_count

	# Computing flop counts depending on the sparsity p
	for i, p in enumerate(p_list):
		MStar_count = 2 * (n ** 2) * p
		S_count = 2 * (n ** 2) * p
		flop_counts[i] = count_no_p + MStar_count + S_count

	inv_count = n ** 3

	# Plotting
	for i, p in enumerate(p_frac_list):
		plt.plot(r_list, flop_counts[i], label='p / n = {}'.format(p))
	plt.axhline(y=inv_count, c='r', ls='--', label='Matrix inversion')
	plt.xlabel('r')
	plt.ylabel('Flop count')
	plt.title('Flop Count of Approximate Inverse, n = {}'.format(n))
	plt.legend()
	plt.savefig('{}.png'.format(savename))
	plt.clf()

def main():
	# runBaselineMatrices(NUM_EMBED_ROWS_LIST, SAVE_NAME)
	runExperiments(completeInverse=True)
	# n = 40000
	# p_frac_list = np.array([1 / 5, 1 / 10, 1 / 20])
	# r_list = np.arange(1, 100)
	# plot_analytical_flop_counts(n, p_frac_list, r_list, 'result/flop_{}'.format(n))

if __name__ == '__main__':
	main()