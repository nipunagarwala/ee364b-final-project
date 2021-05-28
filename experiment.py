"""
Generate some results for midterm report
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import pickle
import time
import SparseLowRankInv as slri
from scipy import sparse

ACTION = 'flop'

# Parameters for running Sparse + Low-rank Inverse
SAVEDIR = 'result_paper'
PROJECTION = 'JLT'
NEG_EIG_VAL_METHOD = 'Abs'

# Parameter for baseline
NUM_EMBED_ROWS_LIST = np.arange(1, 21)  # Target rank r of U

# Parameter for experiment
EXPSIZE = 'small'

# Parameters for flop count plotting
# n = 43681
n = 32768
p_frac_list = np.array([1 / 5, 1 / 10, 1 / 20])
r_list = np.arange(1, 100)

np.random.seed(0)

def calcObjective(UMat, MStar, AMat):
	if slri.MAT_REPR_TYPE == 'Sparse':
		if UMat is None:
			temprod2 = MStar * AMat # Approx Inv A @ A 
		else:
			UMat = sparse.csc_matrix(UMat)
			tempProd = MStar + UMat * UMat.T
			temprod2 = tempProd * AMat
		finalProd = temprod2 + temprod2.T- 2 * sparse.identity(MStar.shape[0])
		obj = sparse.linalg.norm(finalProd,'fro')
	else:
		if UMat is None:
			temprod2 = np.dot(MStar, AMat) # Approx Inv A @ A 
		else:
			tempProd = MStar + np.dot(UMat, UMat.T)
			temprod2 = np.dot(tempProd, AMat) 
		finalProd = temprod2 + np.transpose(temprod2) - 2 * np.identity(MStar.shape[0])
		obj = np.linalg.norm(finalProd,'fro')
	return obj

def runSparseLowRankInvManyRanks(matPaths, targetRanks, completeInverse=False):

	print('---Begin finding approximate inverses with many target ranks---')
	print('Matrix name: {}'.format(matPaths))

	# Prepare logging variable and load matrices
	obj_log = np.zeros(len(targetRanks) + 1)  # First value is rank=0 i.e. only MStar as approximate inverse
	time_log = np.zeros(len(targetRanks))
	AMat, MStar = slri.read_matrices(matPaths)

	obj_log[0] = calcObjective(None, MStar, AMat)
	print('Starting norm: {}'.format(obj_log[0]))

	SMat, SMatCalcTime = slri.findSMat(MStar, AMat)

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

	results['targetRanks'] = targetRanks
	for name, path in zip(names, matPaths):
		results[name] = runSparseLowRankInvManyRanks([path], targetRanks)

	pickle.dump(results, open(os.path.join(SAVEDIR, '{}.p'.format(saveName), 'wb')))

	plotRanks = np.concatenate(([0], targetRanks))
	for name in names:
		plt.plot(plotRanks, results[name], label=name)
	plt.xlabel('Target rank, r')
	plt.ylabel(r'$||UU^TA + AUU^T - S||_F$')
	plt.title('Error of Sparse + Low-rank Approximation')
	plt.legend()
	plt.savefig(os.path.join(SAVEDIR, '{}.png'.format(saveName)), bbox_inches='tight')

def runExperiments(expSize, completeInverse=False):

	if expSize == 'small':
		# Dense-representation-friendly matrices
		mathPaths = [
			['matrices/Trefethen_64.mat', 'matrices/Trefethen_SSAI_64.mat'],
			['matrices/Trefethen_512.mat', 'matrices/Trefethen_SSAI_512.mat'],
			['matrices/Trefethen_4096.mat', 'matrices/Trefethen_SSAI_4096.mat'],
		]
		saveNames = ['Trefethen_64', 'Trefethen_512', 'Trefethen_4096']
		targetRanks = np.arange(2, 64, 4)

	elif expSize == 'large':
		# Large matrices
		mathPaths = [
			['matrices/Trefethen_32768.mat', 'matrices/Trefethen_SSAI_32768.mat'],
			['matrices/Wathen_43681.mat', 'matrices/Wathen_SSAI_43681.mat']
		]
		saveNames = ['Trefethen_32768', 'Wathen_43681']
		targetRanks = np.array([1, 2, 4, 6, 8, 12, 16, 20, 24, 28])

	for saveName, matPaths in zip(saveNames, mathPaths):
		obj_history, time_history, fullInvTime = runSparseLowRankInvManyRanks(matPaths, targetRanks, completeInverse=completeInverse)
		save_data = {
			'obj': obj_history,
			'time': time_history,
			'rank': targetRanks
		}

		# Save results
		pickle.dump(save_data, open(os.path.join(SAVEDIR, '{}.p'.format(saveName), 'wb')))

		# Plot objective
		plotRanks = np.concatenate(([0], targetRanks))
		plt.plot(plotRanks, obj_history)
		plt.xlabel('Target rank, r')
		plt.ylabel(r'$||UU^TA + AUU^T - S||_F$')
		plt.title('Error of Sparse + Low-rank Approximation')
		plt.savefig(os.path.join(SAVEDIR, '{}_obj.png'.format(saveName)), bbox_inches='tight')
		plt.clf()

		# Plot time
		plt.plot(targetRanks, time_history, label='Approximate inverse')
		if fullInvTime is not None:
			plt.axhline(y=fullInvTime, label='Complete inverse')
		plt.xlabel('Target rank, r')
		plt.ylabel('Time (s)')
		plt.title('Computation time')
		plt.legend()
		plt.savefig(os.path.join(SAVEDIR, '{}_time.png'.format(saveName)), bbox_inches='tight')
		plt.clf()

def plot_analytical_flop_counts(n, p_frac_list, r_list, savename):

	flop_counts = np.zeros((len(p_frac_list), len(r_list)))
	p_list = (n * p_frac_list).astype(int)

	# Compute flop counts not depending on the sparisty p
	SJLT_count = (n + 1) * n * r_list
	QR_count = 2 * (n - r_list / 3) * (r_list ** 2)
	cvx_count = 1000 * (r_list ** 3)
	cholesky_count = (r_list ** 3) / 3
	U_count = n * (r_list ** 2)
	count_no_p = SJLT_count + QR_count + cvx_count + cholesky_count + U_count

	# Computing flop counts depending on the sparsity p
	for i, p in enumerate(p_list):
		MStar_count = 2 * n * (p ** 2)
		S_count = 2 * (n ** 2) * p
		A_tilde_count = 2 * n * p * r_list
		E_count = 2 * n * (p ** 2) * r_list
		flop_counts[i] = count_no_p + MStar_count + S_count + A_tilde_count + E_count

	inv_count = n ** 3

	# Plotting
	for i, p in enumerate(p_frac_list):
		plt.plot(r_list, flop_counts[i], label='p / n = {}'.format(p))
	plt.axhline(y=inv_count, c='r', ls='--', label='Matrix inversion')
	plt.xlabel('r')
	plt.ylabel('Flop count')
	plt.title('Flop Count of Approximate Inverse, n = {}'.format(n))
	plt.legend()
	plt.savefig(os.path.join(SAVEDIR, '{}.png'.format(savename)), bbox_inches='tight')
	plt.clf()

def main():
	if ACTION == 'baseline':
		runBaselineMatrices(NUM_EMBED_ROWS_LIST, 'baseline')
	elif ACTION == 'experiment':
		runExperiments(completeInverse=False)
	elif ACTION == 'flop':
		plot_analytical_flop_counts(n, p_frac_list, r_list, 'flop_{}'.format(n))
	else:
		raise ValueError('Invalid ACTION: {}'.format(ACTION))

if __name__ == '__main__':
	main()