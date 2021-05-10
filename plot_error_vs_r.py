"""
Generate some results for midterm report
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
from SparseLowRankInv import *

MAT_REPR_TYPE = "Dense"
AMatPath = "matrices/Trefethen_64.mat"
MStarPath = "matrices/Trefethen_SSAI_64.mat"
IMAGE_NAME = 'Trefethen_64_JLT_Abs2.png'
NUM_EMBED_ROWS_LIST = np.arange(2, 64, 2)
PROJECTION = 'JLT'
NEG_EIG_VAL_METHOD = 'Abs'

def calc_objective(UMat, MStarMatDense, AMat):
	if UMat is None:
		temprod2 = np.dot(MStarMatDense, AMat) 
	else:
		tempProd = MStarMatDense + np.dot(UMat, UMat.T)
		temprod2 = np.dot(tempProd, AMat) 
	finalProd = temprod2 + np.transpose(temprod2) - 2 * np.identity(MStarMatDense.shape[0])
	obj = LA.norm(finalProd,'fro')
	return obj

def main():

	obj_log = np.zeros(len(NUM_EMBED_ROWS_LIST))
	AMat_dict = sio.loadmat(AMatPath, squeeze_me=True)
	AMat = None
	if "Wathen" in AMatPath:
		if MAT_REPR_TYPE == "Sparse":
			AMat = sparse.csr_matrix(np.asarray(AMat_dict['A'].todense()))
		else:
			AMat = np.asarray(AMat_dict['A'].todense()) #FIXME: Change to sparse representation only
	elif "Trefethen" in AMatPath:
		if MAT_REPR_TYPE == "Sparse":
			AMat = sparse.csr_matrix(np.asarray(AMat_dict['tref2'].todense()))
		else:
			AMat = np.asarray(AMat_dict['tref2'].todense()) #FIXME: Change to sparse representation only

	SMat, MStarMatDense = findSMat(MStarPath, "Mst", AMat)
	obj_only_Mstar = calc_objective(None, MStarMatDense, AMat)
	print('Starting norm: {}'.format(obj_only_Mstar))
	for i, NUM_EMBED_ROWS in enumerate(NUM_EMBED_ROWS_LIST):
		QMat, RNumCols = findThinQ(SMat, NUM_EMBED_ROWS, RndType=PROJECTION)
		AMatTilde = findATilde(QMat, AMat)
		EMat = findEMat(QMat, SMat)

		PVal = solveForPSDSymmetricP(EMat,AMatTilde, RNumCols)
		UTildeMat, newNumRows = None, None
		if NEG_EIG_VAL_METHOD == "Abs":
			UTildeMat = doCholeskyFactAbsEigenVal(PVal)
		elif NEG_EIG_VAL_METHOD == "Discard":
			UTildeMat, newNumRows = doCholeskyFactEigenReduction(PVal)

		UMat = findUMat(QMat, UTildeMat, newNumRows)
		obj_log[i] = calc_objective(UMat, MStarMatDense, AMat)
		print('r = {}, norm: {}'.format(NUM_EMBED_ROWS, obj_log[i]))
	
	plt.plot(NUM_EMBED_ROWS_LIST, obj_log, label=r'$M^* + UU^T$')
	plt.axhline(y=obj_only_Mstar, label=r'M^*', c='r')
	plt.xlabel('Rank, r')
	plt.ylabel(r'$||UU^TA + AUU^T - S||_F$')
	plt.title('Error of Sparse + Low-rank approximation')
	plt.legend()
	plt.savefig(IMAGE_NAME)

if __name__ == '__main__':
	main()