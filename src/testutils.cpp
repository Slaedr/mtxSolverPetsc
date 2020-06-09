/** \file
 * \brief Implementation of some testing utilities
 */

#undef NDEBUG
#include <cassert>
#include <stdexcept>
#include <string>
#include <cmath>
#include <float.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <petscksp.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/baij/mpi/mpibaij.h>

#include "utils/mpiutils.hpp"
#include "utils/cmdoptions.hpp"
#include "testutils.h"
#include "testutils.hpp"

#define PETSCOPTION_STR_LEN 30

namespace blasted {

SRMatrixStorage<const PetscScalar,const PetscInt> wrapLocalPetscMat(Mat A, const int bs)
{
	PetscInt firstrow, lastrow, localrows, localcols, globalrows, globalcols;
	int ierr = MatGetOwnershipRange(A, &firstrow, &lastrow); petsc_throw(ierr);
	ierr = MatGetLocalSize(A, &localrows, &localcols); petsc_throw(ierr);
	ierr = MatGetSize(A, &globalrows, &globalcols); petsc_throw(ierr);
	assert(localrows == localcols);
	assert(globalrows == globalcols);

	// get access to local matrix entries
	if(bs == 1) {
		const Mat_SeqAIJ *const Adiag = (const Mat_SeqAIJ*)A->data;
		assert(Adiag != NULL);

		return SRMatrixStorage<const PetscScalar, const PetscInt>(Adiag->i, Adiag->j, Adiag->a,
		                                                          Adiag->diag, Adiag->i + 1,
		                                                          localrows, Adiag->i[localrows],
		                                                          Adiag->i[localrows], 1);
	}
	else {
		const Mat_SeqBAIJ *const Adiag = (const Mat_SeqBAIJ*)A->data;
		assert(Adiag != NULL);
		assert(localrows % bs == 0);
		printf(" wrapLocalPetscMat: wrapping block matrix of size %d.\n", bs);

		return SRMatrixStorage<const PetscScalar, const PetscInt>(Adiag->i, Adiag->j, Adiag->a,
		                                                          Adiag->diag, Adiag->i + 1,
		                                                          localrows/bs, Adiag->i[localrows/bs],
		                                                          Adiag->i[localrows/bs], bs);
	}
}
}

extern "C" {

int compareSolverWithRef(const int refkspiters, const int avgkspiters,
                         Vec uref, Vec u)
{
	const int rank = blasted::get_mpi_rank(MPI_COMM_WORLD);
	if(rank == 0)
		printf("KSP Iters: Reference %d vs BLASTed %d.\n", refkspiters, avgkspiters);
	fflush(stdout);

	const std::string testtype = blasted::parsePetscCmd_string("-test_type", PETSCOPTION_STR_LEN);
	const double error_tol = blasted::parseOptionalPetscCmd_real("-error_tolerance", 2*DBL_EPSILON);
	//const double iters_tol = parseOptionalPetscCmd_real("-iters_tolerance", 1e-2);

	if(rank == 0)
		printf("  Test tolerance = %g.\n", error_tol);

	if(testtype == "compare_its" || testtype == "issame") {
		assert(fabs((double)refkspiters - avgkspiters)/refkspiters <= error_tol);
	}
	else if(testtype == "upper_bound_its") {
		assert(refkspiters > avgkspiters);
	}

	Vec diff;
	int ierr = VecDuplicate(u, &diff); CHKERRQ(ierr);
	ierr = VecSet(diff, 0); CHKERRQ(ierr);
	ierr = VecWAXPY(diff, -1.0, u, uref); CHKERRQ(ierr);
	PetscScalar diffnorm, refnorm;
	ierr = VecNorm(uref, NORM_2, &refnorm); CHKERRQ(ierr);
	ierr = VecNorm(diff, NORM_2, &diffnorm); CHKERRQ(ierr);
	ierr = VecDestroy(&diff); CHKERRQ(ierr);

	if(rank == 0) {
		printf("Difference in solutions = %g.\n", diffnorm);
		printf("Relative difference = %g.\n", diffnorm/refnorm);
	}
	fflush(stdout);
	if(testtype == "compare_error" || testtype == "issame")
		assert(diffnorm/refnorm <= error_tol);

	return 0;
}

static int pretendModifyMatrix(const Mat A)
{
	int firstrow, lastrow;
	int ierr = MatGetOwnershipRange(A, &firstrow, &lastrow); CHKERRQ(ierr);

	ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	return ierr;
}

int runComparisonVsPetsc_cpp(const DiscreteLinearProblem lp)
{
	int rank = 0;
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

	char testtype[PETSCOPTION_STR_LEN];
	PetscBool set = PETSC_FALSE;
	int ierr = PetscOptionsGetString(NULL,NULL,"-test_type",testtype, PETSCOPTION_STR_LEN, &set);
	CHKERRQ(ierr);
	if(!set) {
		printf("Test type not set; testing issame.\n");
		strcpy(testtype,"issame");
	}

	set = PETSC_FALSE;
	PetscInt cmdnumruns;
	ierr = PetscOptionsGetInt(NULL,NULL,"-num_runs",&cmdnumruns,&set); CHKERRQ(ierr);
	const int nruns = set ? cmdnumruns : 1;

	//----------------------------------------------------------------------------------

	// compute reference solution using a preconditioner from PETSc

	Vec uref;
	ierr = VecDuplicate(lp.uexact, &uref); CHKERRQ(ierr);

	KSP kspref;
	ierr = KSPCreate(PETSC_COMM_WORLD, &kspref);
	KSPSetType(kspref, KSPRICHARDSON);
	KSPRichardsonSetScale(kspref, 1.0);
	KSPSetOptionsPrefix(kspref, "ref_");
	KSPSetFromOptions(kspref);

	ierr = KSPSetOperators(kspref, lp.lhs, lp.lhs); CHKERRQ(ierr);

	ierr = KSPSolve(kspref, lp.b, uref); CHKERRQ(ierr);

	KSPConvergedReason ref_ksp_reason;
	ierr = KSPGetConvergedReason(kspref, &ref_ksp_reason); CHKERRQ(ierr);
	assert(ref_ksp_reason > 0);

	PetscInt refkspiters;
	ierr = KSPGetIterationNumber(kspref, &refkspiters);
	PetscReal errnormref = 0;
	ierr = compute_difference_norm(uref,lp.uexact,&errnormref); CHKERRQ(ierr);

	if(rank==0) {
		printf("Ref run: error = %.16f\n", errnormref);
	}

	ierr = KSPDestroy(&kspref); CHKERRQ(ierr);

	// run the solve to be tested as many times as requested

	int avgkspiters = 0;

	Vec u;
	ierr = VecDuplicate(lp.uexact, &u); CHKERRQ(ierr);
	ierr = VecSet(u, 0); CHKERRQ(ierr);

	ierr = MatSetOption(lp.lhs, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE); CHKERRQ(ierr);

	KSP ksp;

	ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
	ierr = KSPSetType(ksp, KSPRICHARDSON); CHKERRQ(ierr);
	ierr = KSPRichardsonSetScale(ksp, 1.0); CHKERRQ(ierr);

	// Options MUST be set before setting shell routines
	ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
	// Operators MUST be set before extracting sub KSPs
	ierr = KSPSetOperators(ksp, lp.lhs, lp.lhs); CHKERRQ(ierr);

	for(int irun = 0; irun < nruns; irun++)
	{
		if(rank == 0)
			printf("Run %d:\n", irun);

		Vec urun;
		ierr = VecDuplicate(lp.uexact, &urun); CHKERRQ(ierr);

		ierr = pretendModifyMatrix(lp.lhs); CHKERRQ(ierr);

		ierr = KSPSolve(ksp, lp.b, urun); CHKERRQ(ierr);

		// post-process
		int kspiters; PetscReal rnorm;
		ierr = KSPGetIterationNumber(ksp, &kspiters); CHKERRQ(ierr);
		avgkspiters += kspiters;

		KSPConvergedReason ksp_reason;
		ierr = KSPGetConvergedReason(ksp, &ksp_reason); CHKERRQ(ierr);
		if(rank == 0) {
			printf("  KSP converged reason = %d.\n", ksp_reason);
			fflush(stdout);
		}
		assert(ksp_reason > 0);

		if(rank == 0) {
			ierr = KSPGetResidualNorm(ksp, &rnorm); CHKERRQ(ierr);
			printf(" KSP residual norm = %f\n", rnorm);
		}

		PetscReal errnormrun = 0;
		ierr = compute_difference_norm(urun, lp.uexact, &errnormrun); CHKERRQ(ierr);

		ierr = VecAXPY(u, 1.0, urun); CHKERRQ(ierr);

		if(rank == 0) {
			printf("Test run:\n");
			printf(" error: %.16f\n", errnormrun);
			printf(" log error: %f\n", log10(errnormrun));
		}

		ierr = VecDestroy(&urun); CHKERRQ(ierr);
	}

	ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

	avgkspiters = avgkspiters/(double)nruns;
	ierr = VecScale(u, 1.0/nruns); CHKERRQ(ierr);

	ierr = compareSolverWithRef(refkspiters, avgkspiters, uref, u); CHKERRQ(ierr);

	ierr = VecDestroy(&u); CHKERRQ(ierr);
	ierr = VecDestroy(&uref); CHKERRQ(ierr);

	return ierr;
}

int getBlockSize(const Mat A)
{
	int bs = 0;
	int ierr = MatGetBlockSize(A, &bs);
	if(ierr != 0)
		throw blasted::Petsc_exception(ierr);

	// Check matrix type and adjust block size
	const char *mattype;
	ierr = MatGetType(A, &mattype);
	if(ierr != 0)
		throw blasted::Petsc_exception(ierr);
	if(!strcmp(mattype, MATSEQAIJ) || !strcmp(mattype, MATSEQAIJMKL) ||
	   !strcmp(mattype, MATMPIAIJ) || !strcmp(mattype, MATMPIAIJMKL) )
	{
		printf(" Matrix is a scalar SR type.\n");
		bs = 1;
	}
	else if(!strcmp(mattype, MATSEQBAIJ) || !strcmp(mattype, MATSEQBAIJMKL) ||
	        !strcmp(mattype, MATMPIBAIJ) || !strcmp(mattype, MATMPIBAIJMKL) )
		printf(" Matrix is a block SR type.\n");
	else
		throw std::runtime_error("Unsupported matrix type " + std::string(mattype));
	return bs;
}

void set_blasted_sweeps(const int nbswp, const int naswp)
{
	// add option
	std::string value = std::to_string(nbswp) + "," + std::to_string(naswp);
	int ierr = PetscOptionsSetValue(NULL, "-blasted_async_sweeps", value.c_str());
	blasted::petsc_throw(ierr);

	// Check
	int checksweeps[2];
	int nmax = 2;
	PetscBool set = PETSC_FALSE;
	ierr = PetscOptionsGetIntArray(NULL,NULL,"-blasted_async_sweeps",checksweeps,&nmax,&set);
	blasted::petsc_throw(ierr);
	if(checksweeps[0] != nbswp || checksweeps[1] != naswp)
		throw std::runtime_error("Async sweeps not set properly!");
}

}
