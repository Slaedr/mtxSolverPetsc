/** \file
 * \brief Implementation of some basic kernels like spmv
 * \author Aditya Kashi
 * 
 * This file is part of BLASTed.
 *   BLASTed is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   BLASTed is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with BLASTed.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "matvecs.hpp"

namespace blasted {

template <typename mscalar, typename mindex, int bs, StorageOptions stor>
void BLAS_BSR<mscalar,mindex,bs,stor>
::matrix_apply(const SRMatrixStorage<mscalar,mindex>&& mat,
               const scalar *const xx, scalar *const __restrict yy)
{
	using Blk = Block_t<scalar,bs,stor>;
	using Seg = Segment_t<scalar,bs>;
	const Blk *data = reinterpret_cast<const Blk*>(&mat.vals[0]);
	const Seg *x = reinterpret_cast<const Seg*>(xx);
	Seg *y = reinterpret_cast<Seg*>(yy);

#ifdef USE_OPENMP
#pragma omp parallel for default(shared)
#endif
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		y[irow] = Vector<scalar>::Zero(bs);

		// loop over non-zero blocks of this block-row
		for(index jj = mat.browptr[irow]; jj < mat.browptr[irow+1]; jj++)
		{
			// multiply the blocks with corresponding sub-vectors
			const index jcol = mat.bcolind[jj];
			y[irow].noalias() += data[jj] * x[jcol];
		}
	}
}

template <typename mscalar, typename mindex, int bs, StorageOptions stor>
void BLAS_BSR<mscalar,mindex,bs,stor>
::gemv3(const SRMatrixStorage<mscalar,mindex>&& mat,
        const scalar a, const scalar *const __restrict xx,
        const scalar b, const scalar *const yy, scalar *const zz)
{
	using Blk = Block_t<scalar,bs,stor>;
	using Seg = Segment_t<scalar,bs>;
	const Blk *data = reinterpret_cast<const Blk*>(&mat.vals[0]);
	const Seg *x = reinterpret_cast<const Seg*>(xx);
	const Seg *y = reinterpret_cast<const Seg*>(yy);
	Seg *z = reinterpret_cast<Seg*>(zz);

#ifdef USE_OPENMP
#pragma omp parallel for default(shared)
#endif
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		z[irow] = b * y[irow];

		// loop over non-zero blocks of this block-row
		for(index jj = mat.browptr[irow]; jj < mat.browptr[irow+1]; jj++)
		{
			const index jcol = mat.bcolind[jj];
			z[irow].noalias() += a * data[jj] * x[jcol];
		}
	}
}

template <typename mscalar, typename mindex>
void BLAS_CSR<mscalar,mindex>::matrix_apply(const SRMatrixStorage<mscalar,mindex>&& mat,
                                            const scalar *const xx, scalar *const __restrict yy) 
{
#ifdef USE_OPENMP
#pragma omp parallel for default(shared)
#endif
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		yy[irow] = 0;

		for(index jj = mat.browptr[irow]; jj < mat.browptr[irow+1]; jj++)
		{
			yy[irow] += mat.vals[jj] * xx[mat.bcolind[jj]];
		}
	}
}

template <typename mscalar, typename mindex>
void BLAS_CSR<mscalar,mindex>::gemv3(const SRMatrixStorage<mscalar,mindex>&& mat,
                                     const scalar a, const scalar *const __restrict xx, 
                                     const scalar b, const scalar *const yy, scalar *const zz)
{
#ifdef USE_OPENMP
#pragma omp parallel for default(shared)
#endif
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		zz[irow] = b * yy[irow];

		for(index jj = mat.browptr[irow]; jj < mat.browptr[irow+1]; jj++)
		{
			zz[irow] += a * mat.vals[jj] * xx[mat.bcolind[jj]];
		}
	}
}

// Instantiations

template struct BLAS_BSR<double,int,3,ColMajor>;
template struct BLAS_BSR<double,int,4,ColMajor>;
template struct BLAS_BSR<double,int,5,ColMajor>;
template struct BLAS_BSR<double,int,7,ColMajor>;

template struct BLAS_BSR<double,int,3,RowMajor>;
template struct BLAS_BSR<double,int,4,RowMajor>;
template struct BLAS_BSR<double,int,7,RowMajor>;

template struct BLAS_BSR<const double,const int,3,ColMajor>;
template struct BLAS_BSR<const double,const int,4,ColMajor>;
template struct BLAS_BSR<const double,const int,5,ColMajor>;
template struct BLAS_BSR<const double,const int,7,ColMajor>;

template struct BLAS_BSR<const double,const int,3,RowMajor>;
template struct BLAS_BSR<const double,const int,4,RowMajor>;
template struct BLAS_BSR<const double,const int,7,RowMajor>;

template struct BLAS_CSR<double,int>;
template struct BLAS_CSR<const double,const int>;

}
