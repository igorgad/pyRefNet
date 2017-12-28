
#include "tmwtypes.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"


#define BLOCKDIM 8
#define A(i,j) A[(i) + (j)*numrows]

__device__ int get_index_x (int ncols, int index ) { 
  if (index == -1)  {
    index = blockDim.x * blockIdx.x + threadIdx.x;
  } else {
    index += gridDim.x;
  }

  if (index >= ncols) index = -1;

  return index;
}

__device__ int get_index_y (int nrows, int index ) { 
  if (index == -1)  {
    index = blockDim.y * blockIdx.y + threadIdx.y;
  } else {
    index += gridDim.y;
  }

  if (index >= nrows) index = -1;

  return index;
}

__device__ int get_index_z (int depth, int index ) { 
  if (index == -1)  {
    index = blockDim.z * blockIdx.z + threadIdx.z;
  } else {
    index += gridDim.z;
  }

  if (index >= depth) index = -1;

  return index;
}


__device__ float Gaussian (float x, float y, float sigma) {
  return (1/sqrt(2*M_PI*sigma)) * exp((-pow(x - y,2)) / (2*pow(sigma,2)));
}

__device__ float Gaussian_prime (float x, float y, float sigma) {
  return ( -(x - y) / ((pow(sigma,3))*sqrt(2*M_PI)) ) *  exp((-pow(x - y,2)) / (2*pow(sigma,2)));
}


__global__ void CC_reduction ( float *redOut, const float *xline, const float *yline, const float *wxline, const float *wyline, int mvalue, float sigma, int linesize) {

  extern __shared__ float gSM[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + tid;
  unsigned int blockSize = blockDim.x;
  unsigned int gridSize = gridDim.x;

  //printf ("CC_reduction tid %d i %d blocksize %d gridsize %d\n", tid, i, blockSize, gridSize);

  while (i < linesize) {
    if (i - mvalue < 0 || i - mvalue > linesize) {
      gSM[tid] = 0;
    } else {
      gSM[tid] = Gaussian(xline[i] * wxline[i], yline[i-mvalue] * wyline[i-mvalue], sigma);      
    }

    i += gridSize;
  }

  __syncthreads();

  if (blockSize >= 512) { if (tid < 256) { gSM[tid] += gSM[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { gSM[tid] += gSM[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid < 64) { gSM[tid] += gSM[tid + 64]; } __syncthreads(); }

  if (tid < 32) {
    if (blockSize >= 64) gSM[tid] += gSM[tid + 32];
    if (blockSize >= 32) gSM[tid] += gSM[tid + 16];
    if (blockSize >= 16) gSM[tid] += gSM[tid + 8];
    if (blockSize >= 8) gSM[tid] += gSM[tid + 4];
    if (blockSize >= 4) gSM[tid] += gSM[tid + 2];
    if (blockSize >= 2) gSM[tid] += gSM[tid + 1];
  }

   if (tid == 0) redOut[blockIdx.x] = gSM[0] * (1 / linesize - abs(mvalue));

}

__global__ void ACm_kernel( float *out, const float *x, const float *y, const float *wx, const float *wy,  const int *marray, float sigma, int msize, int ncols, int nrows, int depth) {

  int idm = get_index_x(msize, -1);
  int idy = get_index_y(nrows, -1);
  int idz = get_index_z(depth, -1);


  while(idz >= 0) {
    while (idy >= 0) {
      while(idm >= 0) {
        int m = marray[idm];

        float *xp = (float*) &x[idy*ncols + idz*nrows*ncols];
        float *yp = (float*) &y[idy*ncols + idz*nrows*ncols];
        float *wxp = (float*) &wx[idm*ncols];
        float *wyp = (float*) &wy[idm*ncols];

        float *outp = (float*) &out[idm + idy*msize + idz*nrows*msize];
       
        CC_reduction<<<1,ncols>>>(outp, xp, yp, wxp, wyp, m, sigma, ncols);

       // out[idm + idy*msize + idz*nrows*msize] = redOut[0] * (1 / ncols - abs(m));

        idm = get_index_x(msize, idm);
      }
      idy = get_index_y (nrows, idy);
    }
    idz = get_index_z (depth, idz);
  }
}


// __global__ void ACm_kernel( float *out, const float *x, const float *y, const float *wx, const float *wy,  const int *marray, float sigma, int msize, int ncols, int nrows, int depth) {
//   float sum = 0;
//   int i = 0;
//   int idm = get_index_x(msize, -1);
//   int idy = get_index_y(nrows, -1);
//   int idz = get_index_z(depth, -1);
//   int m = marray[idm];
//   int cn = 1;

//   while(idz >= 0) {
//     while (idy >= 0) {
//       while(idm >= 0) {
//         sum = 0;
//         cn = 1;

//         //printf ("idx %d, idy %d, idz %d\n", idm,idy,idz);

//         for (i=m; i < ncols; i++) {
//           if (i < 0 || i-m > ncols) {
//             continue;
//           }

//           sum += Gaussian (x[i + idy*ncols + idz*nrows*ncols] * wx[i + idm*ncols], y[i-m + idy*ncols + idz*nrows*ncols] * wy[i-m + idm*ncols], sigma);

//           cn = cn + 1;
//         }

//         out[idm + idy*msize + idz*nrows*msize] = ( 1/((float)cn) ) * sum;

//         idm = get_index_x(msize, idm);
//         m = marray[idm];

//       }
//       idy = get_index_y (nrows, idy);
//     }
//     idz = get_index_z (depth, idz);
//   }
// }


/**
 * MEX gateway
 */
void mexFunction(int /* nlhs */, mxArray *plhs[], int nrhs, mxArray const *prhs[]) {
    
    
    // Receive: gpu_inx, gpu_iny, gpu_wx, gpu_wy, gpu_m, single(sigma)
    // Output a CCC Mat
    

    // Initialize the MathWorks GPU API.
    mxInitGPU();

    //if (nrhs!=11) {
    //    mexErrMsgIdAndTxt("ccc_forward:invalid input arguments", errMsg);
    //}


    float sigma = mxGetScalar(prhs[6]);
    int msize = mxGetScalar(prhs[7]);
    int N = mxGetScalar(prhs[8]) ;
    int nwin = mxGetScalar(prhs[9]);
    int bsize = mxGetScalar(prhs[10]);
    
    mxGPUArray const * xmat = mxGPUCreateFromMxArray(prhs[1]);
    mxGPUArray const * ymat = mxGPUCreateFromMxArray(prhs[2]);
    mxGPUArray const * wxmat = mxGPUCreateFromMxArray(prhs[3]);
    mxGPUArray const * wymat = mxGPUCreateFromMxArray(prhs[4]);
    mxGPUArray const * marray = mxGPUCreateFromMxArray(prhs[5]);

    float *xp = (float*) mxGPUGetDataReadOnly(xmat);
    float *yp = (float*) mxGPUGetDataReadOnly(ymat);
    float *wxp = (float*) mxGPUGetDataReadOnly(wxmat);
    float *wyp = (float*) mxGPUGetDataReadOnly(wymat);
    int *mp = (int*) mxGPUGetDataReadOnly(marray);

    mxGPUArray *out = mxGPUCopyFromMxArray(prhs[0]);
    float *outp = (float*) mxGPUGetData(out);

    dim3 const dimBlock(msize/4, nwin/4, bsize/4);
    dim3 const dimThread(2, 2, 2);

    printf ("mex function calling kernel N %d nwin %d bsize %d msize %d sigma %.2f\n", N, nwin, bsize, msize,sigma);
    
    ACm_kernel<<<dimBlock, dimThread>>>(outp, xp, yp, wxp, wyp, mp, sigma, msize, N, nwin, bsize);

    plhs[0] = mxGPUCreateMxArrayOnGPU(out);
    
}
    
    