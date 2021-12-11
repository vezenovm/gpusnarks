/*****************************************************************************
 Implementation of Fast Fourier Transformation on Finite Elements
 *****************************************************************************
 * @author     Marius van der Wijden
 * Copyright [2019] [Marius van der Wijden]
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include "fft_kernel.h"
#include "device_field.h"
#include "device_field_operators.h"

#define LOG_NUM_THREADS 10
#define NUM_THREADS (1 << LOG_NUM_THREADS)
#define LOG_CONSTRAINTS 16
#define CONSTRAINTS (1 << LOG_CONSTRAINTS)

#define CUDA_CALL( call )               \
{                                       \
cudaError_t result = call;              \
if ( cudaSuccess != result )            \
    std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
}

__device__ __forceinline__
size_t bitreverse(size_t n, const size_t l)
{
    return __brevll(n) >> (64ull - l); 
}

__device__ uint32_t _mod [SIZE] = { 610172929, 1586521054, 752685471, 3818738770, 
    2596546032, 1669861489, 1987204260, 1750781161, 3411246648, 3087994277, 
    4061660573, 2971133814, 2707093405, 2580620505, 3902860685, 134068517, 
    1821890675, 1589111033, 1536143341, 3086587728, 4007841197, 270700578, 764593169, 115910};

__device__ __forceinline__
size_t k_adicity(size_t k, size_t n)
{
    size_t r = 0;
    while (n > 1)
    {
        if (n % k == 0) {
            r += 1;
            n /= k;
        } else {
            return r;
        }
    }
    return r;
}

__device__ __forceinline__
size_t pow_int(size_t base, size_t exp)
{
    size_t res = 1;
    while (exp)
    {
        if (exp & 1)
        {
            res *= base;
        }
        exp /= 2;
        base *= base;
    }
    return res;
}

__device__ __forceinline__
size_t mixed_radix_FFT_permute(size_t two_adicity, size_t q_adicity, size_t q, size_t N, size_t i)
{
    /*
    This is the permutation obtained by splitting into
    2 groups two_adicity times and then
    q groups q_adicity many times
    It can be efficiently described as follows
    write 
    i = 2^0 b_0 + 2^1 b_1 + ... + 2^{two_adicity - 1} b_{two_adicity - 1} 
      + 2^two_adicity ( x_0 + q^1 x_1 + .. + q^{q_adicity-1} x_{q_adicity-1})
    We want to return
    j = b_0 (N/2) + b_1 (N/ 2^2) + ... + b_{two_adicity-1} (N/ 2^two_adicity)
      + x_0 (N / 2^two_adicity / q) + .. + x_{q_adicity-1} (N / 2^two_adicity / q^q_adicity) */
    size_t res = 0;
    size_t shift = N;

    for (size_t s = 0; s < two_adicity; ++s)
    {
      shift = shift / 2;
      res += (i % 2) * shift;
      i = i / 2;
    }

    for (size_t s = 0; s < q_adicity; ++s)
    {
        shift = shift / q;
        res += (i % q) * shift;
        i = i / q;
    }

    return res;
}

template<typename FieldT>
__global__ void _basic_serial_mixed_radix_FFT(FieldT *out, FieldT *field)
{
    // Conceptually, this FFT first splits into 2 sub-arrays two_adicity many times,
    // and then splits into q sub-arrays q_adicity many times.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t log_m = LOG_CONSTRAINTS;
    const size_t length = CONSTRAINTS;
    const size_t block_length = 1ul << (log_m - LOG_NUM_THREADS) ;
    const size_t startidx = idx * block_length;
    assert (CONSTRAINTS == 1ul<<log_m);
    if(startidx > length)
        return;
    FieldT a [block_length];
    const size_t n = CONSTRAINTS;
    // For now leave q set as five or else otherwise the method will crash 
    const size_t q = 5;

    const size_t q_adicity = k_adicity(q, n);
    const size_t q_part = pow_int(q, q_adicity);

    const size_t two_adicity = k_adicity(2, n);
    const size_t two_part = 1u << two_adicity;

    // printf("q_part: %d\n", q_part);
    // printf("two_part: %d\n", two_part);
    //if (n != q_part * two_part) assert("expected n == (1u << logn)");

    //Do not remove log2f(n), otherwise register overflow
    size_t block = block_length, logb = log2f(block);
    assert (block == (1u << logb));
    size_t m = 1; // invariant: m = 2^{s-1}
    //printf("q_adicity: %d\n", q_adicity);
    if (q_adicity > 0)
    {
        // If we're using the other radix, we have to do two things differently than in the radix 2 case.
        // 1. Applying the index permutation is a bit more complicated. It isn't an involution
        // (like it is in the radix 2 case) so we need to remember which elements we've moved as we go along
        // and can't use the trick of just swapping when processing the first element of a 2-cycle.
        //
        // 2. We need to do q_adicity many merge passes, each of which is a bit more complicated than the
        // specialized q=2 case.

        // Applying the permutation
        //std::vector<bool> seen(n, false);
        //printf("got here");
        bool seen[block_length]{};
        for (size_t k = 0; k < block; ++k)
        {
            size_t i = k;
            FieldT a_i = field[i];
            while (! seen[i])
            {
                size_t dest = mixed_radix_FFT_permute(two_adicity, q_adicity, q, block, i);
                printf("dest: %d ", dest);
                FieldT a_dest = field[dest];
                field[dest] = a_i;
                a[dest] = a_i;

                seen[i] = true;

                a_i = a_dest;
                i = dest;
            }
        }
        __syncthreads();

        
        for(size_t i = block_length; i--;) {
            fields::Scalar::print(a[i]);
        }
        //printf("got here 2");
        FieldT omega_q = FieldT(_mod) ^ (block/q);

        //const FieldT omega_q = omega ^ (n / q);
        //std::vector<FieldT> qth_roots(q);
        FieldT qth_roots [q];
        qth_roots[0] = FieldT::one();
        for (size_t i = 1; i < idx % q; ++i) {
            qth_roots[i] = qth_roots[i-1] * omega_q;
        }

        //std::vector<FieldT> terms(q-1);
        const FieldT omega_num_cpus = FieldT(_mod) ^ NUM_THREADS;
        FieldT terms[q-1];
        // Doing the q_adicity passes.
        for (size_t s = 1; s <= q_adicity; ++s)
        {
            const FieldT w_m = omega_num_cpus^(block / (q*m));
            //const FieldT w_m = FieldT(_mod) ^ (n / (q*m));
            for (size_t k = 0; k < logb; k += q*m)
            {
                FieldT w_j = FieldT::one(); // w_j is omega_m ^ j
                for (size_t j = 0; j < m; ++j)
                {
                    FieldT base_term = a[k+j];
                    FieldT w_j_i = w_j;
                    //printf("a size: %d", sizeof(a)/sizeof(a[0]));
                    for (size_t i = 1; i < q; ++i) {
                        //size_t id = (i + (s<<(log_m - LOG_NUM_THREADS))) % (1u << log_m);
                        //printf("k: %d, j: %d, i: %d; ", k, j, i%10);
                        //printf("term: %d; ", terms[i-1]);
                        terms[i-1] = w_j_i * a[k+j + i*m];
                        w_j_i = w_j_i * w_j;
                    }

                    for (size_t i = 0; i < q; ++i) {
                        a[k+j + i*m] = base_term;
                        for (size_t l = 1; l < q; ++l) {
                          a[k+j + i*m] = a[k+j + i*m] + (qth_roots[(i*l) % q] * terms[l-1]);
                        }
                    }
                    w_j = w_j * w_m;
                }
            };
            m = m * q;
            printf("m: %d, ", m);
        }
    }
    else
    {
        //TODO algorithm is non-deterministic because of padding
        FieldT omega_j = FieldT(_mod);
        omega_j = omega_j ^ idx; // pow
        FieldT omega_step = FieldT(_mod);
        omega_step = omega_step ^ (idx << (log_m - LOG_NUM_THREADS));
        FieldT elt = FieldT::one();

        for (size_t i = 0; i < 1ul<<(log_m - LOG_NUM_THREADS); ++i)
        {
            const size_t ri = bitreverse(i, logb);
            for (size_t s = 0; s < NUM_THREADS; ++s)
            {
                // invariant: elt is omega^(j*idx)
                size_t id = (i + (s<<(log_m - LOG_NUM_THREADS))) % (1u << log_m);
                FieldT tmp = field[id];
                tmp = tmp * elt;
                if (s != 0) tmp = tmp + a[ri];
                a[ri] = tmp;
                elt = elt * omega_step;
            }
            elt = elt * omega_j;
        }
    }
    const FieldT omega_num_cpus = FieldT(_mod) ^ NUM_THREADS;
    m = 1; // invariant: m = 2^{s-1}
    for (size_t s = 1; s <= logb; ++s)
    {
        // w_m is 2^s-th root of unity now
        const FieldT w_m = omega_num_cpus^(block/(2*m));
        for (size_t k = 0; k < block; k += 2*m)
        {
            FieldT w = FieldT::one();
            for (size_t j = 0; j < m; ++j)
            {
                const FieldT t = w;
                w = w * a[k+j+m];
                a[k+j+m] = a[k+j] - t;
                a[k+j] = a[k+j] + t;
                w = w * w_m;
            }
        }
        m = m << 1;
    }
    for (size_t j = 0; j < 1ul<<(log_m - LOG_NUM_THREADS); ++j)
    {
        if(((j << LOG_NUM_THREADS) + idx) < length)
            out[(j<<LOG_NUM_THREADS) + idx] = a[j];
    }
}

template<typename FieldT>  
__global__ void cuda_fft(FieldT *out, FieldT *field) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t log_m = LOG_CONSTRAINTS;
    const size_t length = CONSTRAINTS;
    const size_t block_length = 1ul << (log_m - LOG_NUM_THREADS) ;
    const size_t startidx = idx * block_length;
    assert (CONSTRAINTS == 1ul<<log_m);
    if(startidx > length)
        return;
    FieldT a [block_length];

    //TODO algorithm is non-deterministic because of padding
    FieldT omega_j = FieldT(_mod);
    omega_j = omega_j ^ idx; // pow
    FieldT omega_step = FieldT(_mod);
    omega_step = omega_step ^ (idx << (log_m - LOG_NUM_THREADS));
    
    FieldT elt = FieldT::one();
    //Do not remove log2f(n), otherwise register overflow
    size_t n = block_length, logn = log2f(n);
    assert (n == (1u << logn));
    for (size_t i = 0; i < 1ul<<(log_m - LOG_NUM_THREADS); ++i)
    {
        const size_t ri = bitreverse(i, logn);
        for (size_t s = 0; s < NUM_THREADS; ++s)
        {
            // invariant: elt is omega^(j*idx)
            size_t id = (i + (s<<(log_m - LOG_NUM_THREADS))) % (1u << log_m);
            FieldT tmp = field[id];
            tmp = tmp * elt;
            if (s != 0) tmp = tmp + a[ri];
            a[ri] = tmp;
            elt = elt * omega_step;
        }
        elt = elt * omega_j;
    }

    const FieldT omega_num_cpus = FieldT(_mod) ^ NUM_THREADS;
    size_t m = 1; // invariant: m = 2^{s-1}
    for (size_t s = 1; s <= logn; ++s)
    {
        // w_m is 2^s-th root of unity now
        const FieldT w_m = omega_num_cpus^(n/(2*m));
        for (size_t k = 0; k < n; k += 2*m)
        {
            FieldT w = FieldT::one();
            for (size_t j = 0; j < m; ++j)
            {
                const FieldT t = w;
                w = w * a[k+j+m];
                a[k+j+m] = a[k+j] - t;
                a[k+j] = a[k+j] + t;
                w = w * w_m;
            }
        }
        m = m << 1;
    }
    for (size_t j = 0; j < 1ul<<(log_m - LOG_NUM_THREADS); ++j)
    {
        if(((j << LOG_NUM_THREADS) + idx) < length)
            out[(j<<LOG_NUM_THREADS) + idx] = a[j];
    }
}

template<typename FieldT> 
void best_fft (std::vector<FieldT> &a, const FieldT &omg)
{
	int cnt;
    cudaGetDeviceCount(&cnt);
    printf("CUDA Devices: %d, Field size: %lu, Field count: %lu\n", cnt, sizeof(FieldT), a.size());
    assert(a.size() == CONSTRAINTS);

    size_t blocks = NUM_THREADS / 256 + 1;
    size_t threads = NUM_THREADS > 256 ? 256 : NUM_THREADS;
    printf("NUM_THREADS %u, blocks %lu, threads %lu \n",NUM_THREADS, blocks, threads);

    FieldT *in;
    CUDA_CALL( cudaMalloc((void**)&in, sizeof(FieldT) * a.size()); )
    CUDA_CALL( cudaMemcpy(in, (void**)&a[0], sizeof(FieldT) * a.size(), cudaMemcpyHostToDevice); )

    FieldT *out;
    CUDA_CALL( cudaMalloc(&out, sizeof(FieldT) * a.size()); )
    cuda_fft<FieldT> <<<blocks,threads>>>(out, in);
        
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    CUDA_CALL( cudaMemcpy((void**)&a[0], out, sizeof(FieldT) * a.size(), cudaMemcpyDeviceToHost); )

    CUDA_CALL( cudaDeviceSynchronize();)
}

template<typename FieldT> 
void mixed_fft (std::vector<FieldT> &a, const FieldT &omg)
{
	int cnt;
    cudaGetDeviceCount(&cnt);
    printf("CUDA Devices: %d, Field size: %lu, Field count: %lu\n", cnt, sizeof(FieldT), a.size());
    assert(a.size() == CONSTRAINTS);

    size_t blocks = NUM_THREADS / 256 + 1;
    size_t threads = NUM_THREADS > 256 ? 256 : NUM_THREADS;
    printf("NUM_THREADS %u, blocks %lu, threads %lu \n",NUM_THREADS, blocks, threads);

    FieldT *in;
    CUDA_CALL( cudaMalloc((void**)&in, sizeof(FieldT) * a.size()); )
    CUDA_CALL( cudaMemcpy(in, (void**)&a[0], sizeof(FieldT) * a.size(), cudaMemcpyHostToDevice); )

    FieldT *out;
    CUDA_CALL( cudaMalloc(&out, sizeof(FieldT) * a.size()); )
    _basic_serial_mixed_radix_FFT<FieldT> <<<blocks,threads>>>(out, in);
        
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    CUDA_CALL( cudaMemcpy((void**)&a[0], out, sizeof(FieldT) * a.size(), cudaMemcpyDeviceToHost); )

    CUDA_CALL( cudaDeviceSynchronize();)
}

//List with all templates that should be generated
template void best_fft(std::vector<fields::Scalar> &v, const fields::Scalar &omg);
template void mixed_fft(std::vector<fields::Scalar> &a, const fields::Scalar &omg);

