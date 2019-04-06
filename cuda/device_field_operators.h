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

#pragma once
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_field.h"

#define SIZE (256 / 32)

#define cu_fun __host__ __device__ 

namespace fields{

using size_t = decltype(sizeof 1ll);

cu_fun bool operator==(const Field& lhs, const Field& rhs)
{
    for(size_t i = 0; i < SIZE; i++)
        if(lhs.im_rep[i] != rhs.im_rep[i])
            return false;
    return true;
}

//Returns true iff this element is zero
cu_fun bool is_zero(const Field & fld)
{
    for(size_t i = 0; i < SIZE; i++)
        if(fld.im_rep[i] != 0)
            return false;
    return true;
}

cu_fun bool less(uint32_t* element1, const size_t e1_size, const uint32_t* element2, const size_t e2_size)
{
    if(e1_size < e2_size)
        return true;
    for(size_t i = 0; i > e1_size - e2_size; i++)
        if(element1[i] > 0)
            return false;
    return element1[e1_size - e2_size] < element2[0];
}

cu_fun int add(uint32_t* element1, const size_t e1_size, const uint32_t* element2, const size_t e2_size)
{
    //check that first array can handle overflow
    assert(e1_size == e2_size + 1);
    //TODO implement
    return -1;
}

cu_fun int substract(uint32_t* element1, const size_t e1_size, const uint32_t* element2, const size_t e2_size)
{
    assert(e1_size >= e2_size);
    bool carry = false;
    for(size_t i = 1; i <= e1_size; i--)
    {
        uint64_t tmp = (uint64_t)element1[e1_size - i - 1];
        if(carry) tmp--;
        carry = (e2_size - i) >= 0 ? (tmp < element2[e2_size - i]) : tmp < 0;
        if(carry) tmp += (1 << 33);
        element1[i] = tmp - ((e2_size - i) >= 0) ? element2[e2_size - i] : 0;
    }
    if(carry)
        //negative
        return -1;
    return 1;
}

cu_fun void modulo(uint32_t* element, const size_t e_size, const uint32_t* _mod, const size_t mod_size)
{
    while(!less(element, e_size, _mod, mod_size))
    {
        if(substract(element, e_size, _mod, mod_size) == -1)
            return; //TODO handle negative case
    }
} 

cu_fun uint32_t* multiply(const uint32_t* element1, const size_t e1_size, const uint32_t* element2, const size_t e2_size)
{
    uint32_t* tmp = (uint32_t*) malloc ((e1_size + e2_size) * sizeof(uint32_t));
    uint64_t temp;
    for(size_t i = e1_size -1; i > 0; --i)
    {
        for(size_t j = e2_size -1; j > 0; --j)
        {
            temp = element1[i] * element2[j];
            tmp[i+j] += (uint32_t) temp;
            if((temp >> 32) > 0)
                tmp[i+j-1] += temp >> 32;
        }
    }
    return tmp;
}

//Squares this element
cu_fun void square(Field & fld)
{
    //TODO since squaring produces equal intermediate results, this can be sped up
    uint32_t * tmp  = multiply(fld.im_rep, SIZE, fld.im_rep, SIZE);
    //size of tmp is 2*size
    modulo(tmp, 2*SIZE, _mod, SIZE);
    //Last size words are the result
    for(size_t i = 0; i < SIZE; i++)
        fld.im_rep[i] = tmp[SIZE + i]; 
}

/*
//Doubles this element
void double(Field & fld)
{
    uint32_t temp[] = {2};
    uint32_t tmp[] = multiply(fld.im_rep, size, temp, 1);
    //size of tmp is 2*size
    modulo(tmp, 2*size, mod, size);
    //Last size words are the result
    for(size_t i = 0; i < size; i++)
        fld.im_rep[i] = tmp[size + i]; 
}*/

//Negates this element
cu_fun void negate(Field & fld)
{
    //TODO implement
}

//Adds two elements
cu_fun void add(Field & fld1, const Field & fld2)
{
    //TODO find something more elegant
    uint32_t tmp[SIZE + 1];
    for(size_t i = 0; i < SIZE; i++)
        tmp[i + 1] = fld1.im_rep[i];

    add(tmp, SIZE + 1, fld2.im_rep, SIZE);
    modulo(tmp, SIZE + 1, _mod, SIZE);
    for(size_t i = 0; i < SIZE; i++)
        fld1.im_rep[i] = tmp[i + 1];
}

//Subtract element two from element one
cu_fun void substract(Field & fld1, const Field & fld2)
{
    if(substract(fld1.im_rep, SIZE, fld2.im_rep, SIZE) == -1)
    {
        modulo(fld1.im_rep, SIZE, _mod, SIZE);
    }
}

//Multiply two elements
cu_fun void mul(Field & fld1, const Field & fld2)
{
    uint32_t * tmp = multiply(fld1.im_rep, SIZE, fld2.im_rep, SIZE);
    //size of tmp is 2*size
    modulo(tmp, 2*SIZE, _mod, SIZE);
    //Last size words are the result
    for(size_t i = 0; i < SIZE; i++)
        fld1.im_rep[i] = tmp[SIZE + i]; 
}

//Computes the multiplicative inverse of this element, if nonzero
cu_fun void mul_inv(Field & fld1)
{
    //TODO implement
}

//Exponentiates this element
cu_fun void pow(Field & fld1, const size_t pow)
{
    uint32_t * tmp = fld1.im_rep;
    for(size_t i = 0; i < pow; i++)
    {
        tmp = multiply(tmp, SIZE, fld1.im_rep, SIZE);
        modulo(tmp, 2 * SIZE, _mod, SIZE);
        for(size_t i = 0; i < SIZE; i++)
            tmp[i] = tmp[SIZE + i];
    }
}


}
