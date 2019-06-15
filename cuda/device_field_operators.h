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

#include "device_field.h"

#define SIZE (256 / 32)

#ifndef DEBUG
#define cu_fun __host__ __device__ 
#else
#define cu_fun
#include <assert.h>
#include <malloc.h>
#include <cstring>
#endif

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

cu_fun void set_mod(const Field& f)
{
    for(size_t i = 0; i < SIZE; i++)
        _mod[i] = f.im_rep[i];
}

//Returns true if the first element is less than the second element
cu_fun bool less(const uint32_t* element1, const size_t e1_size, const uint32_t* element2, const size_t e2_size)
{
    assert(e1_size >= e2_size);
    size_t diff = e1_size - e2_size;
    for(size_t i = 0; i < diff; i++)
        if(element1[i] > 0)
            return false;
    for(size_t i = 0; i < e2_size; i++)
        if(element1[i + diff] > element2[i])
            return false;
        else if(element1[i + diff] < element2[i])
            return true;
    return false;
}

// Returns the carry, true if there was a carry, false otherwise
// Takes a sign, true if negative
cu_fun bool add(bool sign, uint32_t* element1, const size_t e1_size, const uint32_t* element2, const size_t e2_size)
{
    assert(e1_size >= e2_size);
    bool carry = false;
    for(size_t i = 1; i <= e1_size; i++)
    {
        uint64_t tmp = (uint64_t)element1[e1_size - i];
        if(carry) tmp++;
        element1[e1_size - i] = tmp + (uint64_t)element2[e1_size - i];
        carry = (tmp >> 32) > 0;
    }
    return sign? 0: carry;
}

// Returns the carry, true if the resulting number is negative
cu_fun bool subtract(uint32_t* element1, const size_t e1_size, bool carry,  const uint32_t* element2, const size_t e2_size)
{
    assert(e1_size >= e2_size);
    for(size_t i = 1; i <= e1_size; i++)
    {
        uint64_t tmp = (uint64_t)element1[e1_size - i];
        bool underflow = (tmp == 0);
        if(carry) tmp--;
        carry = (e2_size >= i) ? (tmp < element2[e2_size - i]) : underflow;
        if(carry) tmp += ((uint64_t)1 << 33);
        element1[e1_size - i] = tmp - element2[e2_size - i];
    }
    return carry;
}

cu_fun void ciosMontgomeryMultiply(uint32_t * result, 
const uint32_t* a, const size_t a_size, 
const uint32_t* b, const size_t b_size, 
const uint32_t* n, const size_t n_size,
const uint64_t m_prime)
{
    uint64_t temp;
    for(size_t i = 0; i < a_size; i++)
    {
        uint32_t carry = 0;
        for(size_t j = 0; j < a_size; j++)
        {
            temp = result[j];
            temp += (uint64_t)a[j] * (uint64_t)b[i];
            temp += carry;
            result[j] = (uint32_t)temp;
            carry = temp >> 32;
        }
        temp = result[a_size] + carry;
        result[a_size] = (uint32_t) temp;
        result[a_size + 1] = temp >> 32;
        uint64_t m = (result[0] * m_prime) & 4294967296;
        temp = result[0] + (uint64_t)m * n[0]; 
        carry = temp >> 32;
        for(size_t j = 0; j < a_size; j++)
        {
            temp = result[j];
            temp += (uint64_t)m * (uint64_t)n[j];
            temp += carry;
            result[j - 1] = (uint32_t)temp;
            carry = temp >> 32;
        }
        temp = result[a_size] + carry;
        result[a_size - 1] = (uint32_t) temp;
        result[a_size] = result[a_size + 1] + temp >> 32;
    }
    uint32_t t[SIZE];
    memcpy(t, result, a_size);
    int64_t stemp = 0;
    int carry = 0;
    for(size_t i = 0; i < a_size; i++)
    {
        stemp = (int64_t)result[i];
        stemp -= (int64_t)n[i];
        stemp -= carry;
        if(stemp < 0)
        {
            t[i] = (uint32_t) (stemp + 4294967296);
            carry = 1;
        }
        else
        {
            t[i] = stemp;
            carry = 0;
        }
    }
    stemp = t[a_size] - carry;
    t[a_size] = (uint32_t)stemp;
    if(stemp < 0)
        memcpy(result, t, a_size);
}

cu_fun void modulo(uint32_t* element, const size_t e_size, const uint32_t* mod, const size_t mod_size, bool carry)
{
    if(less(element, e_size, mod, mod_size))
        return;
    printf("tick");

    uint32_t tmp[SIZE * 2];
    memset(tmp, 0, (SIZE * 2) * sizeof(uint32_t));

    ciosMontgomeryMultiply(tmp + 1, Field::one().im_rep, SIZE, element, SIZE, _mod, SIZE, 4294967296L);
    for(size_t i = 0; i < SIZE; i++)
        element[i] = tmp[i];
} 

cu_fun bool multiply(uint32_t * result, const uint32_t* element1, const size_t e1_size, const uint32_t* element2, const size_t e2_size)
{
    bool carry = false;
    uint64_t temp;
    for(size_t i = 0; i < e2_size; i++)
    {
        uint32_t carry = 0;
        for(size_t j = 0; j < e1_size; j++)
        {
            temp = result[i+j + 1];
            temp += (uint64_t)element1[j] * (uint64_t)element2[i];
            temp += carry;
            result[i+j + 1] = (uint32_t)temp;
            carry = temp >> 32;
        }
        result[i + e1_size + 1] = carry;
    }
    return carry;
}

//Squares this element
cu_fun void square(Field & fld)
{
    //TODO since squaring produces equal intermediate results, this can be sped up
    uint32_t tmp[SIZE * 2];
    memset(tmp, 0, SIZE * 2 * sizeof(uint32_t));
    bool carry = multiply(tmp, fld.im_rep, SIZE, fld.im_rep, SIZE);
    //size of tmp is 2*size
    modulo(tmp, 2*SIZE, _mod, SIZE, carry);
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
    bool carry = add(false, fld1.im_rep, SIZE, fld2.im_rep, SIZE);
    if(carry || less(_mod, SIZE, fld1.im_rep, SIZE))
        subtract(fld1.im_rep, SIZE, false, _mod, SIZE);
}

//Subtract element two from element one
cu_fun void subtract(Field & fld1, const Field & fld2)
{
    if(less(fld1.im_rep, SIZE, fld2.im_rep, SIZE))
        add(true, fld1.im_rep, SIZE, _mod, SIZE);
    subtract(fld1.im_rep, SIZE, false, fld2.im_rep, SIZE);
}

//Multiply two elements
cu_fun void mul(Field & fld1, const Field & fld2)
{
    uint32_t tmp[SIZE * 2];
    memset(tmp, 0, (SIZE * 2) * sizeof(uint32_t));
    
    ciosMontgomeryMultiply(tmp + 1, fld1.im_rep, SIZE, fld2.im_rep, SIZE, _mod, SIZE, 4294967296L);
    for(size_t i = 0; i < SIZE; i++)
        fld1.im_rep[i] = tmp[i];
}

//Exponentiates this element
cu_fun void pow(Field & fld1, const size_t pow)
{
    if(pow == 0) 
    {
        fld1 = Field::one();
        return;
    }

    if(pow == 1)
    {
        return;
    }

    uint32_t tmp[SIZE * 2];
    uint32_t temp[SIZE];

    for(size_t i = 0; i < SIZE; i++)
        temp[i] = fld1.im_rep[i];

    for(size_t i = 0; i < pow - 1; i++)
    {
        memset(tmp, 0, (SIZE * 2) * sizeof(uint32_t));
        
        ciosMontgomeryMultiply(tmp + 1, fld1.im_rep, SIZE, temp, SIZE, _mod, SIZE, 4294967296L);
        for(size_t i = 0; i < SIZE; i++)
            fld1.im_rep[i] = tmp[i];
    }
}
}
