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

#define DEBUG

#include "device_field.h"
#include "device_field_operators.h"
#include <assert.h>
#include <gmp.h>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <ctime>  

namespace fields{

    enum operand {add, substract, mul, pow};

    void testAdd()
    {
        printf("Addition test: ");
        fields::Scalar f1(1234);
        fields::Scalar f2(1234);
        fields::Scalar result(2468);
        f1 =  f1 + f2;
        Scalar::testEquality(f1, result);
        printf("successful\n");
    }

    void test_subtract()
    {
        printf("_subtraction test: ");
        fields::Scalar f1(1234);
        fields::Scalar f2(1234);
        f1 = f1 - f2;
        Scalar::testEquality(f1, fields::Scalar::zero());
        fields::Scalar f3(1235);
        f3 = f3 - f2;
        Scalar::testEquality(f3,fields::Scalar::one());
        printf("successful\n");
    }

    void testMultiply()
    {
        printf("Multiply test: ");
        fields::Scalar f1(1234);
        fields::Scalar f2(1234);
        f1 = f1 * f2;
        Scalar::testEquality(f1, fields::Scalar(1522756));
        f1 = f1 * f2;
        Scalar::testEquality(f1, fields::Scalar(1879080904));
        f1 = f1 * f2;
        Scalar::testEquality(f1, fields::Scalar(3798462992)); 
        fields::Scalar f3(1234);
        f3 = f3 * f3;
        Scalar::testEquality(f3, fields::Scalar(1522756));
        printf("successful\n");
    }

    void testModulo()   
    {
        printf("Modulo test: ");
        fields::Scalar f1(uint32_t(0));
        fields::Scalar f2(1234);
        
        fields::Scalar f3();
        printf("successful\n");
    }

    void testPow()
    {
        printf("Scalar::pow test: ");
        fields::Scalar f1(2);
        Scalar::pow(f1, 0);
        Scalar::testEquality(f1, fields::Scalar::one());
        fields::Scalar f2(2);
        Scalar::pow(f2, 2);
        Scalar::testEquality(f2, fields::Scalar(4));
        Scalar::pow(f2, 10);
        Scalar::testEquality(f2, fields::Scalar(1048576));
        fields::Scalar f3(2);
        fields::Scalar f4(1048576);
        Scalar::pow(f3, 20);
        Scalar::testEquality(f3, f4);
        printf("successful\n");

    }

    void testConstructor()
    {
        printf("Constructor test: ");
        fields::Scalar f3(1);
        Scalar::testEquality(f3, fields::Scalar::one());
        fields::Scalar f4;
        Scalar::testEquality(f4, fields::Scalar::zero());
        fields::Scalar f5(uint32_t(0));
        Scalar::testEquality(f5, fields::Scalar::zero());

        fields::Scalar f1;
        fields::Scalar f2(1234);
        f1 = f1 +  fields::Scalar(1234);
        Scalar::testEquality(f1, f2);
        uint32_t tmp [SIZE];
        for(int i = 0; i < SIZE; i++)
            tmp[i] = 0;
        tmp[SIZE -1 ] = 1234;
        fields::Scalar f6(tmp);
        Scalar::testEquality(f6, f2);
        printf("successful\n");
    }

    

    void setMod()
    {  
        assert(SIZE == 24); 
        mpz_t n;
        mpz_init(n);
        mpz_set_str(n, "41898490967918953402344214791240637128170709919953949071783502921025352812571106773058893763790338921418070971888253786114353726529584385201591605722013126468931404347949840543007986327743462853720628051692141265303114721689601", 0);
        size_t size = SIZE;
        mpz_export(_mod, &size, 0, sizeof(_mod[0]), 0, 0, n);
        mpz_import(n, size, 0, sizeof(_mod[0]), 0, 0, _mod);
        gmp_printf ("Mod_prime:  [%Zd] \n",n);        
        /*
        assert(SIZE == 24);
        for(int i = 0; i < SIZE; i ++)
        {
            _mod[i] = 0;
        }
        _mod[SIZE - 3] = 1;*/
    }

    void operate(fields::Scalar & f1, fields::Scalar const f2, int const op)
    {
        switch(op){
            case 0:
                f1 = f1 + f2; break;
            case 1:
                f1 = f1 - f2; break;
            case 2:
                f1 = f1 * f2; break;
            case 3:
                Scalar::pow(f1, (f2.im_rep[SIZE - 1] & 65535)); 
                break;
            default: break;
        } 
    }

    void operate(mpz_t mpz1, mpz_t const mpz2, mpz_t const mod, int const op)
    {
        switch(op){
            case 0:
                mpz_add(mpz1, mpz1, mpz2);
                mpz_mod(mpz1, mpz1, mod);
                break;
            case 1:
                mpz_sub(mpz1, mpz1, mpz2);
                mpz_mod(mpz1, mpz1, mod); 
                break;
            case 2:
                mpz_mul(mpz1, mpz1, mpz2);
                mpz_mod(mpz1, mpz1, mod);
                break;
            case 3:
                mpz_t pow;
                mpz_init(pow);
                mpz_set_ui(pow, 65535);
                mpz_and(pow, mpz2, pow);
                mpz_powm(mpz1, mpz1, pow, mod);
                mpz_clear(pow);
                break;
            default: break;
        }
    }

    void toMPZ(mpz_t ret, fields::Scalar f)
    {
        mpz_init(ret);
        mpz_import(ret, SIZE, 1, sizeof(uint32_t), 0, 0, f.im_rep);   
    }

    void compare(fields::Scalar f1, fields::Scalar f2, mpz_t mpz1, mpz_t mpz2, mpz_t mod, int op)
    {
        mpz_t tmp1;
        mpz_init_set(tmp1, mpz1);
        operate(f1, f2, op);
        operate(mpz1, mpz2, mod, op);
        mpz_t tmp;
        toMPZ(tmp, f1);
        if(mpz_cmp(tmp, mpz1) != 0){
            printf("Missmatch: ");
            gmp_printf ("t: %d [%Zd] %d [%Zd] \n",omp_get_thread_num(), tmp1, op, mpz2);
            gmp_printf ("t: %d CPU: [%Zd] GPU: [%Zd] \n",omp_get_thread_num() , mpz1, tmp);
            Scalar::printScalar(f1);
            assert(!"error");
        }
        mpz_clear(tmp1);
        mpz_clear(tmp);
    }

    void calculateModPrime()
    {
        mpz_t one, minus1, n_prime, n, m, mod, two;
        mpz_init(one);
        mpz_init(minus1);
        mpz_init(n_prime);
        mpz_init(n);
        mpz_init(m);
        mpz_init(one);
        mpz_init(mod);
        mpz_init(two);

        mpz_set_ui(two, 2);
        mpz_set_str(n, "41898490967918953402344214791240637128170709919953949071783502921025352812571106773058893763790338921418070971888253786114353726529584385201591605722013126468931404347949840543007986327743462853720628051692141265303114721689601", 0);
        uint exp = SIZE*32; //log2(n) rounded up
        mpz_pow_ui(m, two, exp); //2^log2(n)

        mpz_set_si(minus1, -1);
        int i = mpz_invert(n_prime, n, m); // n' = n^-1
        uint32_t rop[SIZE];
        uint32_t _mod[SIZE];
        size_t size = SIZE;
        mpz_export(rop, &size, 0, 32, 0, 0, n_prime);
        mpz_export(_mod, &size, 0, 32, 0, 0, n);
        gmp_printf ("Mod_prime:  [%Zd] %d %u %u %u \n",n_prime, i, rop[0], _mod[0], _mod[SIZE -1]);        
        //36893488147419103231
    }

    void fuzzTest()
    {
        printf("Fuzzing test: ");
        
        size_t i_step = 12345671;
        size_t k_step = 76543210;
        auto start = std::chrono::system_clock::now();
    
        //#pragma omp parallel for
        for(size_t i = 0; i < 4294967295; i = i + i_step)
        {
            if(omp_get_thread_num() == 0){
                auto end = std::chrono::system_clock::now();
                std::chrono::duration<double> elapsed_seconds = end-start;

                printf("%f%% %d sec \n", (float(i) / 4294967295) * omp_get_num_threads(), (int)elapsed_seconds.count());
            }
            mpz_t a, b, mod;
            mpz_init(a);
            mpz_init(b);
            mpz_init(mod);
            //mpz_set_str(mod, "18446744073709551616", 0);
            mpz_set_str(mod, "41898490967918953402344214791240637128170709919953949071783502921025352812571106773058893763790338921418070971888253786114353726529584385201591605722013126468931404347949840543007986327743462853720628051692141265303114721689601", 0);
            mpz_set_ui(b, i);
            fields::Scalar f2(i);
            for(size_t k = 0; k < 4294967295; k = k + k_step)
            {
                for(size_t z = 0; z <= 3; z++ )
                {
                    mpz_set_ui(a, k);
                    fields::Scalar f1(k);
                    compare(f1,f2,a,b,mod,z);
                }
            }
            mpz_clear(a);
            mpz_clear(b);
            mpz_clear(mod);
        }
        printf("successful\n");
    }
}

int main(int argc, char** argv)
{
    fields::calculateModPrime();
    
    fields::setMod();
    fields::testConstructor();
    fields::testAdd();
    fields::test_subtract();
    fields::testMultiply();
    fields::testPow();
    fields::fuzzTest();
    printf("\nAll tests successful\n");
    return 0;
}



