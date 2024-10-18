/* This program is public domain. */

#ifdef __cplusplus
extern "C" {
#endif

// MSVC 2008 doesn't define erf()
#if defined(_MSC_VER) && _MSC_VER<=1600
  #define const
  #define __LITTLE_ENDIAN
  #include "erf.c"
#endif

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#ifdef SGI
#include <ieeefp.h>
#endif

#define SQRT1_2   0.707106781186547524400844362104849039  /* 1/sqrt(2) */

// value should have length NP * (NI + 1)
void
build_profile(size_t NZ, size_t NP, size_t NI,
              const double z[], /* length NZ */
              const double offset[],  /* length NI */
              const double roughness[], /* length NI */
              const double contrast[], /* length NP * NI */ 
              const double initial_value[], /* length NP */
              double profile[] /* length NP * NZ (num profiles * num z) */
              )
{
    size_t pi;
    size_t zi;
    size_t ii;
    size_t zi_offset;
    size_t pi_offset;
    size_t NPZ = NP * NZ;

    double offset_i;
    double sigma_i;
    double z_i;
    double contrast_i;
    double blended;
    double delta;
    double value;

    // set all elements to initial_value for all profiles
    zi_offset = 0;
    for (pi=0; pi < NP; pi++) {
        value = initial_value[pi];
        for (zi=0; zi < NZ; zi++) {
            profile[zi_offset++] = value;
        }
    }

    for (ii=0; ii < NI; ii++) {
        offset_i = offset[ii];
        sigma_i = roughness[ii];
        
        for (zi=0; zi < NZ; zi++) {
            z_i = z[zi];
            if (sigma_i <= 0.0) {
                blended = (z_i >= offset_i) ? 1.0 : 0.0;
            }
            else {
                blended = 0.5 * erf(SQRT1_2 * (z_i - offset_i) / sigma_i) + 0.5;
            }

            pi_offset = ii;
            zi_offset = zi;
            for (pi=0; pi < NP; pi++) {
                contrast_i = contrast[pi_offset];
                delta = contrast_i * blended;
                profile[zi_offset] += delta;
                pi_offset += NI;
                zi_offset += NZ;
            }
        }
    }
}

#ifdef __cplusplus
}
#endif