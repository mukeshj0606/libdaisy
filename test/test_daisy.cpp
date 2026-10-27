#include <cmath>
#include <cstdio>
#include <vector>
#include "daisy/daisy.h"

// straightforward re-implementation of the ORIGINAL four-pass, per-element
// branch formulation of the bilinear histogram accumulation. used as an
// independent reference for the fused/branchless version under test.
static void bilinear_accumulate_reference( float* histogram, int hist_size, int shift,
                                           const float* A, const float* B, const float* C, const float* D,
                                           float w0, float w1, float w2, float w3 )
{
   for( int h=0; h<hist_size; h++ )
      histogram[h]  = ( h+shift < hist_size ) ? w0*A[h+shift] : w0*A[h+shift-hist_size];
   for( int h=0; h<hist_size; h++ )
      histogram[h] += ( h+shift < hist_size ) ? w1*C[h+shift] : w1*C[h+shift-hist_size];
   for( int h=0; h<hist_size; h++ )
      histogram[h] += ( h+shift < hist_size ) ? w2*B[h+shift] : w2*B[h+shift-hist_size];
   for( int h=0; h<hist_size; h++ )
      histogram[h] += ( h+shift < hist_size ) ? w3*D[h+shift] : w3*D[h+shift-hist_size];
}

// verify the fused bilinear accumulation matches the original across every
// valid circular shift (0..hist_size). hist_size = 8 matches the daisy default
// (hist_quantization). source arrays are padded to 2*hist_size so the wrapped
// index range is in bounds, mirroring the real hcube layout.
static int test_bilinear_accumulate_matches_reference()
{
   const int hist_size = 8;
   std::vector<float> A(2*hist_size), B(2*hist_size);
   std::vector<float> C(2*hist_size), D(2*hist_size);
   for( int i=0; i<2*hist_size; i++ ) {
      A[i] = 1.0f + (float)i;
      B[i] = 100.0f + (float)i;
      C[i] = -3.0f * (float)i;
      D[i] = 0.5f * (float)i - 2.0f;
   }
   const float w0 = 0.2f, w1 = 0.3f, w2 = 0.15f, w3 = 0.35f;

   int failures = 0;
   for( int shift=0; shift<=hist_size; shift++ ) {
      std::vector<float> got(hist_size, -999.0f), ref(hist_size, -999.0f);
      daisy_detail::bilinear_accumulate_histogram( &got[0], hist_size, shift,
                                                   &A[0], &B[0], &C[0], &D[0], w0, w1, w2, w3 );
      bilinear_accumulate_reference( &ref[0], hist_size, shift,
                                     &A[0], &B[0], &C[0], &D[0], w0, w1, w2, w3 );
      for( int h=0; h<hist_size; h++ ) {
         // same additive association as the original; only fma contraction can
         // differ, which is within a small relative tolerance.
         const double tol = 1e-4 * ( fabs( (double)ref[h] ) + 1.0 );
         if( fabs( (double)got[h] - (double)ref[h] ) > tol ) {
            printf("  shift %d, h %d: got %.9g expected %.9g\n", shift, h, got[h], ref[h]);
            failures++;
         }
      }
   }
   return failures;
}

int main()
{
   int failures = test_bilinear_accumulate_matches_reference();
   printf("[test] BilinearAccumulateMatchesReference: %s\n", failures ? "FAILED" : "passed");
   return failures ? 1 : 0;
}
