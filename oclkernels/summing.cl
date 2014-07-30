/* Copyright 2014
 *  Afshin Haidari
 *  Steve Novakov
 *  Jeff Taylor
 * 
 * Summing kernel.  This kernel adds up the paths tracked by finished particles
 * into a single global buffer.
 */

#include "attrs.h"

__kernel void PdfSum(
  struct particle_attrs attrs,  /* RO */
  global uint* local_pdfs,
  global uint* global_pdf
)
{
  uint index = get_global_id(0) * attrs.sample_ny * attrs.sample_nz
             + get_global_id(1) * attrs.sample_nz
             + get_global_id(2);
  int running_total = 0;
  int i;
  int num_entries = attrs.sample_nx * attrs.sample_ny * attrs.sample_nz;

  if (get_global_id(0) >= attrs.sample_nx)
    return;
  if (get_global_id(1) >= attrs.sample_ny)
    return;
  if (get_global_id(2) >= attrs.sample_nz)
    return;

  for (i = 0; i < attrs.num_wg; ++i)
    running_total += local_pdfs[index + num_entries * i];

  /* TODO(jeff): at some point we'll want to make this a 64-bit count to support
   * >2billion particles.  This also implies initializing it in host code, and
   * calling this kernel once every 2**31 interpolates (or so). */
  global_pdf[index] = running_total;
}
