// Copyright 2014
//  Afshin Haidari
//  Steve Novakov
//  Jeff Taylor
//
//  Global structs.  These are shared between oclptxhandler and the kernels, so
//  if they are changed here, they need to be changed in oclptxhandler as well.

#ifndef ATTRS_H_
#define ATTRS_H_

#include "rng.h"

// Completion codes
#define BREAK_BRAIN_MASK  1
#define BREAK_CURV        2
#define BREAK_LOOPCHECK   3
#define BREAK_INVALID     4
#define BREAK_TERMINATE   5
#define BREAK_EXCLUSION   6
#define BREAK_MAXSTEPS    7
#define BREAK_INIT        8
#define STILL_FINISHED    9
#define ANISO_BREAK       10

// Struct representing the persistent state of a single particle.
struct particle_data
{
  rng_t rng; //RW
  float3 position;
  float3 dr;
} __attribute__((aligned(64)));

// Struct full of useful constants.
struct particle_attrs
{
  float4 brain_mask_dim;
  int steps_per_kernel;
  int max_steps;
  int min_steps;
  int particles_per_side;
  uint sample_nx;
  uint sample_ny;
  uint sample_nz;
  uint num_samples;
  float curvature_threshold;
  uint n_waypoint_masks;
  float step_length;
  uint lx;  // Loopcheck sizes
  uint ly;
  uint lz;
  int fibst;
  int randfib;
  float fibthresh;
  int num_wg;
} __attribute__((aligned(16)));

#endif  // ATTRS_H_
