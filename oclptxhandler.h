/* Copyright 2014
 *  Afshin Haidari
 *  Steve Novakov
 *  Jeff Taylor
 */

#ifndef OCLPTXHANDLER_H_
#define OCLPTXHANDLER_H_

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "customtypes.h"

class OclPtxHandler{
 public:
  struct particle_data
  {
    cl_ulong8 rng;
    cl_float4 position;
    cl_float4 dr;
  } __attribute__((aligned(64)));

  struct particle_attrs
  {
    cl_float4 brain_mask_dim;
    cl_int steps_per_kernel;
    cl_int max_steps;
    cl_int min_steps;
    cl_int particles_per_side;
    cl_uint sample_nx;
    cl_uint sample_ny;
    cl_uint sample_nz;
    cl_uint num_samples;
    cl_float curvature_threshold;
    cl_uint n_waypoint_masks;
    cl_float step_length;
    cl_uint lx;  // Loopcheck sizes
    cl_uint ly;
    cl_uint lz;
    cl_int fibst;
    cl_int randfib;
    cl_float fibthresh;
    cl_int num_wg;
  } __attribute__((aligned(16)));

  OclPtxHandler() {}
  void Init(
      cl::Context *cc,
      cl::CommandQueue *cq,
      cl::Kernel* ptx_kernel,
      cl::Kernel* sum_kernel,
      struct particle_attrs *attrs,
      FILE *path_dump_fd,
      EnvironmentData *env_dat,
      cl::Buffer *global_pdf);
  ~OclPtxHandler();

  int particles_per_side();

  // Write a single particle
  void WriteParticle(struct particle_data *data, int offset);
  // Run Kernel asyncronously
  void RunKernel(int side);
  // Read the "completion" buffer back into the vector pointed to by ret.
  void ReadStatus(int offset, int count, cl_ushort *ret);
  // Dump path to file.
  void DumpPath(int offset, int count);
  // Aggregate the paths.
  void RunSumKernel();

 private:
  size_t ParticleSize();
  void InitParticles();
  void SetInterpArg(int pos, cl::Buffer *buf);
  void SetSumArg(int pos, cl::Buffer *buf);
  void RunInterpKernel(int side);

  struct particle_attrs attrs_;

  // OpenCL Interface
  cl::Context* context_;
  cl::CommandQueue* cq_;
  cl::Kernel* ptx_kernel_;
  cl::Kernel* sum_kernel_;
  size_t wg_size_;

  // Particle Data
  cl::Buffer *gpu_data_;  // Type particle_data
  cl::Buffer *gpu_sets_;  // Type rbtree
  cl::Buffer *gpu_complete_;  // Type ushort array
  cl::Buffer *gpu_waypoints_;
  cl::Buffer *gpu_exclusion_;
  cl::Buffer *gpu_loopcheck_;
  cl::Buffer *gpu_global_pdf_;
  cl::Buffer *gpu_local_pdf_;

  // Debug Data
  cl::Buffer *gpu_path_;  // Type ulong
  cl::Buffer *gpu_step_count_; // Type ushort

  FILE *path_dump_fd_;
  bool first_time_;

  // TODO(jeff) avoid keeping this pointer, instead keep pointer to real env.
  EnvironmentData * env_dat_;
};

#endif  // OCLPTXHANDLER_H_

