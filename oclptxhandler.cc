/*  Copyright (C) 2014
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */

#include "oclptxhandler.h"

#include <cstdlib>

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif


static void die(int reason)
{
  if (CL_MEM_OBJECT_ALLOCATION_FAILURE == reason)
  {
    puts("Ran out of device memory while allocating particle buffers.  "
         "It should be possible to fix this by lowering memrisk "
         "(eg --memrisk=.9) and rerunning.");
    exit(-1);
  }
  else
    abort();
}

void OclPtxHandler::Init(
  cl::Context *cc,
  cl::CommandQueue *cq,
  cl::Kernel *ptx_kernel,
  cl::Kernel *sum_kernel,
  struct OclPtxHandler::particle_attrs *attrs,
  FILE *path_dump_fd,
  EnvironmentData *env_dat,
  cl::Buffer *global_pdf)
{
  context_ = cc;
  cq_ = cq;
  ptx_kernel_ = ptx_kernel;
  sum_kernel_ = sum_kernel;
  first_time_ = 1;
  path_dump_fd_ = path_dump_fd;
  env_dat_ = env_dat;
  attrs_ = *attrs;

  gpu_global_pdf_ = global_pdf;

  // TODO(jeff): Check if we can actually allocate buffers this big.
  attrs_.particles_per_side = env_dat_->dynamic_mem_left / ParticleSize() / 2;
  printf("Allocating %i particles\n", attrs_.particles_per_side * 2);

  InitParticles();
}

size_t OclPtxHandler::ParticleSize()
{
  size_t size = 0;
  size += sizeof(struct particle_data);

  size += sizeof(cl_ushort);  // complete
  size += sizeof(cl_ushort);  // step_count

  size += 32768;  // position rbtree

  if (env_dat_->save_paths)
    size += attrs_.steps_per_kernel * sizeof(cl_float4);

  if (0 < env_dat_->n_waypts)
    size += attrs_.n_waypoint_masks * sizeof(cl_ushort);

  if (env_dat_->exclusion_mask)
    size += sizeof(cl_ushort);

  if (env_dat_->loopcheck)
    size += attrs_.lx * attrs_.ly * attrs_.lz * sizeof(float4);

  printf("Particle size %liB\n", size);

  return size;
}

void OclPtxHandler::InitParticles()
{
  cl_int ret;

  gpu_data_ = new cl::Buffer(
      *context_,
      CL_MEM_READ_WRITE,
      2 * attrs_.particles_per_side * sizeof(struct particle_data));
  if (!gpu_data_)
    abort();

  gpu_sets_ = new cl::Buffer(
      *context_,
      CL_MEM_READ_WRITE,
      2 * attrs_.particles_per_side * 32768);  // 32768 == sizeof(struct rbtree)
  if (!gpu_sets_)
    abort();

  gpu_complete_ = new cl::Buffer(
      *context_,
      CL_MEM_READ_WRITE,
      2 * attrs_.particles_per_side * sizeof(cl_ushort));
  if (!gpu_complete_)
    abort();

  if (env_dat_->save_paths)
  {
    gpu_path_ = new cl::Buffer(
        *context_,
        CL_MEM_WRITE_ONLY,
        2 * attrs_.particles_per_side *
          attrs_.steps_per_kernel * sizeof(cl_float4));
    if (!gpu_path_)
      abort();
  }
  else
    gpu_path_ = NULL;

  gpu_step_count_ = new cl::Buffer(
      *context_,
      CL_MEM_READ_WRITE,
      2 * attrs_.particles_per_side * sizeof(cl_ushort));
  if (!gpu_step_count_)
    abort();

  if (0 < env_dat_->n_waypts)
  {
    gpu_waypoints_ = new cl::Buffer(
        *context_,
        CL_MEM_READ_WRITE,
        2 * attrs_.particles_per_side * attrs_.n_waypoint_masks * sizeof(cl_ushort));
    if (!gpu_waypoints_)
      abort();
  }
  else
    gpu_waypoints_ = NULL;

  if (env_dat_->exclusion_mask)
  {
    gpu_exclusion_ = new cl::Buffer(
        *context_,
        CL_MEM_READ_WRITE,
        2 * attrs_.particles_per_side * sizeof(cl_ushort));
    if (!gpu_exclusion_)
      abort();
  }
  else
    gpu_exclusion_ = NULL;

  if (env_dat_->loopcheck)
  {
    gpu_loopcheck_ = new cl::Buffer(
        *context_,
        CL_MEM_READ_WRITE,
        2 * attrs_.particles_per_side * attrs_.lx * attrs_.ly * attrs_.lz * sizeof(float4));
    if (!gpu_loopcheck_)
      abort();
  }
  else
    gpu_loopcheck_ = NULL;

  // Initialize "completion" buffer.
  cl_ushort *temp_completion = new cl_ushort[2*attrs_.particles_per_side];
  for (int i = 0; i < 2 * attrs_.particles_per_side; ++i)
    temp_completion[i] = 8;  // BREAK_INIT

  ret = cq_->enqueueWriteBuffer(
      *gpu_complete_,
      true,
      0,
      2 * attrs_.particles_per_side * sizeof(cl_ushort),
      reinterpret_cast<void*>(temp_completion));
  if (CL_SUCCESS != ret)
    die(ret);

  delete[] temp_completion;
}

OclPtxHandler::~OclPtxHandler()
{
  delete gpu_data_;
  delete gpu_sets_;
  delete gpu_complete_;
  if (gpu_waypoints_)
    delete gpu_waypoints_;
  if (gpu_exclusion_)
    delete gpu_exclusion_;
  if (gpu_loopcheck_)
    delete gpu_loopcheck_;
  // we let OclEnv delete gpu_global_pdf_
}

int OclPtxHandler::particles_per_side()
{
  return attrs_.particles_per_side;
}

void OclPtxHandler::WriteParticle(
    struct particle_data *data,
    int offset)
{
  // Note: locking.  This function is technically thread-unsafe, but that
  // shouldn't matter because threading is set up for only one thread to ever
  // call these methods.
  cl_int ret;
  cl_ushort zero = 0;
  assert(offset < 2 * attrs_.particles_per_side);

  if (NULL != path_dump_fd_)
    fprintf(path_dump_fd_, "%i:%f,%f,%fn\n",
        offset,
        data->position.s[0],
        data->position.s[1],
        data->position.s[2]);

  // Write particle_data
  ret = cq_->enqueueWriteBuffer(
      *gpu_data_,
      true,
      offset * sizeof(struct particle_data),
      sizeof(struct particle_data),
      reinterpret_cast<void*>(data));
  if (CL_SUCCESS != ret)
  {
    puts("Write failed!");
    die(ret);
  }

  // gpu_complete_ = 0
  ret = cq_->enqueueWriteBuffer(
      *gpu_complete_,
      true,
      offset * sizeof(cl_ushort),
      sizeof(cl_ushort),
      reinterpret_cast<void*>(&zero));
  if (CL_SUCCESS != ret)
  {
    puts("Write failed!");
    die(ret);
  }

  // step_count = 0
  ret = cq_->enqueueWriteBuffer(
      *gpu_step_count_,
      true,
      offset * sizeof(cl_ushort),
      sizeof(cl_ushort),
      reinterpret_cast<void*>(&zero));
  if (CL_SUCCESS != ret)
  {
    puts("Write failed!");
    die(ret);
  }

  // Initialize particle loopcheck
  if (gpu_loopcheck_)
  {
    int loopcheck_entries_per_particle = (attrs_.lx
                              * attrs_.ly
                              * attrs_.lz);
    cl_float4 *temp_loopcheck = new cl_float4[loopcheck_entries_per_particle];
    cl_float4 zero_f4;
    zero_f4.s[0] = 0.;
    zero_f4.s[1] = 0.;
    zero_f4.s[2] = 0.;
    zero_f4.s[3] = 0.;
    for (int i = 0; i < loopcheck_entries_per_particle; ++i)
      temp_loopcheck[i] = zero_f4;

    ret = cq_->enqueueWriteBuffer(
        *gpu_loopcheck_,
        true,
        offset * loopcheck_entries_per_particle * sizeof(cl_float4),
        loopcheck_entries_per_particle * sizeof(cl_float4),
        temp_loopcheck);
    if (CL_SUCCESS != ret)
      die(ret);

    delete[] temp_loopcheck;
  }

  //TODO(jeff): Can we allocate in Init?
  if (gpu_waypoints_)
  {
    cl_ushort *temp_waypoints = new cl_ushort[attrs_.n_waypoint_masks];
    for (cl_uint i = 0; i < attrs_.n_waypoint_masks; ++i)
      temp_waypoints[i] = 0;

    ret = cq_->enqueueWriteBuffer(
        *gpu_waypoints_,
        true,
        offset * attrs_.n_waypoint_masks * sizeof(cl_ushort),
        attrs_.n_waypoint_masks * sizeof(cl_ushort),
        temp_waypoints);
    if (CL_SUCCESS != ret)
      die(ret);

    delete[] temp_waypoints;
  }

  if (gpu_exclusion_)
  {
    cl_ushort temp_zero = 0;

    ret = cq_->enqueueWriteBuffer(
      *gpu_exclusion_,
      true,
      offset * sizeof(cl_ushort),
      sizeof(cl_ushort),
      &temp_zero
    );
    if (CL_SUCCESS != ret)
      die(ret);
  }
}

void OclPtxHandler::SetInterpArg(int pos, cl::Buffer *buf)
{
  if (buf)
    ptx_kernel_->setArg(pos, *buf);
  else
    ptx_kernel_->setArg(pos, NULL);
}

void OclPtxHandler::SetSumArg(int pos, cl::Buffer *buf)
{
  if (buf)
    sum_kernel_->setArg(pos, *buf);
  else
    sum_kernel_->setArg(pos, NULL);
}

void OclPtxHandler::RunInterpKernel(int side)
{
  cl_int ret;
  cl::NDRange particles_to_compute(attrs_.particles_per_side);
  cl::NDRange particle_offset(attrs_.particles_per_side * side);

  ptx_kernel_->setArg(
      0,
      sizeof(struct OclPtxHandler::particle_attrs),
      reinterpret_cast<void*>(&attrs_));
  SetInterpArg(1, gpu_data_);
  SetInterpArg(2, gpu_sets_);
  SetInterpArg(3, gpu_path_);
  SetInterpArg(4, gpu_step_count_);
  SetInterpArg(5, gpu_complete_);
  SetInterpArg(6, gpu_global_pdf_);
  SetInterpArg(7, gpu_waypoints_);
  SetInterpArg(8, gpu_exclusion_);
  SetInterpArg(9, gpu_loopcheck_);

  SetInterpArg(10, env_dat_->f_samples_buffers[0]);
  SetInterpArg(11, env_dat_->phi_samples_buffers[0]);
  SetInterpArg(12, env_dat_->theta_samples_buffers[0]);
  SetInterpArg(13, env_dat_->f_samples_buffers[1]);
  SetInterpArg(14, env_dat_->phi_samples_buffers[1]);
  SetInterpArg(15, env_dat_->theta_samples_buffers[1]);
  SetInterpArg(16, env_dat_->brain_mask_buffer);
  SetInterpArg(17, env_dat_->waypoint_masks_buffer);
  SetInterpArg(18, env_dat_->termination_mask_buffer);
  SetInterpArg(19, env_dat_->exclusion_mask_buffer);

  ret = cq_->enqueueNDRangeKernel(
    *(ptx_kernel_),
    particle_offset,
    particles_to_compute,
    cl::NullRange,
    NULL,
    NULL);
  if (CL_SUCCESS != ret)
    die(ret);

  ret = cq_->finish();
  if (CL_SUCCESS != ret)
    die(ret);
}

void OclPtxHandler::RunSumKernel(int side)
{
  cl_int ret;

  cl::NDRange space_to_compute((attrs_.sample_nx
                              * attrs_.sample_ny
                              * attrs_.sample_nz / 32) + 1);
  cl::NDRange space_offset(0);

  sum_kernel_->setArg(
      0,
      sizeof(struct OclPtxHandler::particle_attrs),
      reinterpret_cast<void*>(&attrs_));
  sum_kernel_->setArg(1, side);
  SetSumArg(2, gpu_complete_);
  //SetSumArg(3, gpu_local_pdf_);
  SetSumArg(4, gpu_waypoints_);
  SetSumArg(5, gpu_exclusion_);
  SetSumArg(6, gpu_step_count_);
  SetSumArg(7, gpu_global_pdf_);

  ret = cq_->enqueueNDRangeKernel(
    *(sum_kernel_),
    space_offset,
    space_to_compute,
    cl::NullRange,
    NULL,
    NULL);
  if (CL_SUCCESS != ret)
    die(ret);

  // This line consumes >95% of our time.
  ret = cq_->finish();
  if (CL_SUCCESS != ret)
    die(ret);
}

void OclPtxHandler::RunKernel(int side)
{
  RunInterpKernel(side);
}

void OclPtxHandler::ReadStatus(int offset, int count, cl_ushort *ret)
{
  cl_int err = cq_->enqueueReadBuffer(
      *gpu_complete_,
      true,
      offset * sizeof(cl_ushort),
      count * sizeof(cl_ushort),
      reinterpret_cast<cl_ushort*>(ret));
  if (CL_SUCCESS != err)
    die(err);
}

void OclPtxHandler::DumpPath(int offset, int count)
{
  if (!env_dat_->save_paths)
    return;

  cl_float4 *path_buf = new cl_float4[count * attrs_.steps_per_kernel];
  cl_ushort *step_count_buf = new cl_ushort[count];
  int ret;
  cl_float4 value;

  assert(NULL != path_dump_fd_);

  // kludge(jeff): The first time this is called by threading::Worker, there
  // is only garbage on the GPU, which we'd like to avoid dumping to file---
  // as it makes automatic verification more challenging.  My 5-second kludge
  // is to see if this is the first time we've been called.  If so, return
  // without printing anything.
  if (first_time_)
  {
    first_time_ = 0;
    return;
  }

  ret = cq_->enqueueReadBuffer(
      *gpu_path_,
      true,
      offset * attrs_.steps_per_kernel * sizeof(cl_float4),
      count * attrs_.steps_per_kernel * sizeof(cl_float4),
      reinterpret_cast<void*>(path_buf));
  if (CL_SUCCESS != ret)
  {
    puts("Failed to read back path");
    die(ret);
  }

  ret = cq_->enqueueReadBuffer(
      *gpu_step_count_,
      true,
      offset * sizeof(cl_ushort),
      count * sizeof(cl_ushort),
      reinterpret_cast<void*>(step_count_buf));
  if (CL_SUCCESS != ret)
  {
    puts("Failed to read back path");
    die(ret);
  }

  // Now dumpify.
  for (int id = 0; id < count; ++id)
  {
    for (int step = 0; step < attrs_.steps_per_kernel; ++step)
    {
      value = path_buf[id * attrs_.steps_per_kernel + step];
      // Only dump if this element is before the path's end.
      if ((0 == step_count_buf[id] % attrs_.steps_per_kernel
        && 0 != step_count_buf[id])
        || step < step_count_buf[id] % attrs_.steps_per_kernel)
        fprintf(path_dump_fd_, "%i:%f,%f,%f\n",
            id + offset,
            value.s[0],
            value.s[1],
            value.s[2]);
    }
  }

  delete path_buf;
  delete step_count_buf;
}
