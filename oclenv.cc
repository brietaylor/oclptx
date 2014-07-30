/*  Copyright (C) 2014
 *    Afshin Haidari
 *    Steve Novakov
 *    Jeff Taylor
 */

/* oclenv.cc
 *
 *
 * Part of
 *    oclptx
 * OpenCL-based, GPU accelerated probtrackx algorithm module, to be used
 * with FSL - FMRIB's Software Library
 *
 * This file is part of oclptx.
 *
 * oclptx is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * oclptx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with oclptx.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
//#include <mutex>
//#include <thread>

#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "oclenv.h"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
  static const std::string slash="\\";
#else
  static const std::string slash="/";
#endif

static void die(int reason);

//*********************************************************************
//
// OclEnv Constructors/Destructors
//
//*********************************************************************
//
// Constructor(s)
//
OclEnv::OclEnv()
{
  this->env_data.f_samples_buffers = NULL;
  this->env_data.phi_samples_buffers = NULL;
  this->env_data.theta_samples_buffers = NULL;
  this->env_data.brain_mask_buffer = NULL;
  this->env_data.exclusion_mask_buffer = NULL;
  this->env_data.termination_mask_buffer = NULL;
  this->env_data.waypoint_masks_buffer = NULL;
}

//
// Destructor
//
OclEnv::~OclEnv()
{
  if (this->env_data.f_samples_buffers != NULL)
  {
    for (uint32_t s = 0; s < 2; s++)
    {
      if(this->env_data.f_samples_buffers[s] != NULL)
        delete this->env_data.f_samples_buffers[s];
      if(this->env_data.phi_samples_buffers[s] != NULL)
        delete this->env_data.phi_samples_buffers[s];
      if(this->env_data.theta_samples_buffers[s] != NULL)
        delete this->env_data.theta_samples_buffers[s];
    }
    
    delete[] this->env_data.f_samples_buffers;
    delete[] this->env_data.phi_samples_buffers;
    delete[] this->env_data.theta_samples_buffers;
  }

  delete this->env_data.brain_mask_buffer;
  
  if (this->env_data.exclusion_mask_buffer != NULL)
    delete this->env_data.exclusion_mask_buffer;
  if (this->env_data.termination_mask_buffer != NULL)
    delete this->env_data.termination_mask_buffer;
  if (this->env_data.waypoint_masks_buffer != NULL)
    delete this->env_data.waypoint_masks_buffer;

  for (uint32_t i = 0; i < this->device_global_pdf_buffers.size(); i++)
    delete device_global_pdf_buffers.at(i);
}

static void die(int reason)
{
  if (CL_MEM_OBJECT_ALLOCATION_FAILURE == reason)
  {
    puts("Ran out of device memory while allocating static buffers.");
    exit(-1);
  }
  else
    abort();
}

//*********************************************************************
//
// OclEnv Container Set/Get
//
//*********************************************************************

cl::Context * OclEnv::GetContext()
{
  return &(this->ocl_context);
}

cl::CommandQueue * OclEnv::GetCq(unsigned int device_num)
{
  return &(this->ocl_device_queues.at(device_num));
}

cl::Kernel * OclEnv::GetKernel(unsigned int kernel_num)
{
  return &(this->ocl_kernel_set.at(kernel_num));
}

cl::Kernel * OclEnv::GetSumKernel(unsigned int kernel_num)
{
  return &(this->sum_kernel_set.at(kernel_num));
}

void OclEnv::SetOclRoutine(std::string new_routine)
{
  this->CreateKernels(new_routine);
}

EnvironmentData * OclEnv::GetEnvData()
{
  return &(this->env_data);
}

cl::Buffer * OclEnv::GetDevicePdf(uint32_t device_num)
{
  return this->device_global_pdf_buffers.at(device_num);
}


//*********************************************************************
//
// OclEnv OpenCL Interface
//
//*********************************************************************

//
// Currently ignores all other devices that arent GPU.
//
//
void OclEnv::OclInit()
{
  // Prevent Nvidia from (erroneously) caching stale build files.
  setenv("CUDA_CACHE_DISABLE", "1", 1);

  cl::Platform::get(&(this->ocl_platforms));

  if (0 == this->ocl_platforms.size())
  {
    printf("No OpenCL platforms found.\n");
    exit(-1);
  }

  cl_context_properties con_prop[3] =
  {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties) (this->ocl_platforms[0]) (),
    0
  };

  this->ocl_platforms.at(0);

  // cl::Context test_context = cl::Context(CL_DEVICE_TYPE_GPU, con_prop);
  // std::vector<cl::Device> test_device_list =
  //   this->ocl_context.getInfo<CL_CONTEXT_DEVICES>();

  this->ocl_context = cl::Context(CL_DEVICE_TYPE_GPU, con_prop);
  // GPU DEVICES ONLY, FOR CPU, (don't use CPU unless informed, not
  // quite the same physical interface):
  // this->oclContext = cl::Context(CL_DEVICE_TYPE_CPU, conProp);
  this->ocl_devices = this->ocl_context.getInfo<CL_CONTEXT_DEVICES>();
}

void OclEnv::OclDeviceInfo()
{
  std::cout<<"\nLocal OpenCL Devices\n";

  size_t siT[3];
  cl_uint print_int;
  cl_ulong print_ulong;
  std::string print_string;

  std::string device_name;

  for(std::vector<cl::Device>::iterator dit = this->ocl_devices.begin();
    dit != this->ocl_devices.end(); ++dit){

    dit->getInfo(CL_DEVICE_NAME, &print_string);
    dit->getInfo(CL_DEVICE_NAME, &device_name);
    dit->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &print_int);
    dit->getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &siT);
    dit->getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &siT);


    std::cout<<"\tDEVICE\n";
    std::cout<<"\tDevice Name: " << print_string << "\n";
    std::cout<<"\tMax Compute Units: " << print_int << "\n";
    std::cout<<"\tMax Work Group Size (x*y*z): " << siT[0] << "\n";
    std::cout<<"\tMax Work Item Sizes (x, y, z): " << siT[0] <<
      ", " << siT[1] << ", " << siT[2] << "\n";

    dit->getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &print_ulong);
    std::cout<<"\tMax Mem Alloc Size: " << print_ulong << "\n";

    std::cout<<"\n";
  }
}

unsigned int OclEnv::HowManyDevices()
{
  return this->ocl_devices.size();
}

unsigned int OclEnv::HowManyCQ()
{
  return this->ocl_device_queues.size();
}

//
//
//
void OclEnv::NewCLCommandQueues(std::string gpu_select)
{
  this->ocl_device_queues.clear();

  if (gpu_select == "")
  {
    for (uint32_t k = 0; k < this->ocl_devices.size(); k++ )
    {
      std::cout<<"Create CommQueue, Kernel, Device: "<<k<<"\n";

      this->ocl_device_queues.push_back(
        cl::CommandQueue(
          this->ocl_context,
          this->ocl_devices[k]
        )
      );
    }
  }
  else
  {
    std::vector<uint32_t> gpu_list;
    uint32_t next_item;
    std::string parse_string = gpu_select;

    for (uint32_t s = 0; s < gpu_select.size(); s++)
    {
      if (s%2 != 0) // ignore commas, NOT TRUE CSV, DOESNT WORK FOR >10 DEVICES
        continue;
      
      next_item = std::atoi(string(1, gpu_select[s]).c_str());

      if ( (std::find(gpu_list.begin(), gpu_list.end(), next_item)
        == gpu_list.end()) && (next_item < this->ocl_devices.size()))
        // protect against adding same device twice (bad...)
      {
        gpu_list.push_back(next_item);
      }
    }

    for (uint32_t k = 0; k < gpu_list.size(); k++ )
    {
      std::cout<<"Create CommQueue, Kernel, Device: "<< gpu_list.at(k) <<"\n";

      this->ocl_device_queues.push_back(
        cl::CommandQueue(
          this->ocl_context,
          this->ocl_devices[gpu_list.at(k)]
        )
      );
    }
  }
}


//
//
//
void OclEnv::CreateKernels( std::string kernel_name )
{
  this->ocl_kernel_set.clear();
  this->sum_kernel_set.clear();

  cl_int err;

  // Read Source
  std::string fold = "oclkernels";

  std::string interp_kernel_source;
  std::string sum_kernel_source = fold + slash + "summing.cl";
  std::string define_list =  "-I ./oclkernels";

  if (kernel_name == "standard")
  {
    interp_kernel_source = fold + slash + "interpolate.cl";
  }
  else if (kernel_name == "rng_test")
  {
    interp_kernel_source = fold + slash + "rng_test.cl";
  }
  else if (kernel_name == "interptest")
  {
    interp_kernel_source = fold + slash + "interptest.cl";
  }
  else if (kernel_name == "basic")
  {
    interp_kernel_source = fold + slash + "basic.cl";
  }
  else
  {
    printf("Invalid Kernel Name: Terminating...\n");
    exit(EXIT_FAILURE);
  }

  if (!(this->env_data.deterministic))
    define_list += " -D PRNG";
  if (this->env_data.bpx_dirs > 1)
    define_list += " -D TWO_DIR";
  if (this->env_data.n_waypts > 0)
    define_list += " -D WAYPOINTS";
  if (this->env_data.terminate_mask)
    define_list += " -D TERMINATION";
  if (this->env_data.exclusion_mask)
    define_list += " -D EXCLUSION";
  if (this->env_data.euler_streamline)
    define_list += " -D EULER_STREAMLINE";
  if (this->env_data.way_and)
    define_list += " -D WAYAND";
  if (this->env_data.save_paths)
    define_list += " -D PATH_SAVE";
  if (this->env_data.loopcheck)
    define_list += " -D LOOPCHECK";
  if (this->env_data.aniso_const)
    define_list += " -D ANISOTROPIC";

  // Compute the rbtree size
  char buf[32];
  snprintf(buf, 32, " -D kMaxSize=%i", env_data.max_steps);
  define_list += buf;

  snprintf(buf, 32, " -D kMaxDepth=%i",
      (int) (2 * std::ceil(std::log2(env_data.max_steps))));
  define_list += buf;

  std::ifstream main_stream(interp_kernel_source);
  std::string main_code(  (std::istreambuf_iterator<char>(main_stream) ),
                            (std::istreambuf_iterator<char>()));

  std::ifstream sum_stream(sum_kernel_source);
  std::string sum_code(  (std::istreambuf_iterator<char>(sum_stream) ),
                            (std::istreambuf_iterator<char>()));

  printf("Build Options: %s\n", define_list.c_str());
  //
  // Build Program files here
  //

  cl::Program::Sources main_source(
    1,
    std::make_pair(main_code.c_str(), main_code.length())
  );

  cl::Program::Sources sum_source(
    1,
    std::make_pair(sum_code.c_str(), sum_code.length())
  );

  cl::Program main_program(this->ocl_context, main_source);

  err = main_program.build(this->ocl_devices, define_list.c_str());

  if(err != CL_SUCCESS)
  {
    std::cout<<"ERROR: " <<
      " ( " << this->OclErrorStrings(err) << ")\n";

    std::vector<cl::Device>::iterator dit = this->ocl_devices.begin();

    std::cout<<"BUILD OPTIONS: \n" <<
      main_program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(*dit) <<
       "\n";
    std::cout<<"BUILD LOG: \n" <<
      main_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*dit) <<"\n";

    exit(EXIT_FAILURE);
  }

  cl::Program sum_program(this->ocl_context, sum_source);

  err = sum_program.build(this->ocl_devices, define_list.c_str());

  if(err != CL_SUCCESS)
  {
    std::cout<<"ERROR: " <<
      " ( " << this->OclErrorStrings(err) << ")\n";

    std::vector<cl::Device>::iterator dit = this->ocl_devices.begin();

    std::cout<<"BUILD OPTIONS: \n" <<
      sum_program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(*dit) <<
       "\n";
    std::cout<<"BUILD LOG: \n" <<
      sum_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*dit) <<"\n";

    exit(EXIT_FAILURE);
  }
  

  //
  // Compile Kernels from Program
  //
  for( unsigned int k = 0; k < this->ocl_device_queues.size(); k++)
  {
    if (kernel_name == "standard" )
    {
      this->ocl_kernel_set.push_back(cl::Kernel(main_program,
                                                "OclPtxInterpolate",
                                                NULL));
      this->sum_kernel_set.push_back(cl::Kernel(sum_program,
                                                "PdfSum",
                                                NULL));
    }
    else if (kernel_name == "rng_test")
    {
      this->ocl_kernel_set.push_back(cl::Kernel(main_program,
                                                "RngTest",
                                                NULL));
    }
    else if (kernel_name == "interptest" )
    {
      this->ocl_kernel_set.push_back(cl::Kernel(main_program,
                                                "InterpolateTestKernel",
                                                NULL));
    }
    else if (kernel_name == "basic" )
    {
      this->ocl_kernel_set.push_back(cl::Kernel(main_program,
                                                "BasicInterpolate",
                                                NULL));
    }
  }
}


//
// Matches OCL error codes to their meaning.
//
std::string OclEnv::OclErrorStrings(cl_int error)
{
  const std::string cl_error_string[] =
  {
    "CL_SUCCESS",
    "CL_DEVICE_NOT_FOUND",
    "CL_DEVICE_NOT_AVAILABLE",
    "CL_COMPILER_NOT_AVAILABLE",
    "CL_MEM_OBJECT_ALLOCATION_FAILURE",
    "CL_OUT_OF_RESOURCES",
    "CL_OUT_OF_HOST_MEMORY",
    "CL_PROFILING_INFO_NOT_AVAILABLE",
    "CL_MEM_COPY_OVERLAP",
    "CL_IMAGE_FORMAT_MISMATCH",
    "CL_IMAGE_FORMAT_NOT_SUPPORTED",
    "CL_BUILD_PROGRAM_FAILURE",
    "CL_MAP_FAILURE",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "CL_INVALID_VALUE",
    "CL_INVALID_DEVICE_TYPE",
    "CL_INVALID_PLATFORM",
    "CL_INVALID_DEVICE",
    "CL_INVALID_CONTEXT",
    "CL_INVALID_QUEUE_PROPERTIES",
    "CL_INVALID_COMMAND_QUEUE",
    "CL_INVALID_HOST_PTR",
    "CL_INVALID_MEM_OBJECT",
    "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
    "CL_INVALID_IMAGE_SIZE",
    "CL_INVALID_SAMPLER",
    "CL_INVALID_BINARY",
    "CL_INVALID_BUILD_OPTIONS",
    "CL_INVALID_PROGRAM",
    "CL_INVALID_PROGRAM_EXECUTABLE",
    "CL_INVALID_KERNEL_NAME",
    "CL_INVALID_KERNEL_DEFINITION",
    "CL_INVALID_KERNEL",
    "CL_INVALID_ARG_INDEX",
    "CL_INVALID_ARG_VALUE",
    "CL_INVALID_ARG_SIZE",
    "CL_INVALID_KERNEL_ARGS",
    "CL_INVALID_WORK_DIMENSION",
    "CL_INVALID_WORK_GROUP_SIZE",
    "CL_INVALID_WORK_ITEM_SIZE",
    "CL_INVALID_GLOBAL_OFFSET",
    "CL_INVALID_EVENT_WAIT_LIST",
    "CL_INVALID_EVENT",
    "CL_INVALID_OPERATION",
    "CL_INVALID_GL_OBJECT",
    "CL_INVALID_BUFFER_SIZE",
    "CL_INVALID_MIP_LEVEL",
    "CL_INVALID_GLOBAL_WORK_SIZE"
  };

  return cl_error_string[ -1*error];
}

//*********************************************************************
//
// Resource Allocation
//
//*********************************************************************

// TODO @STEVE
// Right now I've just kluged together AllocateSamples, but really
// AvailableGPUMem should run more thoroughly and calculate a lot of the vallues
// currently bneing calculated in AllocateSamples
//
uint32_t OclEnv::AvailableGPUMem(
  const BedpostXData* f_data,
  const oclptxOptions& ptx_options,
  uint32_t n_waypoints,
  const unsigned short int* exclusion_mask,
  const unsigned short int* termination_mask
)
{
  // ***********************************************
  //  Hardware Parameters
  // ***********************************************
  cl_ulong max_buff_size;
  cl_ulong gl_mem_size;
  cl_ulong useful_gl_mem_size;

  std::vector<cl::Device>::iterator dit = this->ocl_devices.begin();

  dit->getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &max_buff_size);
  this->env_data.max_buffer_size = max_buff_size;

  dit->getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &gl_mem_size);
  this->env_data.global_mem_size = gl_mem_size;

  useful_gl_mem_size =
    std::floor(gl_mem_size * ptx_options.mem_risk_frac.value());

  // ***********************************************
  //  BPX Sample Parameters + Masks
  // ***********************************************

  this->env_data.bpx_dirs = 1;
  // oclptx supports up to two fiber directions per vertex. Supporting 3
  // is unpractical for the typical workstation GPU.
  if( f_data->data.size() > 2)
    this->env_data.bpx_dirs = 2;

  this->env_data.nx = f_data->nx;
  this->env_data.ny = f_data->ny;
  this->env_data.nz = f_data->nz;
  this->env_data.ns = f_data->ns;

  cl_uint single_direction_size =
    f_data->nx * f_data->ny * f_data->nz;

  cl_uint single_pdf_mask_size = (single_direction_size / 32)  + 1;

  cl_uint brain_mem_size =
    single_direction_size * sizeof(unsigned short int);

  this->env_data.mask_mem_size = brain_mem_size;

  cl_uint single_direction_mem_size =
    single_direction_size*f_data->ns*sizeof(float);

  this->env_data.single_sample_mem_size = single_direction_mem_size;

  cl_uint num_samp = 2;
  // Anisotropic Constraint
  this->env_data.aniso_const = ptx_options.usef.value();
  if (this->env_data.aniso_const)
  {
    printf("\nUsing Anisotropic Constraint for Tracking\n");
    num_samp = 3;
  }

  cl_uint total_mem_size =
    num_samp*single_direction_mem_size * this->env_data.bpx_dirs +
    brain_mem_size*(1 + n_waypoints);

  if (exclusion_mask != NULL)
  {
    printf("Exmask\n");
    total_mem_size += brain_mem_size;
    this->env_data.exclusion_mask = true;
  }
  else
    this->env_data.exclusion_mask = false;

  if (termination_mask != NULL)
  {
    printf("termimask\n");
    total_mem_size += brain_mem_size;
    this->env_data.terminate_mask = true;
  }
  else
    this->env_data.terminate_mask = false;

  if (ptx_options.waycond.value() == "AND")
    this->env_data.way_and = true;
  else
    this->env_data.way_and = false;

  this->env_data.n_waypts = n_waypoints;

  // ***********************************************
  //  PDFS
  // ***********************************************

  this->env_data.global_pdf_size = single_pdf_mask_size * 32;
  this->env_data.global_pdf_mem_size =
    this->env_data.global_pdf_size * sizeof(uint32_t);

  this->env_data.total_static_gpu_mem =
    total_mem_size + this->env_data.global_pdf_mem_size;

  this->env_data.dynamic_mem_left = useful_gl_mem_size - this->env_data.total_static_gpu_mem;

  // ***********************************************
  //  Dynamic Particle Containers/Parameters
  // ***********************************************

  this->env_data.particle_pdf_mask_mem_size =
    single_pdf_mask_size * sizeof(uint32_t);
  this->env_data.pdf_entries_per_particle = single_pdf_mask_size;
  
  // Loopcheck
  uint32_t loopcheck_x;
  uint32_t loopcheck_y;
  uint32_t loopcheck_z;
  cl_uint single_loopcheck_size;
  cl_uint single_loopcheck_mask_size;

  // Hard Coded, just like in ptx2
  uint32_t loopcheck_fraction = 5;

  if (ptx_options.loopcheck.value())
  {
    this->env_data.loopcheck = true;
    loopcheck_x = (f_data->nx / loopcheck_fraction) +
      ((f_data->nx %loopcheck_fraction > 0)? 1 : 0);
    loopcheck_y = (f_data->ny / loopcheck_fraction) +
      ((f_data->ny %loopcheck_fraction > 0)? 1 : 0);
    loopcheck_z = (f_data->nz / loopcheck_fraction) +
      ((f_data->nz %loopcheck_fraction > 0)? 1 : 0);

    this->env_data.lx = loopcheck_x;
    this->env_data.ly = loopcheck_y;
    this->env_data.lz = loopcheck_z;

    single_loopcheck_size = loopcheck_x * loopcheck_y * loopcheck_z;
    single_loopcheck_mask_size = (single_loopcheck_size / 32)  +
      ((single_loopcheck_size%32 > 0)? 1 : 0);

    this->env_data.loopcheck_dir_size = single_loopcheck_size;
    this->env_data.loopcheck_location_size = single_loopcheck_mask_size;

    this->env_data.particle_loopcheck_location_mem_size =
      single_loopcheck_mask_size * sizeof(uint32_t);
    this->env_data.particle_loopcheck_dir_mem_size =
      loopcheck_x * loopcheck_y *loopcheck_z * sizeof(float4);
  }
  else
  {
    this->env_data.loopcheck = false;
    this->env_data.particle_loopcheck_location_mem_size = 0;
    this->env_data.particle_loopcheck_dir_mem_size = 0;
  }

  // Modified Euler

  this->env_data.euler_streamline = ptx_options.modeuler.value();
  if (this->env_data.euler_streamline)
    printf("\nUsing Modified Euler Integration Method\n");

  // Step Length
  printf("Step Length: %f\n", ptx_options.steplength.value());

  // PRNG?

  this->env_data.deterministic = ptx_options.norng.value();

  // paths?
  this->env_data.max_steps = ptx_options.nsteps.value();
  this->env_data.save_paths = ptx_options.save_paths.value();
  
  if (this->env_data.save_paths)
    printf("Saving Path Data\n");

  printf("Voxel Dims (x,y,z): %u, %u, %u\n",
    f_data->nx, f_data->ny, f_data->nz);
  printf("Num Samples: %u\n", f_data->ns);
  printf("Single Dir Sample Mem Size: %u (B), %.4f (MB), \n",
  single_direction_mem_size, single_direction_mem_size/1e6);
  printf("Num_directions: %u \n", this->env_data.bpx_dirs);
  printf("Total GPU Device Memory: %.4f (MB) \n", gl_mem_size/1e6);
  printf("Total Attempted Use of GPU Device Memory: %.4f (MB) \n",
    useful_gl_mem_size/1e6);
  printf("Total Static Data Memory Requirement: %.4f (MB) \n",
    this->env_data.total_static_gpu_mem/1e6);
  printf("Remaining GPU Device Memory: %.4f (MB) \n",
    this->env_data.dynamic_mem_left/1e6);

  if (single_direction_mem_size > max_buff_size){
    printf("ERROR: BPX DATA > MAX BUFFER SIZE: %.4f (MB) vs %.4f (MB)\n",
      single_direction_mem_size/1e6, max_buff_size/1e6);
    printf("TERMINATING PROGRAM...\n");
    exit(EXIT_FAILURE);
  }

  // Check for OOM before going any further
  if (this->env_data.dynamic_mem_left < 0)
  {
      printf("Not enough device memory to support static buffers.\n");
      exit(-1);
  }

  return 1;
}


void OclEnv::AllocateSamples(
  const BedpostXData* f_data,
  const BedpostXData* phi_data,
  const BedpostXData* theta_data,
  const unsigned short int* brain_mask,
  const unsigned short int* exclusion_mask,
  const unsigned short int* termination_mask,
  std::vector<unsigned short int*>* waypoint_masks
)
{
  uint32_t n_dirs = this->env_data.bpx_dirs;
  cl_int ret;

  this->env_data.f_samples_buffers = new cl::Buffer*[2];
  this->env_data.phi_samples_buffers = new cl::Buffer*[2];
  this->env_data.theta_samples_buffers = new cl::Buffer*[2];

  for (uint32_t n = 0; n < 2; n++)
  {
    this->env_data.f_samples_buffers[n] = NULL;
    this->env_data.phi_samples_buffers[n] = NULL;
    this->env_data.theta_samples_buffers[n] = NULL;
  }

  for (uint32_t s = 0; s < n_dirs; s++)
  {
    if (this->env_data.aniso_const)
    {
      this->env_data.f_samples_buffers[s] = new
        cl::Buffer(
          this->ocl_context,
          CL_MEM_READ_ONLY,
          this->env_data.single_sample_mem_size,
          NULL,
          &ret
        );
      if (CL_SUCCESS != ret)
        die(ret);
    }

    this->env_data.theta_samples_buffers[s] = new
      cl::Buffer(
        this->ocl_context,
        CL_MEM_READ_ONLY,
        this->env_data.single_sample_mem_size,
        NULL,
        &ret
      );
    if (CL_SUCCESS != ret)
      die(ret);

    this->env_data.phi_samples_buffers[s] = new
      cl::Buffer(
        this->ocl_context,
        CL_MEM_READ_ONLY,
        this->env_data.single_sample_mem_size,
        NULL,
        &ret
      );
    if (CL_SUCCESS != ret)
      die(ret);
  }

  this->env_data.brain_mask_buffer = new
    cl::Buffer(
      this->ocl_context,
      CL_MEM_READ_ONLY,
      this->env_data.mask_mem_size,
      NULL,
      &ret
    );
  if (CL_SUCCESS != ret)
    die(ret);

    if (exclusion_mask != NULL)
    {
      this->env_data.exclusion_mask_buffer = new
        cl::Buffer(
          this->ocl_context,
          CL_MEM_READ_ONLY,
          this->env_data.mask_mem_size,
          NULL,
          &ret
        );
      if (CL_SUCCESS != ret)
        die(ret);
    }

    if (termination_mask != NULL)
    {
      this->env_data.termination_mask_buffer = new
        cl::Buffer(
          this->ocl_context,
          CL_MEM_READ_ONLY,
          this->env_data.mask_mem_size,
          NULL,
          &ret
        );
      if (CL_SUCCESS != ret)
        die(ret);
    }

    if (this->env_data.n_waypts > 0)
    {
      this->env_data.waypoint_masks_buffer = new
        cl::Buffer(
          this->ocl_context,
          CL_MEM_READ_ONLY,
          this->env_data.n_waypts * this->env_data.mask_mem_size,
          NULL,
          &ret
        );
      if (CL_SUCCESS != ret)
        die(ret);
    }

    for (uint32_t d = 0; d < this->ocl_device_queues.size(); d++)
    {
      this->device_global_pdf_buffers.push_back(
        new cl::Buffer(
          this->ocl_context,
          CL_MEM_WRITE_ONLY,
          this->env_data.global_pdf_mem_size,
          NULL,
          NULL
        )
      );
    }

    uint32_t *global_init =
      new uint32_t[this->env_data.global_pdf_size];
    for (uint32_t j = 0; j < this->env_data.global_pdf_size; j++)
      global_init[j] = 0;

    for (uint32_t d = 0; d < this->ocl_device_queues.size(); d++)
    {
      for (uint32_t s = 0; s < n_dirs; s++)
      {
        if (this->env_data.aniso_const)
        {
          ret = this->ocl_device_queues.at(d).enqueueWriteBuffer(
            *(this->env_data.f_samples_buffers[s]),
            CL_FALSE,
            static_cast<unsigned int>(0),
            this->env_data.single_sample_mem_size,
            f_data->data.at(s),
            NULL,
            NULL
          );
          if (CL_SUCCESS != ret)
            die(ret);
        }

        ret = this->ocl_device_queues.at(d).enqueueWriteBuffer(
          *(this->env_data.theta_samples_buffers[s]),
          CL_FALSE,
          static_cast<unsigned int>(0),
          this->env_data.single_sample_mem_size,
          theta_data->data.at(s),
          NULL,
          NULL
        );
        if (CL_SUCCESS != ret)
          die(ret);

        ret = this->ocl_device_queues.at(d).enqueueWriteBuffer(
          *(this->env_data.phi_samples_buffers[s]),
          CL_FALSE,
          static_cast<unsigned int>(0),
          this->env_data.single_sample_mem_size,
          phi_data->data.at(s),
          NULL,
          NULL
        );
        if (CL_SUCCESS != ret)
          die(ret);
      }

      ret = this->ocl_device_queues.at(d).enqueueWriteBuffer(
        *(this->env_data.brain_mask_buffer),
        CL_FALSE,
        static_cast<unsigned int>(0),
        this->env_data.mask_mem_size,
        const_cast<unsigned short int*>(brain_mask),
        NULL,
        NULL
      );
      if (CL_SUCCESS != ret)
        die(ret);

      if (exclusion_mask != NULL)
      {
        ret = this->ocl_device_queues.at(d).enqueueWriteBuffer(
          *(this->env_data.exclusion_mask_buffer),
          CL_FALSE,
          static_cast<unsigned int>(0),
          this->env_data.mask_mem_size,
          const_cast<unsigned short int*>(exclusion_mask),
          NULL,
          NULL
        );
        if (CL_SUCCESS != ret)
          die(ret);
      }

      if (termination_mask != NULL)
      {
        ret = this->ocl_device_queues.at(d).enqueueWriteBuffer(
          *(this->env_data.termination_mask_buffer),
          CL_FALSE,
          static_cast<unsigned int>(0),
          this->env_data.mask_mem_size,
          const_cast<unsigned short int*>(termination_mask),
          NULL,
          NULL
        );
        if (CL_SUCCESS != ret)
          die(ret);
      }

      for (uint32_t w = 0; w < this->env_data.n_waypts; w++)
      {
        printf("w: %u, addr:  %hu\n", w, waypoint_masks->at(w));
        ret = this->ocl_device_queues.at(d).enqueueWriteBuffer(
          *(this->env_data.waypoint_masks_buffer),
          CL_FALSE,
          w * this->env_data.mask_mem_size,
          this->env_data.mask_mem_size,
          waypoint_masks->at(w),
          NULL,
          NULL
        );
        if (CL_SUCCESS != ret)
          die(ret);
      }

      this->ocl_device_queues.at(d).enqueueWriteBuffer(
        *(this->device_global_pdf_buffers.at(d)),
        CL_FALSE,
        static_cast<unsigned int>(0),
        this->env_data.global_pdf_mem_size,
        global_init,
        NULL,
        NULL
      );

      ret = this->ocl_device_queues.at(d).flush();
      if (CL_SUCCESS != ret)
        die(ret);
    }

    // can maybe move this to oclptxhandler, for slight performance improvement
    for (uint32_t d = 0; d < this->ocl_device_queues.size(); d++)
    {
      ret = this->ocl_device_queues.at(d).finish();
      if (CL_SUCCESS != ret)
        die(ret);
    }
}

void OclEnv::PdfsToFile(std::string filename)
{
  uint32_t *temp_pdf = new uint32_t[this->env_data.global_pdf_size];
  uint32_t *total_pdf = new uint32_t[this->env_data.global_pdf_size];

  for (uint32_t i = 0; i < this->env_data.global_pdf_size; i++)
  {
    temp_pdf[i] = 0;
    total_pdf[i] = 0;
  }

  for (uint32_t d = 0; d < this->ocl_device_queues.size(); d++)
  {
    this->ocl_device_queues.at(d).enqueueReadBuffer(
      *(this->device_global_pdf_buffers.at(d)),
      CL_TRUE,
      static_cast<unsigned int>(0),
      this->env_data.global_pdf_mem_size,
      temp_pdf
    );
    for (uint32_t i = 0; i < this->env_data.global_pdf_size; i++)
    {
      total_pdf[i] += temp_pdf[i];
    }
  }

  //
  // now write to file
  //

  FILE * pdf_file;
  pdf_file = fopen(filename.c_str(), "wb");

  uint32_t index = 0;

  for (uint32_t k = 0; k < this->env_data.nz; k++)
  {
    for (uint32_t j = 0; j < this->env_data.ny; j++)
    {
      for (uint32_t i = 0; i < this->env_data.nx; i++)
      {
        index =
          i*(this->env_data.ny*this->env_data.nz) + j*(this->env_data.nz) + k;
        fprintf(pdf_file, "%u", total_pdf[index]);

        if (i < this->env_data.nx - 1)
          fprintf(pdf_file, " ");
      }
      fprintf(pdf_file, "\n");
    }
  }

  fclose(pdf_file);

  delete[] temp_pdf;
  delete[] total_pdf;
}

//EOF
