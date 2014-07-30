/* Copyright 2014
 *  Afshin Haidari
 *  Steve Novakov
 *  Jeff Taylor
 */

#include <unistd.h>
#include <cassert>
#include <thread>
#include <chrono>
#include <cmath>

#include "fifo.h"
#include "oclenv.h"
#include "oclptxhandler.h"
#include "particlegen.h"
#include "samplemanager.h"
#include "threading.h"


cl_ulong8 rng_zero = {{0,}};

std::chrono::high_resolution_clock::time_point t_end;
std::chrono::high_resolution_clock::time_point t_start;

void start_timer()
{
  t_start = std::chrono::high_resolution_clock::now();
}

void end_timer(const char *verb)
{
  t_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> delta_t = t_end - t_start;

  printf("Time to %s: %fs\n", verb, delta_t.count());
}

int main(int argc, char **argv)
{
  const int kStepsPerKernel = 1000;
  const int kNumReducers = 1;
  FILE *global_fd;
  Fifo<struct OclPtxHandler::particle_data> *particles_fifo;

  // Startup the samplemanager
  puts("Loading samples...");
  start_timer();
  SampleManager sample_manager;
  sample_manager.ParseCommandLine(argc, argv);
  end_timer("load samples");

  puts("Setting up OpenCL...");
  start_timer();

  // Create our oclenv
  OclEnv env;
  env.OclInit();
  env.NewCLCommandQueues(
    sample_manager.GetOclptxOptions().gpuselect.value());

  const unsigned short int * rubbish_mask =
    sample_manager.GetExclusionMaskToArray();
  const unsigned short int * stop_mask =
    sample_manager.GetTerminationMaskToArray();
  std::vector<unsigned short int*>* waypoints =
    sample_manager.GetWayMasksToVector();

  env.AvailableGPUMem(
    sample_manager.GetFDataPtr(),
    sample_manager.GetOclptxOptions(),
    waypoints->size(),
    rubbish_mask,
    stop_mask
  );

  env.CreateKernels("standard");

  env.AllocateSamples(
    sample_manager.GetFDataPtr(),
    sample_manager.GetPhiDataPtr(),
    sample_manager.GetThetaDataPtr(),
    sample_manager.GetBrainMaskToArray(),
    rubbish_mask,
    stop_mask,
    waypoints
  );

  global_fd = fopen("./path_output", "w");
  if (NULL == global_fd)
  {
    perror("Couldn't open file");
    exit(1);
  }

  cl_float4 dims = sample_manager.brain_mask_dim();
  int min_steps = ceil(
    sample_manager.GetOclptxOptions().distthresh.value()
  * sample_manager.GetOclptxOptions().steplength.value()
  / dims.s[0]);

  int fibst = sample_manager.GetOclptxOptions().fibst.value() - 1;
  if (fibst > 1);
    fibst = 1;

  struct OclPtxHandler::particle_attrs attrs = {
    sample_manager.brain_mask_dim(),
    kStepsPerKernel,
    sample_manager.GetOclptxOptions().nsteps.value(), // max_steps
    min_steps,
    0, // Particles per side not determined here.
    env.GetEnvData()->nx,
    env.GetEnvData()->ny,
    env.GetEnvData()->nz,
    env.GetEnvData()->ns, // num_samples
    sample_manager.GetOclptxOptions().c_thr.value(), // curv threshold
    env.GetEnvData()->n_waypts,
    sample_manager.GetOclptxOptions().steplength.value(),
    env.GetEnvData()->lx,
    env.GetEnvData()->ly,
    env.GetEnvData()->lz,
    fibst,
    sample_manager.GetOclptxOptions().randfib.value(),
    sample_manager.GetOclptxOptions().fibthresh.value()
    }; // num waymasks.
  int num_dev = env.HowManyCQ();

  // Create a new oclptxhandler.
  OclPtxHandler *handler = new OclPtxHandler[num_dev];
  std::thread *gpu_managers[num_dev];

  int total_particles = 0;

  for (int i = 0; i < num_dev; ++i)
  {
    handler[i].Init(env.GetContext(),
                    env.GetCq(i),
                    env.GetKernel(i),
                    env.GetSumKernel(i),
                    &attrs,
                    global_fd,
                    env.GetEnvData(),
                    env.GetDevicePdf(i));

    total_particles += handler[i].particles_per_side();
  }

  ParticleGenerator particle_gen;
  particles_fifo = particle_gen.Init(total_particles);

  for (int i = 0; i < num_dev; ++i)
  {
    gpu_managers[i] = new std::thread(
        threading::RunThreads,
        &handler[i],
        particles_fifo,
        kNumReducers);
  }

  end_timer("set up OpenCL");

  puts("Tracking...");
  start_timer();

  int64_t count = 0;
  float percent;
  float rate;
  int64_t total = particle_gen.total_particles();
  while (count < total)
  {
    count = particles_fifo->count();
    percent = (100. * count) / total;
    t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> track_time = (t_end - t_start);

    rate = count / track_time.count();

    // Internally, we count each particle twice (once per direction).  The
    // user doesn't expect that, so we correct it here by dividing by two.
    printf("Processed %li/%li. [%2.2f%%] [%.f particles/sec]\r",
           count / 2, total / 2, percent, rate / 2);
    fflush(stdout);
    usleep(100000);  // .1s
  }
  printf("\n");

  for (int i = 0; i < num_dev; ++i)
  {
    gpu_managers[i]->join();
  }

  end_timer("track");

  puts("Writing to file...");
  start_timer();

  for (int i = 0; i < num_dev; ++i)
    handler[i].RunSumKernel();
  env.PdfsToFile("pdf_out");

  end_timer("write to file");

  delete[] handler;

  fclose(global_fd);

  return 0;
}
