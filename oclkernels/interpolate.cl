/* Copyright 2014
 *  Afshin Haidari
 *  Steve Novakov
 *  Jeff Taylor
 *
 * This kernel traces out the paths taken by particles.  Each thread acts as a
 * single particle.
 */
 
#include "attrs.h"
#include "rbtree.h"
#include "rng.h"

float3 f_theta_phi_to_xyz(float3 f_theta_phi)
{
  float3 xyz;

  xyz.s0 = cos(f_theta_phi.s2) * sin(f_theta_phi.s1);
  xyz.s1 = sin(f_theta_phi.s2) * sin(f_theta_phi.s1);
  xyz.s2 = cos(f_theta_phi.s1);

  return xyz;
}

float3 get_f_theta_phi(global float *f_samples,
                       global float *theta_samples,
                       global float *phi_samples,
                       float3 particle_pos,
                       const struct particle_attrs attrs,
                       global rng_t *rng)
{
  /* calculate current index in diffusion space */
  ulong3 rng_output;
  float f = 0.;
  float theta = 0.;
  float phi = 0.;
  uint diffusion_index;
  uint sample;

  uint3 current_select_vertex = convert_uint3(floor(particle_pos));
  float3 volume_fraction = particle_pos - convert_float3(current_select_vertex);
  float3 vol_frac = volume_fraction * kRandMax;

  /* Pick Sample */
  sample = Rand(rng) % attrs.num_samples;

  /* Volume Fraction Selection */
  rng_output = (ulong3) (Rand(rng), Rand(rng), Rand(rng));

  current_select_vertex += (convert_float3(rng_output) > vol_frac)? 1: 0;

  /* pick flow vertex */
  diffusion_index =
    sample*(attrs.sample_nz*attrs.sample_ny*attrs.sample_nx)+
    current_select_vertex.s0*(attrs.sample_nz*attrs.sample_ny) +
    current_select_vertex.s1*(attrs.sample_nz) +
    current_select_vertex.s2;

  if (f_samples)
    f = f_samples[diffusion_index];
  theta = theta_samples[diffusion_index];
  phi = phi_samples[diffusion_index];

  return (float3) (f, theta, phi);
}

#if WAYAND
#define WAYOP(chk, pts) ((chk) &= (pts))
#else  /* WAYOR */
#define WAYOP(chk, pts) ((chk) |= (pts))
#endif

void do_particle_finish(uint glid,
                        const struct particle_attrs attrs,
                        ushort done,
                        ushort steps, //RW
                        global ushort *particle_exclusion,
                        global ushort *particle_waypoints,
                        global struct rbtree *position_set,
                        global uint *global_pdf)
{
  int i;
  int waypoint_check;
  if (steps < attrs.min_steps)
    return;

#if EXCLUSION
  if (particle_exclusion[glid])
    return;
#endif  /* EXCLUSION */

#if WAYPOINTS
  /* TODO(jeff/steve): double check this code.  WAYOR seems broken to Jeff. */
  waypoint_check = 1;
  for (i = 0; i < attrs.n_waypoint_masks; i++)
    WAYOP(waypoint_check, particle_waypoints[glid*attrs.n_waypoint_masks + i]);

  if (waypoint_check == 0)
    return;
#endif  /* WAYPOINTS */

  if ((done)
   && (BREAK_INVALID  != done)
   && (BREAK_INIT     != done)
   && (STILL_FINISHED != done)) {
    /* Walk the tree out of order */
    for (i = 0; i < position_set->num_entries; ++i) {
      /* position = x*ny*nz + y*nz + z */
      int index = rbtree_data(position_set, i);

      atomic_inc(&global_pdf[index]);
    }
  }
}

__kernel void OclPtxInterpolate(
  struct particle_attrs attrs,  /* RO */
  __global struct particle_data *state,  /* RW */
  __global struct rbtree *position_set, /* RW */

  // Debugging info
  __global float3 *particle_paths, //RW
  __global ushort *particle_steps, //RW

  // Output
  __global ushort *particle_done, //RW
  __global uint   *global_pdf, //RW
  __global ushort *particle_waypoints, //W
  __global ushort *particle_exclusion, //W
  __global float3 *particle_loopcheck_lastdir, //RW

  // Global Data
  __global float *f_samples, //R
  __global float *phi_samples, //R
  __global float *theta_samples, //R
  __global float *f_samples_2,  //R
  __global float *phi_samples_2,  //R
  __global float *theta_samples_2,  //R
  __global ushort *brain_mask, //R
  __global ushort *waypoint_masks,  //R
  __global ushort *termination_mask,  //R
  __global ushort *exclusion_mask //R
)
{
  uint glid = get_global_id(0);
  
  int i;
  uint path_index;
  uint step;
  uint mask_index;
  ushort bounds_test;
  uint vertex_num;
  uint entry_num;
  uint shift_num;
  float3 f_theta_phi;
  float3 temp_pos = state[glid].position;
  float3 new_dr = (float3) (0.0f);
  float3 min = (float3) (0.0f);
  float3 max = (float3) (attrs.sample_nx * 1.0,
                         attrs.sample_ny * 1.0,
                         attrs.sample_nz * 1.0);

#ifdef WAYPOINTS
  uint mask_size = attrs.sample_nx * attrs.sample_ny * attrs.sample_nz;
#endif
#ifdef EULER_STREAMLINE
  float3 dr2 = (float3) (0.0f);
#endif
#ifdef LOOPCHECK
  uint3 loopcheck_voxel;

  uint loopcheck_dir_size = attrs.lx * attrs.ly * attrs.lz;

  uint loopcheck_index;
  float3 last_loopcheck_dr;
  float loopcheck_product;
#endif // LOOPCHECK

  /* No new valid data.  Likely the host is out of data.  Signal that. */
  if (particle_done[glid])
  {
    particle_done[glid] = STILL_FINISHED;
    particle_steps[glid] = 0;
    return;
  }

  /* New particle.  Do any in-kernel initialization here. */
  /* TODO(jeff): Initialize waymasks, etc. here instead of in oclptxhandler for
   * possible performance improvement? */
  if (0 == particle_steps[glid])
    rbtree_init(&position_set[glid]);

  /* Main loop */
  for (step = 0; step < attrs.steps_per_kernel; ++step)
  {
    f_theta_phi = get_f_theta_phi(f_samples, theta_samples, phi_samples,
                                  temp_pos, attrs, &(state[glid].rng));

    new_dr = f_theta_phi_to_xyz(f_theta_phi);
    
    /* Align direction to keep angle under 90 degrees */
    if (dot(new_dr, state[glid].dr) < 0.0 )
      new_dr *= -1;

    new_dr = new_dr / attrs.brain_mask_dim;
    new_dr = new_dr * attrs.step_length;

#ifdef ANISOTROPIC
    if (f_theta_phi.s0 * kRandMax < Rand(&(state[glid].rng)))
    {
      particle_done[glid] = ANISO_BREAK;
      break;
    }
#endif /* ANISOTROPIC */

#ifdef EULER_STREAMLINE
    // update particle position
    temp_pos = state[glid].position + new_dr;

    f_theta_phi = get_f_theta_phi(f_samples, theta_samples, phi_samples,
                                  temp_pos, &attrs, &(state[glid].rng));

#ifdef ANISOTROPIC
    if (f_theta_phi.s0 * kRandMax < Rand(&(state[glid].rng)))
    {
      particle_done[glid] = ANISO_BREAK;
      break;
    }
#endif // ANISOTROPIC
    
    dr2 = f_theta_phi_to_xyz(f_theta_phi);

    /* Keep angle under 90 degrees */
    if (dot(dr2, state[glid].dr) < 0.0 )
      dr2 *= -1;

    dr2 = dr2 / attrs.brain_mask_dim;
    dr2 = dr2 * attrs.step_length;

    new_dr = 0.5 * (new_dr + dr2);
#endif  /* EULER_STREAMLINE */

    temp_pos = state[glid].position + new_dr;

    /* Curvature threshold check */
    new_dr = normalize(new_dr);
    if (particle_steps[glid] > 1 
      && dot(new_dr, state[glid].dr) < attrs.curvature_threshold)
    {
      particle_done[glid] = BREAK_CURV;
      break;
    }

    /* Out of bounds? */
    if (any(temp_pos > max || min > temp_pos))
    {
      particle_done[glid] = BREAK_INVALID;
      break;
    }

    /* Brain Mask Test - Checks NEAREST vertex. */
    mask_index =
      round(temp_pos.s0)*(attrs.sample_nz*attrs.sample_ny) +
      round(temp_pos.s1)*(attrs.sample_nz) + round(temp_pos.s2);

    bounds_test = brain_mask[mask_index];
    if (bounds_test == 0)
    {
      particle_done[glid] = BREAK_BRAIN_MASK;
      break;
    }

#ifdef TERMINATION
    bounds_test = termination_mask[mask_index];
    if (bounds_test == 1)
    {
      particle_done[glid] = BREAK_TERM;
      break;
    }
#endif  /* TERMINATION */

#ifdef EXCLUSION
    bounds_test = exclusion_mask[mask_index];
    if (bounds_test == 1)
    {
      particle_exclusion[glid] = 1;
      particle_done[glid] = BREAK_EXCLUSION;
      break;
    }
#endif  /* EXCLUSION */

#ifdef WAYPOINTS
    for (uint w = 0; w < attrs.n_waypoint_masks; w++)
    {
      bounds_test = waypoint_masks[w*mask_size + mask_index];
      if (bounds_test > 0)
        particle_waypoints[glid*attrs.n_waypoint_masks + w] |= 1;
    }
#endif  /* WAYPOINTS */

#ifdef LOOPCHECK
  loopcheck_voxel = convert_uint3(round(temp_pos) / 5);

  loopcheck_index = loopcheck_voxel.s0*(attrs.ly*attrs.lz) +
    loopcheck_voxel.s1*attrs.lz + loopcheck_voxel.s2;

  last_loopcheck_dr =
      particle_loopcheck_lastdir[glid*loopcheck_dir_size +
        loopcheck_index];

  loopcheck_product = last_loopcheck_dr.s0*new_dr.s0 +
    last_loopcheck_dr.s1*new_dr.s1 + last_loopcheck_dr.s2*new_dr.s2;

  if (loopcheck_product < 0)
  {
    particle_done[glid] = BREAK_LOOPCHECK;
    break;
  }

  particle_loopcheck_lastdir[glid*loopcheck_dir_size +
    loopcheck_index] = new_dr;

#endif  /* LOOPCHECK */

    /* update step location */
    state[glid].position = temp_pos;

    /* update last flow vector */
    state[glid].dr = new_dr;

    /* add to particle paths */
    path_index = glid * attrs.steps_per_kernel + step;

    if (particle_paths)
      particle_paths[path_index] = temp_pos;
  
    /* Add position to position list */
    uint index = floor(temp_pos.x) * attrs.sample_ny * attrs.sample_nz
               + floor(temp_pos.y) * attrs.sample_nz
               + floor(temp_pos.z);
    rbtree_insert(&position_set[glid], index);
    
    if (particle_steps[glid] + 1 == attrs.max_steps) {
      particle_done[glid] = BREAK_MAXSTEPS;
      break;
    }

    if (!particle_done[glid])
    {
      particle_steps[glid] += 1;
    }
  } /* Main loop */

  /* If the host is reading path data, no new data has been added.  We need to
   * signal that.
   */
  if (0 == step)
    particle_steps[glid] = 0;

  /* If finished, add steps to global pdf by walking the set out-of-order */
  do_particle_finish(glid,
                     attrs,
                     particle_done[glid],
                     particle_steps[glid],
                     particle_exclusion,
                     particle_waypoints,
                     &position_set[glid],
                     global_pdf);
}
