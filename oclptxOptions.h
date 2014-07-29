/*  oclptxOptions.h

    Tim Behrens, Saad Jbabdi, FMRIB Image Analysis Group

    Copyright (C) 2010 University of Oxford  */

/*  Part of FSL - FMRIB's Software Library
    http://www.fmrib.ox.ac.uk/fsl
    fsl@fmrib.ox.ac.uk

    Developed at FMRIB (Oxford Centre for Functional Magnetic Resonance
    Imaging of the Brain), Department of Clinical Neurology, Oxford
    University, Oxford, UK


    LICENCE

    FMRIB Software Library, Release 5.0 (c) 2012, The University of
    Oxford (the "Software")

    The Software remains the property of the University of Oxford ("the
    University").

    The Software is distributed "AS IS" under this Licence solely for
    non-commercial use in the hope that it will be useful, but in order
    that the University as a charitable foundation protects its assets for
    the benefit of its educational and research purposes, the University
    makes clear that no condition is made or to be implied, nor is any
    warranty given or to be implied, as to the accuracy of the Software,
    or that it will be suitable for any particular purpose or for use
    under any specific conditions. Furthermore, the University disclaims
    all responsibility for the use which is made of the Software. It
    further disclaims any liability for the outcomes arising from using
    the Software.

    The Licensee agrees to indemnify the University and hold the
    University harmless from and against any and all claims, damages and
    liabilities asserted by third parties (including claims for
    negligence) which arise directly or indirectly from the use of the
    Software or the sale of any products based on the Software.

    No part of the Software may be reproduced, modified, transmitted or
    transferred in any form or by any means, electronic or mechanical,
    without the express permission of the University. The permission of
    the University is not required if the said reproduction, modification,
    transmission or transference is done without financial return, the
    conditions of this Licence are imposed upon the receiver of the
    product, and all original and amended source code is included in any
    transmitted product. You may be held legally responsible for any
    copyright infringement that is caused or encouraged by your failure to
    abide by these terms and conditions.

    You are not permitted under this Licence to use this Software
    commercially. Use for which any financial return is received shall be
    defined as commercial use, and includes (1) integration of all or part
    of the source code or the Software into a product for sale or license
    by or on behalf of Licensee to third parties or (2) use of the
    Software or any derivative of it for research with the final aim of
    developing software products for sale or license to a third party or
    (3) use of the Software or any derivative of it for research with the
    final aim of developing non-software products for sale or license to a
    third party, or (4) use of the Software to provide any service to an
    external organisation for which payment is received. If you are
    interested in using the Software commercially, please contact Isis
    Innovation Limited ("Isis"), the technology transfer company of the
    University, to negotiate a licence. Contact details are:
    innovation@isis.ox.ac.uk quoting reference DE/9564. */

#if !defined(oclptxOptions_h)
#define oclptxOptions_h

#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include "utils/options.h"
#include "utils/log.h"

using namespace Utilities;

class oclptxOptions {
  public:
    static oclptxOptions& getInstance();
    ~oclptxOptions() { delete gopt; }

    Option<int>              verbose;
    Option<bool>             help;

    Option<std::string>           basename;
    Option<std::string>           outfile;
    Option<std::string>           logdir;
    Option<bool>             forcedir;

    Option<std::string>           maskfile;
    Option<std::string>           seedfile;

    Option<bool>             simple;
    Option<bool>             network;
    Option<bool>             simpleout;
    Option<bool>             pathdist;
    Option<std::string>           pathfile;
    Option<bool>             s2tout;
    Option<bool>             s2tastext;

    Option<std::string>           targetfile;
    Option<std::string>           waypoints;
    Option<std::string>           waycond;
    Option<bool>             wayorder;
    Option<bool>             onewaycondition;
    // apply waypoint conditions to half the tract
    Option<std::string>           rubbishfile;
    Option<std::string>           stopfile;

    Option<bool>             matrix1out;
    Option<float>            distthresh1;
    Option<bool>             matrix2out;
    Option<std::string>           lrmask;
    Option<bool>             matrix3out;
    Option<std::string>           mask3;
    Option<std::string>           lrmask3;
    Option<float>            distthresh3;
    Option<bool>             matrix4out;
    Option<std::string>           mask4;
    Option<std::string>           dtimask;

    Option<std::string>           seeds_to_dti;
    Option<std::string>           dti_to_seeds;
    Option<std::string>           seedref;
    Option<std::string>           meshspace;

    Option<int>              nparticles;
    Option<int>              nsteps;
    Option<float>            steplength;


    Option<float>            distthresh;
    Option<float>            c_thr;
    Option<float>            fibthresh;
    Option<bool>             loopcheck;
    Option<bool>             usef;
    Option<bool>             modeuler;

    Option<float>            sampvox;
    Option<int>              randfib;
    Option<int>              fibst;
    Option<int>              rseed;

    Option<float>             mem_risk_frac;
    Option<bool>              norng;

    Option<std::string>       gpuselect;

    // hidden options
    FmribOption<std::string>      prefdirfile;
    // inside this mask, pick orientation closest to whatever is in here
    FmribOption<std::string>      skipmask;
    // inside this mask, ignore data (inertia)
    FmribOption<bool>        forcefirststep;
    // always take at least one step
    FmribOption<bool>        osampfib;
    // not yet
    FmribOption<bool>        onewayonly;
    // in surface mode, track towards the brain (assumes surface normal
    // points towards the brain)
    FmribOption<bool>        opathdir;
    // like fdt_paths but with average local tract orientation
    FmribOption<bool>        save_paths;
    // save paths to ascii file
    FmribOption<std::string>      locfibchoice;
    // inside this mask, define local rules for fibre picking
    FmribOption<std::string>      loccurvthresh;
    // inside this mask, define local curvature threshold
    FmribOption<bool>        targetpaths;
    // output separate fdt_paths for each target

    void parse_command_line(int argc, char** argv);
    void modecheck();
    void modehelp();
    void matrixmodehelp();
    void status();
  private:
    oclptxOptions();
    const oclptxOptions& operator=(oclptxOptions&);
    oclptxOptions(oclptxOptions&);

    OptionParser options;

    static oclptxOptions* gopt;
};


inline oclptxOptions::oclptxOptions():

  verbose(std::string("-V,--verbose"), 0,
      std::string("Verbose level, [0-2]"), false, requires_argument),
  help(std::string("-h,--help"), false,
    std::string("Display this message\n\n"), false, no_argument),

  basename(std::string("-s,--samples"),"",
    std::string("Basename for samples files - e.g. 'merged'"),
      true, requires_argument),

  outfile(std::string("-o,--out"), std::string("fdt_paths"),
   std::string("Output file (default='fdt_paths')"),
   false, requires_argument),
  logdir(std::string("--dir"), std::string("logdir"),
  std::string("\tDirectory to put the final volumes in - code makes this\
     directory - default='logdir'"),
  false, requires_argument),
  forcedir(std::string("--forcedir"), false,
    std::string("Use the actual directory name given - i.e. don't add\
      + to make a new directory\n\n"),
      false, no_argument),
  maskfile(std::string("-m,--mask"),"",
    std::string("Bet binary mask file in diffusion space"),
      true, requires_argument),
  seedfile(std::string("-x,--seed"),"",
    std::string("Seed volume or list (ascii text file) of volumes and/or\
       surfaces"), false, requires_argument),

  simple(std::string("--simple"),false,
  std::string("\tTrack from a list of voxels (seed must be a ASCII list of\
    coordinates)"), false, no_argument),
  network(std::string("--network"), false,
   std::string("Activate network mode - only keep paths going through at\
    least one of the other seed masks"), false, no_argument),
  simpleout(std::string("--opd"), false,
    std::string("\tOutput path distribution"),
      false, no_argument),
  pathdist(std::string("--pd"), false,
    std::string("\tCorrect path distribution for the length of the pathways"),
    false, no_argument),
  pathfile(std::string("--fopd"), "",
    std::string("\tOther mask for binning tract distribution"),
    false, requires_argument),
  s2tout(std::string("--os2t"), false,
  std::string("\tOutput seeds to targets"),
    false, no_argument),
  s2tastext(std::string("--s2tastext"), false,
     std::string("Output seed-to-target counts as a text file\
      (default in simple mode)\n\n"), false, no_argument),
  targetfile(std::string("--targetmasks"),"",
      std::string("File containing a list of target masks - for seeds_to_targets classification"),
      false, requires_argument),
      
  waypoints(std::string("--waypoints"), std::string(""),
     std::string("Waypoint mask or ascii list of waypoint masks - only keep paths going through ALL the masks"),
     false, requires_argument),
  waycond(std::string("--waycond"),"AND",
   std::string("Waypoint condition. Either 'AND' (default) or 'OR'"),
   false, requires_argument),
  wayorder(std::string("--wayorder"),false,
    std::string("Reject streamlines that do not hit waypoints in given order. Only valid if waycond=AND"),
    false,no_argument),
    
  onewaycondition(std::string("--onewaycondition"),false,
    std::string("Apply waypoint conditions to each half tract separately"),
    false, no_argument),
  rubbishfile(std::string("--avoid"), std::string(""),
       std::string("\tReject pathways passing through locations given by this mask"),
       false, requires_argument),
  stopfile(std::string("--stop"), std::string(""),
       std::string("\tStop tracking at locations given by this mask file\n\n"),
       false, requires_argument),

  matrix1out(std::string("--omatrix1"), false,
      std::string("Output matrix1 - SeedToSeed Connectivity"),
      false, no_argument),
  distthresh1(std::string("--distthresh1"), 0,
       std::string("Discards samples (in matrix1) shorter than this threshold (in mm - default=0)"),
       false, requires_argument),
  matrix2out(std::string("--omatrix2"), false,
      std::string("Output matrix2 - SeedToLowResMask"),
      false, no_argument),
  lrmask(std::string("--target2"), std::string(""),
  std::string("Low resolution binary brain mask for storing connectivity distribution in matrix2 mode"),
  false, requires_argument),
  matrix3out(std::string("--omatrix3"), false,
      std::string("Output matrix3 (NxN connectivity matrix)"),
      false, no_argument),
  mask3(std::string("--target3"), "",
  std::string("Mask used for NxN connectivity matrix (or Nxn if lrtarget3 is set)"),
  false, requires_argument),
  lrmask3(std::string("--lrtarget3"), "",
  std::string("Column-space mask used for Nxn connectivity matrix"),
  false, requires_argument),
  distthresh3(std::string("--distthresh3"), 0,
       std::string("Discards samples (in matrix3) shorter than this threshold (in mm - default=0)"),
       false, requires_argument),
  matrix4out(std::string("--omatrix4"), false,
      std::string("Output matrix4 - DtiMaskToSeed (special Oxford Sparse Format)"),
      false, no_argument),
  mask4(std::string("--colmask4"), std::string(""),
  std::string("Mask for columns of matrix4 (default=seed mask)"),
  false, requires_argument),
  dtimask(std::string("--target4"), std::string(""),
  std::string("Brain mask in DTI space\n\n"),
  false, requires_argument),

  seeds_to_dti(std::string("--xfm"),"",
  std::string("\tTransform taking seed space to DTI space (either FLIRT matrix or FNIRT warpfield) - default is identity"),
  false, requires_argument),
  dti_to_seeds(std::string("--invxfm"), std::string(""),
  std::string("Transform taking DTI space to seed space (compulsory when using a warpfield for seeds_to_dti)"),
  false, requires_argument),
  seedref(std::string("--seedref"),"",
   std::string("Reference vol to define seed space in simple mode - diffusion space assumed if absent"),
   false, requires_argument),
  meshspace(std::string("--meshspace"), std::string("caret"),
     std::string("Mesh reference space - either 'caret' (default) or 'freesurfer' or 'first' or 'vox' \n\n"),
     false, requires_argument),

  nparticles(std::string("-P,--nsamples"), 5000,
      std::string("Number of samples - default=5000"),
      false, requires_argument),
  nsteps(std::string("-S,--nsteps"), 2000,
  std::string("Number of steps per sample - default=2000"),
  false, requires_argument),
  steplength(std::string("--steplength"), 0.5,
      std::string("Steplength in mm - default=0.5\n\n"),
      false, requires_argument),

  distthresh(std::string("--distthresh"), 0,
      std::string("Discards samples shorter than this threshold (in mm - default=0)"),
      false, requires_argument),
  c_thr(std::string("-c,--cthr"), 0.2,
  std::string("Curvature threshold - default=0.2"),
  false, requires_argument),
  fibthresh(std::string("--fibthresh"), 0.01,
     std::string("Volume fraction before subsidary fibre orientations are considered - default=0.01"),
     false, requires_argument),
  loopcheck(std::string("-l,--loopcheck"), false,
     std::string("Perform loopchecks on paths - slower, but allows lower curvature threshold"),
     false, no_argument),
     
  usef(std::string("-f,--usef"), false,
    std::string("Use anisotropy to constrain tracking"), false, no_argument),

  modeuler(std::string("--modeuler"), false,
    std::string("Use modified euler streamlining\n\n"), false, no_argument),


  sampvox(std::string("--sampvox"), 0,
   std::string("Sample random points within x mm sphere seed voxels\
    (e.g. --sampvox=5). Default=0"),  true, requires_argument),

  randfib(std::string("--randfib"), 0,
   std::string("Default 0. Set to 1 to randomly sample initial fibres\
      (with f > fibthresh). \n Set to 2 to sample in proportion fibres\
          (with f>fibthresh) to f. \n Set to 3 to sample ALL populations at\
            random (even if f<fibthresh)"), false, requires_argument),

  fibst(std::string("--fibst"),1,
    std::string("\tForce a starting fibre for tracking - default=1, i.e. first\
      fibre orientation. Only works if randfib==0"), false, requires_argument),

  rseed(std::string("--rseed"), 0,
    std::string("\tRandom seed"), false, requires_argument),

  mem_risk_frac(std::string("--memrisk"), 0.8,
  std::string("Threshold for fraction (0.0 to 1.0) of \
    CL_DEVICE_GLOBAL_MEM_SIZE the application will attempt to \
      use per device"), false, optional_argument),

  norng(std::string("--norng"), false,
  std::string("Deterministic tracking only (zero sample, no vol frac."),
    false, no_argument),

  gpuselect(std::string("--gpuselect"), "",
    std::string("Run with specified gpu devices ONLY."),
      false, requires_argument),

  prefdirfile(std::string("--prefdir"), std::string(""),
       std::string("Prefered orientation preset in a 4D mask"),
       false, requires_argument),
  skipmask(std::string("--no_integrity"), std::string(""),
    std::string("No explanation needed"),
    false, requires_argument),
  forcefirststep(std::string("--forcefirststep"),false,
    std::string("In case seed and stop masks are the same"),
    false, no_argument),
  osampfib(std::string("--osampfib"),false,
    std::string("Output sampled fibres"),
    false, no_argument),
  onewayonly(std::string("--onewayonly"),false,
      std::string("Track in one direction only (towards the brain - only\
        valid for surface seeds)"), false, no_argument),
  opathdir(std::string("--opathdir"),false,
    std::string("Output average local tract orientation (tangent)"),
      false, no_argument),
  save_paths(std::string("--savepaths"),false,
    std::string("Save path coordinates to ascii file"),
      false, no_argument),
  locfibchoice(std::string("--locfibchoice"),std::string(""),
    std::string("Local rules for fibre choice - 0=closest direction(default),\
      1=equal prob (f>thr), 2=equal prob with angle threshold (=40 deg)"),
        false, requires_argument),
  loccurvthresh(std::string("--loccurvthresh"),std::string(""),
      std::string("Local curvature threshold"),
      false, requires_argument),
  targetpaths(std::string("--otargetpaths"),false,
      std::string("Output separate fdt_paths for targets (assumes --os2t\
         is on)"), false, no_argument),

  options("oclptx","oclptx -s <basename> -m <maskname> -x <seedfile>\
  -o <output> --targetmasks=<textfile>\n oclptx --help\n")
{
  try
  {
    options.add(verbose);
    options.add(help);

    options.add(basename);
    options.add(outfile);
    options.add(logdir);
    options.add(forcedir);

    options.add(maskfile);
    options.add(seedfile);

    options.add(simple);
    options.add(network);
    options.add(simpleout);
    options.add(pathdist);
    options.add(pathfile);
    options.add(s2tout);
    options.add(s2tastext);

    options.add(targetfile);
    options.add(waypoints);
    options.add(waycond);
    options.add(wayorder);
    options.add(onewaycondition);
    options.add(rubbishfile);
    options.add(stopfile);

    options.add(matrix1out);
    options.add(distthresh1);
    options.add(matrix2out);
    options.add(lrmask);
    options.add(matrix3out);
    options.add(mask3);
    options.add(lrmask3);
    options.add(distthresh3);
    options.add(matrix4out);
    options.add(mask4);
    options.add(dtimask);

    options.add(seeds_to_dti);
    options.add(dti_to_seeds);
    options.add(seedref);
    options.add(meshspace);


    options.add(nparticles);
    options.add(nsteps);
    options.add(steplength);

    options.add(distthresh);
    options.add(c_thr);
    options.add(fibthresh);
    options.add(loopcheck);
    options.add(usef);
    options.add(modeuler);

    options.add(sampvox);
    options.add(randfib);
    options.add(fibst);
    options.add(rseed);

    options.add(skipmask);
    options.add(prefdirfile);
    options.add(forcefirststep);
    options.add(osampfib);
    options.add(onewayonly);
    options.add(opathdir);
    options.add(save_paths);
    options.add(locfibchoice);
    options.add(loccurvthresh);
    options.add(targetpaths);

    options.add(mem_risk_frac);
    options.add(norng);
    options.add(gpuselect);
  }
  catch(X_OptionError& e)
  {
    options.usage();
    std::cerr << std::endl << e.what() << std::endl;
  }
  catch(std::exception &e)
  {
    std::cerr << e.what() << std::endl;
  }
}


#endif







