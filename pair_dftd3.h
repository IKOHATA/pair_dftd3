/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(dftd3,PairDFTD3)
// clang-format on
#else

#ifndef LMP_PAIR_DFTD3_H
#define LMP_PAIR_DFTD3_H

#include "pair.h"

#define BOHRTOANG 0.529177249
#define BOHRTOANGPOW2 0.28002856085
#define BOHRTOANGPOW6 0.02195871819
#define BOHRTOANGPOW8 0.00614906825
#define AUTOEV 27.21138505

namespace LAMMPS_NS {

class PairDFTD3 : public Pair {
 public:
  PairDFTD3(class LAMMPS *);
  ~PairDFTD3() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  template <int SHIFT_FLAG, int EVFLAG, int EFLAG, int VFLAG_ATOM> void eval();

  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

  struct Singleparam {
    double rcov;    // Ang
    double r2r4;    // arb. unit.
    int ielement;
  };

  struct Doubleparam {
    double r0ab;
    double c6ab0[25];
    double c6ab1[25];
    double c6ab2[25];
    double c6ab0_sign[25];
    double cocut,cut,cocutsq,cutsq;
    int ielement, jelement;
  };

 protected:
  int nmax;
  Singleparam *singleparams;    // parameter set for an Za atom
  Doubleparam *doubleparams;    // parameter set for an Za-Zb atom pair
  int *elem2singleparam;        // mapping from element to parameters
  int **elem2doubleparam;       // mapping from element pair to parameters
  double cutmax;      // max cutoff for all elements
  double cut_global;                // max cutoff for atom pair
  double cut_coord;                // max cutoff for coordination number
  int maxshort;       // size of short neighbor list array
  int *neighshort;    // short neighbor list array
  int nparams1;                  // # of stored parameter sets
  int maxparam1;                 // max # of parameter sets
  int nparams2;
  int maxparam2; 

  int shift_flag;

  double *cn;                // Cn of each atom (nmax,)
  double *prefactor;


  double d3_k1, d3_k2, d3_k3;
  double s6, rs6, s18, rs18;

  virtual void allocate();
  virtual void read_file(char *);
  virtual void setup_params();

  void getc6ab(Doubleparam *, double, double, double &, double &, double &);


};

}    // namespace LAMMPS_NS

#endif
#endif
