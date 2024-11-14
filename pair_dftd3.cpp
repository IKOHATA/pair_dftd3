// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Ikuma Kohata (The University of Tokyo)
------------------------------------------------------------------------- */

#include "pair_dftd3.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "math_extra.h"
#include "math_special.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include "suffix.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <float.h>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;
using namespace MathExtra;

#define DELTA 4

/* ---------------------------------------------------------------------- */

PairDFTD3::PairDFTD3(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  centroidstressflag = CENTROID_NOTAVAIL;
  unit_convert_flag = utils::get_supported_conversions(utils::ENERGY);
  
  nmax = 0;
  singleparams = nullptr;
  doubleparams = nullptr;
  elem2singleparam = nullptr;
  elem2doubleparam = nullptr;

  cn = nullptr;
  prefactor = nullptr;

  maxshort = 10;
  neighshort = nullptr;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairDFTD3::~PairDFTD3()
{
  if (copymode) return;

  memory->destroy(singleparams);
  memory->destroy(doubleparams);
  memory->destroy(elem2singleparam);
  memory->destroy(elem2doubleparam);
  memory->destroy(cn);
  memory->destroy(prefactor);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(neighshort);
  }
}

/* ---------------------------------------------------------------------- */

void PairDFTD3::compute(int eflag, int vflag)
{
  int i,j,k,l,ii,jj,kk,ll,inum,jnum,lnum;
  int itype,jtype,ktype,ltype,iparam_ij,iparam_ji,iparam_i,iparam_j,iparam_k;
  tagint itag,jtag;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double jxtmp,jytmp,jztmp;
  double rsq,rsq1,rsq2,r1,r2,expco;
  double rco, rr, cnab, pf_i, pf_j, ddamp;
  double delr1[3],delr2[3],fi[3],fj[3],fk[3];
  double r1_hat[3],r2_hat[3];
  int *ilist,*jlist,*llist,*numneigh,**firstneigh;

  evdwl = 0.0;
  ev_init(eflag,vflag);

  if (atom->nmax > nmax) {
    memory->destroy(cn);
    memory->destroy(prefactor);
    nmax = atom->nmax;
    memory->create(cn,nmax,"pair:cn");
    memory->create(prefactor,nmax,"pair:prefactor");
  }

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int newton_pair = force->newton_pair;
  const double cutshortsq = cutmax*cutmax;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double fxtmp,fytmp,fztmp;

  if (newton_pair) {
    for (i = 0; i < nall; i++) {
      cn[i] = 0.0; 
      prefactor[i] = 0.0;
    }
  } else for (i = 0; i < nlocal; i++){
      cn[i] = 0.0; 
      prefactor[i] = 0.0;
  }

  // loop over full neighbor list of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = map[type[i]];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    fxtmp = fytmp = fztmp = 0.0;

    // two-body interactions, skip half of them

    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      // shift rsq and store correction for force

      jtype = map[type[j]];
      iparam_ij = elem2doubleparam[itype][jtype];
      if (rsq >= doubleparams[iparam_ij].cocutsq) continue;

      iparam_i = elem2singleparam[itype];
      iparam_j = elem2singleparam[jtype];
      
      rco = singleparams[iparam_i].rcov+singleparams[iparam_j].rcov;

      r1 = sqrt(rsq);

      expco = exp(-d3_k1*(rco/r1-1.0));
      cnab = 1.0/(1.0+expco);
      
      cn[i] += cnab;
    }
  }

  //comm->forward_comm_pair(this);
  comm->forward_comm(this);

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = map[type[i]];
    iparam_i = elem2singleparam[itype];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    fxtmp = fytmp = fztmp = 0.0;

    // two-body interactions, skip half of them

    jlist = firstneigh[i];
    jnum = numneigh[i];

    // three-body interactions
    // skip immediately if I-J is not within cutoff
    double fjxtmp,fjytmp,fjztmp; 
    double c6ab,c8; 
    double tmp,tmp2,tmp6,tmp8;
    double r5,r6,r7,r8,e6,e8;
    double fpair,kp,f6ab,f8ab,f6ab1,f8ab1,dc6,dc6a,dc6b,prefactora,prefactorb,f2x,f2y,f2z;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      llist = firstneigh[j];
      lnum = numneigh[j];
      
      jtype = map[type[j]];
      iparam_j = elem2singleparam[jtype];
      iparam_ij = elem2doubleparam[itype][jtype];

      delr1[0] = xtmp - x[j][0];
      delr1[1] = ytmp - x[j][1];
      delr1[2] = ztmp - x[j][2];
      rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];

      if (rsq1 >= doubleparams[iparam_ij].cutsq) continue;

      jxtmp = x[j][0];
      jytmp = x[j][1];
      jztmp = x[j][2];

      fjxtmp = fjytmp = fjztmp = 0.0;

      r1 = sqrt(rsq1);

      r6 = cube(rsq1);
      r5 = r6 / r1;
      r8 = r6 * rsq1;
      r7 = r8 / r1;

      kp = 3.0 * singleparams[iparam_i].r2r4 * singleparams[iparam_j].r2r4;

      tmp = rs6 * sqrt(kp) + rs18;
      tmp2 = square(tmp);
      tmp6 = cube(tmp2);
      tmp8 = tmp6 * tmp2;
      e6 = -1.0 / (r6 + tmp6);
      f6ab = -6.0 * square(e6) * r5;

      e8 = -1.0 / (r8 + tmp8);
      f8ab = -8.0 * square(e8) * r7;

      e6 = 0.5 * s6 * e6;
      e8 = 0.5 * s18 * e8;

      getc6ab(&doubleparams[iparam_ij],cn[i],cn[j],c6ab,dc6a,dc6b);

      c8 = kp * c6ab;
      f6ab1 = c6ab * f6ab * s6 * 0.5;
      f8ab1 = c8 * f8ab * s18 * 0.5;

      evdwl = (c6ab * e6 + c8 * e8);

      fpair = (f6ab1+f8ab1)/r1;

      prefactora = dc6a*(e6+kp*e8);
      prefactorb = dc6b*(e6+kp*e8);

      prefactor[i] += prefactora;
      prefactor[j] += prefactorb;

      fxtmp += delr1[0]*fpair;
      fytmp += delr1[1]*fpair;
      fztmp += delr1[2]*fpair;
      f[j][0] -= delr1[0]*fpair;
      f[j][1] -= delr1[1]*fpair;
      f[j][2] -= delr1[2]*fpair;

      if (evflag) ev_tally(i,j,nlocal,newton_pair,
                           evdwl,0.0,fpair,delr1[0],delr1[1],delr1[2]);

      // attractive term via loop over j
    }
    f[i][0] += fxtmp;
    f[i][1] += fytmp;
    f[i][2] += fztmp;
  }

  comm->reverse_comm(this);

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = map[type[i]];
    iparam_i = elem2singleparam[itype];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    fxtmp = fytmp = fztmp = 0.0;

    // two-body interactions, skip half of them

    jlist = firstneigh[i];
    jnum = numneigh[i];

    pf_i = prefactor[i];

    for (jj = 0; jj < jnum; jj++) {
      
      j = jlist[jj];
      j &= NEIGHMASK;

      pf_j = prefactor[j];
      
      jtype = map[type[j]];
      iparam_j = elem2singleparam[jtype];
      iparam_ij = elem2doubleparam[itype][jtype];

      delr1[0] = x[j][0] - xtmp;
      delr1[1] = x[j][1] - ytmp;
      delr1[2] = x[j][2] - ztmp;
      rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];
      if (rsq1 >= doubleparams[iparam_ij].cocutsq) continue;

      r1 = sqrt(rsq1);

      rco = singleparams[iparam_i].rcov+singleparams[iparam_j].rcov;

      expco = exp(-d3_k1*(rco/r1-1.0));
      cnab = 1.0/(1.0+expco);
      ddamp = -expco*d3_k1*rco*square(cnab)/rsq1;

      fpair = ddamp*pf_i/r1;

      fxtmp += delr1[0]*fpair;
      fytmp += delr1[1]*fpair;
      fztmp += delr1[2]*fpair;
      f[j][0] -= delr1[0]*fpair;
      f[j][1] -= delr1[1]*fpair;
      f[j][2] -= delr1[2]*fpair;

      if (evflag) ev_tally(i,j,nlocal,newton_pair,
                           0.0,0.0,-fpair,-delr1[0],-delr1[1],-delr1[2]);
    }

    f[i][0] += fxtmp;
    f[i][1] += fytmp;
    f[i][2] += fztmp;
  }


  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairDFTD3::allocate()
{
  allocated = 1;
  //int n = atom->ntypes;
  int np1 = atom->ntypes + 1;

  memory->create(setflag,np1,np1,"pair:setflag");
  memory->create(cutsq,np1,np1,"pair:cutsq");
  memory->create(neighshort,maxshort,"pair:neighshort");
  map = new int[np1];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairDFTD3::settings(int narg, char **arg)
{

  if (narg != 3) error->all(FLERR,"Illegal pair_style command");

  d3_k1 = 16.000;
  d3_k2 = 4.00 / 3.00;
  d3_k3 = 4.000;

  s6 = 1.0*AUTOEV*BOHRTOANGPOW6;

  if (strcmp(arg[0],"pbe") == 0) {
    rs6 = 0.4289*BOHRTOANG;
    s18 = 0.7875*AUTOEV*BOHRTOANGPOW8;
    rs18 = 4.4407*BOHRTOANG;
  } else if (strcmp(arg[0],"b3-lyp") == 0){
    rs6 = 0.3981*BOHRTOANG;
    s18 = 1.9889*AUTOEV*BOHRTOANGPOW8;
    rs18 = 4.4211*BOHRTOANG;
  } else {
    error->all(FLERR,"Illegal xc functional");
  }
  cut_global = utils::numeric(FLERR,arg[1],false,lmp);
  cut_coord = utils::numeric(FLERR,arg[2],false,lmp);

}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairDFTD3::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  map_element2type(narg-3,arg+3);

  // read potential file and initialize potential parameters

  read_file(arg[2]);
  setup_params();
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairDFTD3::init_style()
{
  //if (atom->tag_enable == 0)
    //error->all(FLERR,"Pair style DFTD3 requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style DFTD3 requires newton pair on");

  // need a full neighbor list

  neighbor->add_request(this,NeighConst::REQ_FULL);
  //neighbor->add_request(this);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairDFTD3::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");
  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairDFTD3::read_file(char *file)
{
  memory->sfree(singleparams);
  singleparams = nullptr;
  nparams1 = maxparam1 = 0;

  memory->sfree(doubleparams);
  doubleparams = nullptr;
  nparams2 = maxparam2 = 0;

  // open file on proc 0

  if (comm->me == 0) {
    PotentialFileReader reader(lmp, file, "dftd3", unit_convert_flag);
    char *line;

    // transparently convert units for supported conversions

    int unit_convert = reader.get_unit_convert();
    double conversion_factor = utils::get_conversion_factor(utils::ENERGY,unit_convert);

    int count = 1;

    while (count<95) {
      count++;
      line = reader.next_line(3);
      try {
        ValueTokenizer values(line);

        std::string iname = values.next_string();

        //std::cout << iname << "\n";

        // ielement,jelement,kelement = 1st args
        // if all 3 args are in element list, then parse this line
        // else skip to next entry in file
        int ielement;

        for (ielement = 0; ielement < nelements; ielement++)
          if (iname == elements[ielement]) break;
        if (ielement == nelements) continue;

        // load up parameter settings and error check their values

        if (nparams1 == maxparam1) {
          maxparam1 += DELTA;
          singleparams = (Singleparam *) memory->srealloc(singleparams,maxparam1*sizeof(Singleparam), "pair:singleparams");

          // make certain all addional allocated storage is initialized
          // to avoid false positives when checking with valgrind

          memset(singleparams + nparams1, 0, DELTA*sizeof(Singleparam));
        }

        singleparams[nparams1].ielement  = ielement;
        singleparams[nparams1].rcov = values.next_double()*0.52917726;
        singleparams[nparams1].r2r4 = values.next_double();

        //std::cout << singleparams[nparams1].r2r4 << "\n";
        

      } catch (TokenizerException &e) {
        error->one(FLERR, e.what());
      }

      nparams1++;
    }

    while ((line = reader.next_line(2))) {
      try {
        ValueTokenizer values(line);

        std::string iname = values.next_string();
        std::string jname = values.next_string();

        //std::cout << iname << " " << jname << "\n";

        // ielement,jelement,kelement = 1st args
        // if all 3 args are in element list, then parse this line
        // else skip to next entry in file
        int ielement, jelement;

        for (ielement = 0; ielement < nelements; ielement++)
          if (iname == elements[ielement]) break;
        if (ielement == nelements){
          for (int i=0;i<4;i++){reader.skip_line();}
          continue;
        }
        for (jelement = 0; jelement < nelements; jelement++)
          if (jname == elements[jelement]) break;
        if (jelement == nelements) {
          for (int i=0;i<4;i++){reader.skip_line();}
          continue;
        }

        // load up parameter settings and error check their values

        if (nparams2 == maxparam2) {
          maxparam2 += DELTA;
          doubleparams = (Doubleparam *) memory->srealloc(doubleparams,maxparam2*sizeof(Doubleparam), "pair:doubleparams");

          // make certain all addional allocated storage is initialized
          // to avoid false positives when checking with valgrind

          memset(doubleparams + nparams2, 0, DELTA*sizeof(Doubleparam));
        }

        doubleparams[nparams2].ielement  = ielement;
        doubleparams[nparams2].jelement  = jelement;
        //std::cout << ielement << " " << jelement << "\n";
        //std::cout << iname << " " << jname << "\n";
        doubleparams[nparams2].r0ab= reader.next_double();
        //std::cout << doubleparams[nparams2].r0ab << "\n";
        reader.next_dvector(doubleparams[nparams2].c6ab0, 25);
        reader.next_dvector(doubleparams[nparams2].c6ab1, 25);
        reader.next_dvector(doubleparams[nparams2].c6ab2, 25);

        for (int i=0;i<25;i++){
          if(doubleparams[nparams2].c6ab0[i]>0.0){
            doubleparams[nparams2].c6ab0_sign[i] = 1.0;
          }else{
            doubleparams[nparams2].c6ab0_sign[i] = 0.0;
          }
        }

        

      } catch (TokenizerException &e) {
        error->one(FLERR, e.what());
      }

      nparams2++;
    }
  }

  MPI_Bcast(&nparams1, 1, MPI_INT, 0, world);
  MPI_Bcast(&maxparam1, 1, MPI_INT, 0, world);
  MPI_Bcast(&nparams2, 1, MPI_INT, 0, world);
  MPI_Bcast(&maxparam2, 1, MPI_INT, 0, world);

  if (comm->me != 0) {
    singleparams = (Singleparam *) memory->srealloc(singleparams,maxparam1*sizeof(Singleparam), "pair:singleparams");
    doubleparams = (Doubleparam *) memory->srealloc(doubleparams,maxparam2*sizeof(Doubleparam), "pair:doubleparams");
  }

  MPI_Bcast(singleparams, maxparam1*sizeof(Singleparam), MPI_BYTE, 0, world);
  MPI_Bcast(doubleparams, maxparam2*sizeof(Doubleparam), MPI_BYTE, 0, world);
}

/* ---------------------------------------------------------------------- */

void PairDFTD3::setup_params()
{
  int i,j,k,m,n;

  // set elem3param for all element triplet combinations
  // must be a single exact match to lines read from file
  // do not allow for ACB in place of ABC

  memory->destroy(elem2singleparam);
  memory->create(elem2singleparam,nelements,"pair:elem2single");

  for (i = 0; i < nelements; i++) {
    n = -1;
    for (m = 0; m < nparams1; m++) {
      if (i == singleparams[m].ielement) {
        if (n >= 0) error->all(FLERR,"Potential file has a duplicate entry for: {}",
                                   elements[i]);
        n = m;
      }
    }
    if (n < 0) error->all(FLERR,"Potential file is missing an entry for: {}",
                              elements[i]);
    elem2singleparam[i] = n;
  }

  memory->destroy(elem2doubleparam);
  memory->create(elem2doubleparam,nelements,nelements,"pair:elem2paramdouble");

  for (i = 0; i < nelements; i++)
    for (j = 0; j < nelements; j++) {
      n = -1;
      for (m = 0; m < nparams2; m++) {
        if (i == doubleparams[m].ielement && j == doubleparams[m].jelement) {
          if (n >= 0) error->all(FLERR,"Potential file has a duplicate entry for: {} {}",
                                   elements[i],elements[j]);
          n = m;
        }
      }
      if (n < 0) error->all(FLERR,"Potential file is missing an entry for: {} {}",
                              elements[i],elements[j]);
      elem2doubleparam[i][j] = n;
  }


  // compute parameter values derived from inputs

  for (m = 0; m < nparams2; m++) {
    doubleparams[m].cocut = cut_coord;
    doubleparams[m].cut = cut_global;
    doubleparams[m].cocutsq = doubleparams[m].cocut*doubleparams[m].cocut;
    doubleparams[m].cutsq = doubleparams[m].cut*doubleparams[m].cut;
  }

  // set cutmax to max of all params

  cutmax = 0.0;
  for (m = 0; m < nparams2; m++)
    if (doubleparams[m].cut > cutmax) cutmax = doubleparams[m].cut;
}

/* ---------------------------------------------------------------------- */


void PairDFTD3::getc6ab(Doubleparam *doubleparam, double nci, double ncj, double &c6ab, double &dc6a, double &dc6b)
{
  double lij,r,k3rmax,k3r_tmp;
  double k3r[25];
  double lijsum = 0;
  double dlijdca = 0;
  double dlijdcb = 0;
  double dc6abdca = 0;
  double dc6abdcb = 0;
  c6ab = 0;
  k3rmax = -DBL_MAX;

  for (int i=0;i<25;i++){
    if(doubleparam->c6ab0[i]<0){
      k3r[i] = 0;
    }else{
      k3r_tmp = -d3_k3*(square(nci-doubleparam->c6ab1[i]) + square(ncj-doubleparam->c6ab2[i]));
      k3r[i] = k3r_tmp;
      if(k3rmax < k3r_tmp){
        k3rmax = k3r_tmp;
      }
    }
  }

  for (int i=0;i<25;i++){
    if(doubleparam->c6ab0[i]<0) continue;
    lij = exp(k3r[i]-k3rmax);
    dlijdca -= 2.0*d3_k3*lij*(nci-doubleparam->c6ab1[i]);
    dlijdcb -= 2.0*d3_k3*lij*(ncj-doubleparam->c6ab2[i]);
    lijsum += lij;
    c6ab += doubleparam->c6ab0[i]*lij;
    dc6abdca -= doubleparam->c6ab0[i]*2.0*d3_k3*lij*(nci-doubleparam->c6ab1[i]);
    dc6abdcb -= doubleparam->c6ab0[i]*2.0*d3_k3*lij*(ncj-doubleparam->c6ab2[i]);
  }
  
  c6ab /= lijsum;
  dc6a = (dc6abdca - c6ab * dlijdca) / lijsum;
  dc6b = (dc6abdcb - c6ab * dlijdcb) / lijsum;
}

/* ---------------------------------------------------------------------- */

int PairDFTD3::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++){
    buf[m++] = prefactor[i];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairDFTD3::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    prefactor[j] += buf[m++];
  }
}


/* ---------------------------------------------------------------------- */

int PairDFTD3::pack_forward_comm(int n, int *list, double *buf,
                               int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = cn[j];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairDFTD3::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++){
    cn[i] = buf[m++];
  }
}
