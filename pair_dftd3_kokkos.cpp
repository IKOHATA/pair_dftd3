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

#include "pair_dftd3_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "math_const.h"
#include "math_special_kokkos.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "neighbor_kokkos.h"
#include "pair_kokkos.h"

#include <cmath>
#include <iostream>
#include <float.h>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecialKokkos;

#define MAXLINE 1024
#define DELTA 4

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairDFTD3Kokkos<DeviceType>::PairDFTD3Kokkos(LAMMPS *lmp) : PairDFTD3(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  //datamask_read = X_MASK | F_MASK | TAG_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  //datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;

  host_flag = (execution_space == Host);
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

template<class DeviceType>
PairDFTD3Kokkos<DeviceType>::~PairDFTD3Kokkos()
{
  if (!copymode) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
    eatom = nullptr;
    vatom = nullptr;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairDFTD3Kokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  atomKK->sync(execution_space,datamask_read);
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);

  if (atom->nmax > nmax) {
    nmax = atom->nmax;
    k_cn = DAT::tdual_efloat_1d("pair:cn",nmax);
    k_prefactor = DAT::tdual_efloat_1d("pair:prefactor",nmax);
    d_cn = k_cn.template view<DeviceType>();
    d_prefactor = k_prefactor.template view<DeviceType>();
    h_cn = k_cn.h_view;
    h_prefactor = k_prefactor.h_view;
  }

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  nlocal = atom->nlocal;
  newton_pair = force->newton_pair;
  nall = atom->nlocal + atom->nghost;

  const int inum = list->inum;
  const int ignum = inum + list->gnum;
  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_ilist = k_list->d_ilist;
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;

  need_dup = lmp->kokkos->need_dup<DeviceType>();
  if (need_dup) {
    dup_cn   = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_cn);
    dup_prefactor   = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_prefactor);
    dup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(f);
    dup_eatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_eatom);
    dup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_vatom);
  } else {
    ndup_cn     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_cn);
    ndup_prefactor     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_prefactor);
    ndup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(f);
    ndup_eatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_eatom);
    ndup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_vatom);
  }

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDFTD3Initialize>(0,nall),*this);


  // build short neighbor list

  int max_neighs = d_neighbors.extent(1);

  if (((int) d_neighbors_short.extent(1) < max_neighs) ||
      ((int) d_neighbors_short.extent(0) < ignum)) {
    d_neighbors_short = Kokkos::View<int**,DeviceType>("DFTD3::neighbors_short",ignum*1.1,max_neighs);
    //d_c6ab = Kokkos::View<double**,DeviceType>("DFTD3::c6ab",ignum*1.1,max_neighs);
    //d_dc6a = Kokkos::View<double**,DeviceType>("DFTD3::dc6a",ignum*1.1,max_neighs);
    //d_dc6b = Kokkos::View<double**,DeviceType>("DFTD3::dc6b",ignum*1.1,max_neighs);
  }
  if ((int)d_numneigh_short.extent(0) < ignum)
    d_numneigh_short = Kokkos::View<int*,DeviceType>("DFTD3::numneighs_short",ignum*1.1);

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,TagPairDFTD3ComputeShortNeigh>(0,inum), *this);
  
  /*
  if (neighflag == HALF){
    std::cout << 1 << '\n';
  }else if (neighflag == HALFTHREAD){
    std::cout << 2 << '\n';
  }

  if (need_dup){
    std::cout << 3 << '\n';
  } else{
    std::cout << 4 << '\n';
  }
  */
  
  if (neighflag == HALF) {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDFTD3ComputeCN<HALF>>(0,inum),*this);
  } else if (neighflag == HALFTHREAD) {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDFTD3ComputeCN<HALFTHREAD>>(0,inum),*this);
  }

  if (need_dup)
    Kokkos::Experimental::contribute(d_cn, dup_cn);

  // communicate and sum densities (on the host)

  k_cn.template modify<DeviceType>();
  comm->forward_comm(this);
  k_cn.template sync<DeviceType>();


  int chunksize = inum;
  chunk_size = MIN(chunksize,inum); // "chunksize" variable is set by user
  chunk_offset = 0;

  EV_FLOAT ev;
  EV_FLOAT ev_all;

  while (chunk_offset < inum) {

  if (chunk_size > inum - chunk_offset)
    chunk_size = inum - chunk_offset;

  //Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType,TagPairDFTD3Computec6ab>(0,chunk_size), *this);

  if (neighflag == HALF) {
    if (evflag)
      Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairDFTD3ComputeForward<HALF,1>>(0,chunk_size),*this,ev);
    else
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDFTD3ComputeForward<HALF,0>>(0,chunk_size),*this);
  } else if (neighflag == HALFTHREAD) {
    if (evflag)
      Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairDFTD3ComputeForward<HALFTHREAD,1>>(0,chunk_size),*this,ev);
    else
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDFTD3ComputeForward<HALFTHREAD,0>>(0,chunk_size),*this);
  }

  if (eflag) {
    eng_vdwl += ev.evdwl;
    ev.evdwl = 0.0;
  }
    ev_all += ev;
    chunk_offset += chunk_size;
  }
  

  if (need_dup)
    Kokkos::Experimental::contribute(d_prefactor, dup_prefactor);

  k_prefactor.template modify<DeviceType>();
  comm->reverse_comm(this);
  k_prefactor.template sync<DeviceType>();


  chunk_size = MIN(chunksize,inum); // "chunksize" variable is set by user
  chunk_offset = 0;

  while (chunk_offset < inum) {

    if (chunk_size > inum - chunk_offset)
      chunk_size = inum - chunk_offset;

  if (neighflag == HALF) {
    if (evflag)
      Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairDFTD3ComputeBackward<HALF,1> >(0,chunk_size),*this,ev);
    else
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDFTD3ComputeBackward<HALF,0> >(0,chunk_size),*this);
  } else if (neighflag == HALFTHREAD) {
    if (evflag)
      Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairDFTD3ComputeBackward<HALFTHREAD,1> >(0,chunk_size),*this,ev);
    else
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDFTD3ComputeBackward<HALFTHREAD,0> >(0,chunk_size),*this);
  }
    ev_all += ev;
    chunk_offset += chunk_size;
 }
  
  

  if (need_dup)
    Kokkos::Experimental::contribute(f, dup_f);

  //if (eflag_global) eng_vdwl += ev.evdwl;
  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  if (eflag_atom) {
    if (need_dup)
      Kokkos::Experimental::contribute(d_eatom, dup_eatom);
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    if (need_dup)
      Kokkos::Experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  copymode = 0;

  // free duplicated memory
  
  if (need_dup) {
    dup_cn   = decltype(dup_cn)();
    dup_prefactor   = decltype(dup_prefactor)();
    dup_f            = decltype(dup_f)();
    dup_eatom        = decltype(dup_eatom)();
    dup_vatom        = decltype(dup_vatom)();
  }
  
}


/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDFTD3Kokkos<DeviceType>::operator()(TagPairDFTD3ComputeShortNeigh, const int& ii) const {
    const int i = d_ilist[ii];
    const int itype = d_map[type[i]];
    const X_FLOAT xtmp = x(i,0);
    const X_FLOAT ytmp = x(i,1);
    const X_FLOAT ztmp = x(i,2);

    const int jnum = d_numneigh[i];
    int inside = 0;
    for (int jj = 0; jj < jnum; jj++) {
      int j = d_neighbors(i,jj);
      j &= NEIGHMASK;
      const int jtype = d_map[type[j]];

      const X_FLOAT delx = xtmp - x(j,0);
      const X_FLOAT dely = ytmp - x(j,1);
      const X_FLOAT delz = ztmp - x(j,2);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      const int ijparam = d_elem2doubleparam(itype,jtype);
      if (rsq < d_doubleparams[ijparam].cutsq) {
        d_neighbors_short(ii,inside) = j;
        inside++;
      }
    }
    d_numneigh_short(ii) = inside;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDFTD3Kokkos<DeviceType>::operator()(TagPairDFTD3Computec6ab, const int& ii) const {

    const int ii_shift = ii + chunk_offset;
    const int i = d_ilist[ii_shift];
    //const int i = d_ilist[ii];
    const int itype = d_map[type[i]];
    const X_FLOAT xtmp = x(i,0);
    const X_FLOAT ytmp = x(i,1);
    const X_FLOAT ztmp = x(i,2);
    F_FLOAT c6ab,dc6a,dc6b;

    const int jnum = d_numneigh_short[ii_shift];

    F_FLOAT nci = d_cn[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = d_neighbors_short(ii_shift,jj);

      const int jtype = d_map[type[j]];

      F_FLOAT ncj = d_cn[j];

      const int iparam = d_elem2singleparam(itype);
      const int jparam = d_elem2singleparam(jtype);

      const X_FLOAT delx = xtmp - x(j,0);
      const X_FLOAT dely = ytmp - x(j,1);
      const X_FLOAT delz = ztmp - x(j,2);

      const int ijparam = d_elem2doubleparam(itype,jtype);

      getc6ab(d_doubleparams(ijparam),nci,ncj,c6ab,dc6a,dc6b);

      d_c6ab(ii_shift,jj) = c6ab;
      d_dc6a(ii_shift,jj) = dc6a;
      d_dc6b(ii_shift,jj) = dc6b;
    }
}

/* ---------------------------------------------------------------------- */

////Specialisation for Neighborlist types Half, HalfThread
template<class DeviceType>
template<int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION
void PairDFTD3Kokkos<DeviceType>::operator()(TagPairDFTD3ComputeCN<NEIGHFLAG>, const int &ii) const {

  // cn = coornination number of each atom
  // loop over neighbors of my atoms

  // The rho array is duplicated for OpenMP, atomic for CUDA, and neither for Serial

  auto v_cn = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_cn),decltype(ndup_cn)>::get(dup_cn,ndup_cn);
  auto a_cn = v_cn.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();
  //Kokkos::View<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > a_cn = d_cn;

  const int i = d_ilist[ii];
  const X_FLOAT xtmp = x(i,0);
  const X_FLOAT ytmp = x(i,1);
  const X_FLOAT ztmp = x(i,2);
  const int itype = d_map(type(i));

  const int jnum = d_numneigh_short[ii];

  F_FLOAT cntmp = 0.0;
  F_FLOAT rco, expco, cnab;

  for (int jj = 0; jj < jnum; jj++) {
    int j = d_neighbors_short(ii,jj);
    const X_FLOAT delx = xtmp - x(j,0);
    const X_FLOAT dely = ytmp - x(j,1);
    const X_FLOAT delz = ztmp - x(j,2);
    const int jtype = d_map(type(j));
    const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

    const int iparam_ij = d_elem2doubleparam(itype,jtype);
    const F_FLOAT cutsq1 = d_doubleparams(iparam_ij).cocutsq;

    if (rsq < cutsq1) {
      F_FLOAT r1 = sqrt(rsq);

      const int iparam_i = d_elem2singleparam(itype);
      const int iparam_j = d_elem2singleparam(jtype);

      rco = d_singleparams(iparam_i).rcov + d_singleparams(iparam_j).rcov;

      expco = exp(-d3_k1*(rco/r1-1.0));
      cnab = 1.0/(1.0+expco);

      cntmp += cnab;
      //a_cn[j] += cnab;
    }
  }
  a_cn[i] += cntmp;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairDFTD3Kokkos<DeviceType>::operator()(TagPairDFTD3ComputeForward<NEIGHFLAG,EVFLAG>, const int &ii, EV_FLOAT& ev) const {

  // The f array is duplicated for OpenMP, atomic for CUDA, and neither for Serial

  auto v_f = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();
  auto v_prefactor = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_prefactor),decltype(ndup_prefactor)>::get(dup_prefactor,ndup_prefactor);
  auto a_prefactor = v_prefactor.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  F_FLOAT c6ab,dc6a,dc6b,kp;
  F_FLOAT evdwl,fpair,prefactora,prefactorb;
  F_FLOAT r1,r6,c8,tmp,tmp2,tmp6;
  F_FLOAT e6,f6ab,e8,f8ab,f6ab1,f8ab1,e61,e81;

  //const int i = d_ilist[ii];
  const int ii_shift = ii + chunk_offset;
  const int i = d_ilist[ii_shift];
  const int itype = d_map[type[i]];
  const X_FLOAT xtmp = x(i,0);
  const X_FLOAT ytmp = x(i,1);
  const X_FLOAT ztmp = x(i,2);

  // two-body interactions, skip half of them

  const int jnum = d_numneigh_short[ii_shift];
  //const int jnum = d_numneigh[i];

  F_FLOAT nci = d_cn[i];

  F_FLOAT fxtmpi = 0.0;
  F_FLOAT fytmpi = 0.0;
  F_FLOAT fztmpi = 0.0;

  for (int jj = 0; jj < jnum; jj++) {
    int j = d_neighbors_short(ii_shift,jj);
    //int j = d_neighbors(i,jj);
    //j &= NEIGHMASK;

    const int jtype = d_map[type[j]];

    F_FLOAT ncj = d_cn[j];

    const int iparam = d_elem2singleparam(itype);
    const int jparam = d_elem2singleparam(jtype);

    const X_FLOAT delx = xtmp - x(j,0);
    const X_FLOAT dely = ytmp - x(j,1);
    const X_FLOAT delz = ztmp - x(j,2);
    F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

    const int ijparam = d_elem2doubleparam(itype,jtype);

    getc6ab(d_doubleparams(ijparam),nci,ncj,c6ab,dc6a,dc6b);

    //getpref(d_singleparams(iparam),d_singleparams(jparam),d_doubleparams(ijparam),nci,ncj,rsq,evdwl,fpair,prefactora,prefactorb);

    //c6ab = d_c6ab(ii_shift,jj);
    //dc6a = d_dc6a(ii_shift,jj);
    //dc6b = d_dc6b(ii_shift,jj);

    r1 = sqrt(rsq);
    r6 = cube(rsq);

    kp = 3.0 * d_singleparams[iparam].r2r4 * d_singleparams[jparam].r2r4;
    
    tmp = rs6 * sqrt(kp) + rs18;
    tmp2 = square(tmp);
    tmp6 = cube(tmp2);
    e6 = -1.0 / (r6 + tmp6);
    f6ab = -6.0 * square(e6) * (r6/r1);
    
    e8 = -1.0 / (r6*rsq + tmp6*tmp2);
    f8ab = -8.0 * square(e8) * (r6*r1);
    
    e61 = 0.5 * s6 * e6;
    e81 = 0.5 * s18 * e8;
    c8 = kp * c6ab;
    evdwl = c6ab * e61 + c8 * e81;
    
    f6ab1 = c6ab * f6ab * s6 * 0.5;
    f8ab1 = c8 * f8ab * s18 * 0.5;
    
    fpair = (f6ab1+f8ab1)/r1;
    prefactora = dc6a*(e61+kp*e81);
    prefactorb = dc6b*(e61+kp*e81);

    a_prefactor[i] += prefactora;
    a_prefactor[j] += prefactorb;

    fxtmpi += delx*fpair;
    fytmpi += dely*fpair;
    fztmpi += delz*fpair;
    a_f(j,0) -= delx*fpair;
    a_f(j,1) -= dely*fpair;
    a_f(j,2) -= delz*fpair;

    if (EVFLAG) {
      if (eflag) ev.evdwl += evdwl;
      if (vflag_either || eflag_atom) this->template ev_tally<NEIGHFLAG>(ev,i,j,evdwl,fpair,delx,dely,delz);
    }
  }
  a_f(i,0) += fxtmpi;
  a_f(i,1) += fytmpi;
  a_f(i,2) += fztmpi;
}

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairDFTD3Kokkos<DeviceType>::operator()(TagPairDFTD3ComputeForward<NEIGHFLAG,EVFLAG>, const int &ii) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,EVFLAG>(TagPairDFTD3ComputeForward<NEIGHFLAG,EVFLAG>(), ii, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairDFTD3Kokkos<DeviceType>::operator()(TagPairDFTD3ComputeBackward<NEIGHFLAG,EVFLAG>, const int &ii, EV_FLOAT& ev) const {

  // The f array is duplicated for OpenMP, atomic for CUDA, and neither for Serial

  //Kokkos::View<F_FLOAT*[3], typename DAT::t_f_array::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > a_f = f;
  auto v_f = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  const int ii_shift = ii + chunk_offset;
  const int i = d_ilist[ii_shift];
  //const int i = d_ilist[ii];
  const tagint itag = tag[i];
  const int itype = d_map[type[i]];
  const X_FLOAT xtmp = x(i,0);
  const X_FLOAT ytmp = x(i,1);
  const X_FLOAT ztmp = x(i,2);

  // two-body interactions, skip half of them

  const int jnum = d_numneigh_short[ii_shift];
  //const int jnum = d_numneigh[i];

  F_FLOAT fxtmpi = 0.0;
  F_FLOAT fytmpi = 0.0;
  F_FLOAT fztmpi = 0.0;


  F_FLOAT pf, r1, rco, expco, cnab, ddamp, fpair;
  pf = d_prefactor[i];

  const int iparam_i = d_elem2singleparam(itype);

  

  for (int jj = 0; jj < jnum; jj++) {
    int j = d_neighbors_short(ii_shift,jj);
    //int j = d_neighbors(i,jj);
    //j &= NEIGHMASK;

    const int jtype = d_map[type[j]];

    const X_FLOAT delx = x(j,0) - xtmp;
    const X_FLOAT dely = x(j,1) - ytmp;
    const X_FLOAT delz = x(j,2) - ztmp;
    const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

    const int ijparam = d_elem2doubleparam(itype,jtype);

    if (rsq < d_doubleparams(ijparam).cocutsq){
  
      r1 = sqrt(rsq);

      const int iparam_j = d_elem2singleparam(jtype);

      rco = d_singleparams[iparam_i].rcov+d_singleparams[iparam_j].rcov;

      expco = exp(-d3_k1*(rco/r1-1.0));
      cnab = 1.0/(1.0+expco);

      ddamp = -expco*d3_k1*rco*square(cnab)/rsq;
      fpair = ddamp*pf/r1;

      fxtmpi += delx*fpair;
      fytmpi += dely*fpair;
      fztmpi += delz*fpair;
      a_f(j,0) -= delx*fpair;
      a_f(j,1) -= dely*fpair;
      a_f(j,2) -= delz*fpair;

      if (EVFLAG) {
        if (vflag_either || eflag_atom) this->template ev_tally<NEIGHFLAG>(ev,i,j,0.0,fpair,delx,dely,delz);
      }
    }
  }
  a_f(i,0) += fxtmpi;
  a_f(i,1) += fytmpi;
  a_f(i,2) += fztmpi;
}

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairDFTD3Kokkos<DeviceType>::operator()(TagPairDFTD3ComputeBackward<NEIGHFLAG,EVFLAG>, const int &ii) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,EVFLAG>(TagPairDFTD3ComputeBackward<NEIGHFLAG,EVFLAG>(), ii, ev);
}


/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template<class DeviceType>
void PairDFTD3Kokkos<DeviceType>::coeff(int narg, char **arg)
{
  PairDFTD3::coeff(narg,arg);

  // sync map

  int n = atom->ntypes;

  DAT::tdual_int_1d k_map = DAT::tdual_int_1d("pair:map",n+1);
  HAT::t_int_1d h_map = k_map.h_view;

  for (int i = 1; i <= n; i++)
    h_map[i] = map[i];

  k_map.template modify<LMPHostType>();
  k_map.template sync<DeviceType>();

  d_map = k_map.template view<DeviceType>();
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairDFTD3Kokkos<DeviceType>::init_style()
{
  PairDFTD3::init_style();

  // adjust neighbor list request for KOKKOS

  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
                           !std::is_same<DeviceType,LMPDeviceType>::value);
  request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);

  if (neighflag == FULL)
    error->all(FLERR,"Must use half neighbor list style with pair dftd3/kk");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairDFTD3Kokkos<DeviceType>::setup_params()
{
  PairDFTD3::setup_params();

  // sync elem2singleparam and elem2doubleparam

  tdual_int_1d k_elem2singleparam = tdual_int_1d("pair:elem2singleparam",nelements);
  t_host_int_1d h_elem2singleparam = k_elem2singleparam.h_view;

  tdual_int_2d k_elem2doubleparam = tdual_int_2d("pair:elem2doubleparam",nelements,nelements);
  t_host_int_2d h_elem2doubleparam = k_elem2doubleparam.h_view;

  // sync singleparam and doubleparam
  
  tdual_singleparam_1d k_singleparams = tdual_singleparam_1d("pair:singleparams",nparams1);
  t_host_singleparam_1d h_singleparams = k_singleparams.h_view;

  tdual_doubleparam_1d k_doubleparams = tdual_doubleparam_1d("pair:doubleparams",nparams2);
  t_host_doubleparam_1d h_doubleparams = k_doubleparams.h_view;

  for (int i = 0; i < nelements; i++)
        h_elem2singleparam(i) = elem2singleparam[i];
  
  for (int i = 0; i < nelements; i++)
    for (int j = 0; j < nelements; j++)
        h_elem2doubleparam(i,j) = elem2doubleparam[i][j];

  for (int m = 0; m < nparams1; m++)
    h_singleparams[m] = singleparams[m];

  for (int m = 0; m < nparams2; m++)
    h_doubleparams[m] = doubleparams[m];

  k_elem2singleparam.modify_host();
  k_elem2singleparam.template sync<DeviceType>();
  k_elem2doubleparam.modify_host();
  k_elem2doubleparam.template sync<DeviceType>();
  k_singleparams.modify_host();
  k_singleparams.template sync<DeviceType>();
  k_doubleparams.modify_host();
  k_doubleparams.template sync<DeviceType>();


  d_elem2singleparam = k_elem2singleparam.template view<DeviceType>();
  d_elem2doubleparam = k_elem2doubleparam.template view<DeviceType>();
  d_singleparams = k_singleparams.template view<DeviceType>();
  d_doubleparams = k_doubleparams.template view<DeviceType>();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDFTD3Kokkos<DeviceType>::operator()(TagPairDFTD3Initialize, const int &i) const {
  d_cn[i] = 0.0;
  d_prefactor[i] = 0.0;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDFTD3Kokkos<DeviceType>::getc6ab(const Doubleparam& doubleparam, const F_FLOAT &nci, const F_FLOAT &ncj, double &c6ab, double &dc6a, double &dc6b) const
{
  F_FLOAT lij,k3r_tmp;
  F_FLOAT k3r[25];
  F_FLOAT lijsum = 0;
  F_FLOAT dlijdca = 0;
  F_FLOAT dlijdcb = 0;
  F_FLOAT dc6abdca = 0;
  F_FLOAT dc6abdcb = 0;
  c6ab = 0;
  F_FLOAT k3rmax = -DBL_MAX;

  for (int i=0;i<25;i++){
    if(doubleparam.c6ab0[i]<0){
      k3r[i] = 0;
    }else{
      k3r_tmp = -d3_k3*(square(nci-doubleparam.c6ab1[i]) + square(ncj-doubleparam.c6ab2[i]));
      k3r[i] = k3r_tmp;
      if(k3rmax < k3r_tmp){
        k3rmax = k3r_tmp;
      }
    }
  }

  for (int i=0;i<25;i++){
    if(doubleparam.c6ab0[i]<0) continue;
    lij = exp(k3r[i]-k3rmax);
    dlijdca -= 2.0*d3_k3*lij*(nci-doubleparam.c6ab1[i]);
    dlijdcb -= 2.0*d3_k3*lij*(ncj-doubleparam.c6ab2[i]);
    lijsum += lij;
    c6ab += doubleparam.c6ab0[i]*lij;
    dc6abdca -= doubleparam.c6ab0[i]*2.0*d3_k3*lij*(nci-doubleparam.c6ab1[i]);
    dc6abdcb -= doubleparam.c6ab0[i]*2.0*d3_k3*lij*(ncj-doubleparam.c6ab2[i]);
  }
  
  c6ab /= lijsum;
  dc6a = (dc6abdca - c6ab * dlijdca) / lijsum;
  dc6b = (dc6abdcb - c6ab * dlijdcb) / lijsum;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDFTD3Kokkos<DeviceType>::getpref(const Singleparam& singleparami, const Singleparam& singleparamj, const Doubleparam& doubleparamij, const F_FLOAT &nci, const F_FLOAT &ncj, const F_FLOAT &rsq, double &evdwl, double &fpair, double &prefactora, double &prefactorb) const
{
  F_FLOAT r1,r6,r5,r8,r7,kp,c8,tmp,tmp2,tmp6,tmp8;
  F_FLOAT e6,f6ab,e8,f8ab,f6ab1,f8ab1;
  F_FLOAT c6ab,dc6a,dc6b;
  F_FLOAT lij,k3r_tmp;
  F_FLOAT k3r[25];
  F_FLOAT lijsum = 0;
  F_FLOAT dlijdca = 0;
  F_FLOAT dlijdcb = 0;
  F_FLOAT dc6abdca = 0;
  F_FLOAT dc6abdcb = 0;
  c6ab = 0;
  F_FLOAT k3rmax = -DBL_MAX;

  for (int i=0;i<25;i++){
    if(doubleparamij.c6ab0[i]<0){
      k3r[i] = 0;
    }else{
      k3r_tmp = -d3_k3*(square(nci-doubleparamij.c6ab1[i]) + square(ncj-doubleparamij.c6ab2[i]));
      k3r[i] = k3r_tmp;
      if(k3rmax < k3r_tmp){
        k3rmax = k3r_tmp;
      }
    }
  }

  for (int i=0;i<25;i++){
    if(doubleparamij.c6ab0[i]<0) continue;
    lij = exp(k3r[i]-k3rmax);
    dlijdca -= 2.0*d3_k3*lij*(nci-doubleparamij.c6ab1[i]);
    dlijdcb -= 2.0*d3_k3*lij*(ncj-doubleparamij.c6ab2[i]);
    lijsum += lij;
    c6ab += doubleparamij.c6ab0[i]*lij;
    dc6abdca -= doubleparamij.c6ab0[i]*2.0*d3_k3*lij*(nci-doubleparamij.c6ab1[i]);
    dc6abdcb -= doubleparamij.c6ab0[i]*2.0*d3_k3*lij*(ncj-doubleparamij.c6ab2[i]);
  }
  
  c6ab /= lijsum;
  dc6a = (dc6abdca - c6ab * dlijdca) / lijsum;
  dc6b = (dc6abdcb - c6ab * dlijdcb) / lijsum;

  kp = 3.0 * singleparami.r2r4 * singleparamj.r2r4;

  r1 = sqrt(rsq);
  r6 = cube(rsq);
  r5 = r6 / r1;
  r8 = r6 * rsq;
  r7 = r8 / r1;
    
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

  c8 = kp * c6ab;

  f6ab1 = c6ab * f6ab * s6 * 0.5;
  f8ab1 = c8 * f8ab * s18 * 0.5;

  evdwl = c6ab * e6 + c8 * e8;

  fpair = (f6ab1+f8ab1)/r1;

  prefactora = dc6a*(e6+kp*e8);
  prefactorb = dc6b*(e6+kp*e8);

}

template<class DeviceType>
int PairDFTD3Kokkos<DeviceType>::pack_forward_comm_kokkos(int n, DAT::tdual_int_2d k_sendlist,
                                                        int iswap_in, DAT::tdual_xfloat_1d &buf,
                                                        int /*pbc_flag*/, int * /*pbc*/)
{
  d_sendlist = k_sendlist.view<DeviceType>();
  iswap = iswap_in;
  v_buf = buf.view<DeviceType>();
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDFTD3PackForwardComm>(0,n),*this);
  return n;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDFTD3Kokkos<DeviceType>::operator()(TagPairDFTD3PackForwardComm, const int &i) const {
  int j = d_sendlist(iswap, i);
  v_buf[i] = d_cn[j];
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairDFTD3Kokkos<DeviceType>::unpack_forward_comm_kokkos(int n, int first_in, DAT::tdual_xfloat_1d &buf)
{
  first = first_in;
  v_buf = buf.view<DeviceType>();
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDFTD3UnpackForwardComm>(0,n),*this);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDFTD3Kokkos<DeviceType>::operator()(TagPairDFTD3UnpackForwardComm, const int &i) const {
  d_cn[i + first] = v_buf[i];
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int PairDFTD3Kokkos<DeviceType>::pack_forward_comm(int n, int *list, double *buf,
                                                 int /*pbc_flag*/, int * /*pbc*/)
{
  k_cn.sync_host();

  int i,j;

  for (i = 0; i < n; i++) {
    j = list[i];
    buf[i] = h_cn[j];
  }
  return n;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairDFTD3Kokkos<DeviceType>::unpack_forward_comm(int n, int first, double *buf)
{
  k_cn.sync_host();

  for (int i = 0; i < n; i++) {
    h_cn[i + first] = buf[i];
  }

  k_cn.modify_host();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int PairDFTD3Kokkos<DeviceType>::pack_reverse_comm(int n, int first, double *buf)
{
  k_prefactor.sync_host();

  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) buf[m++] = h_prefactor[i];
  return m;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairDFTD3Kokkos<DeviceType>::unpack_reverse_comm(int n, int *list, double *buf)
{
  k_prefactor.sync_host();

  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    h_prefactor[j] += buf[m++];
  }

  k_prefactor.modify_host();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION
void PairDFTD3Kokkos<DeviceType>::ev_tally(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &epair, const F_FLOAT &fpair, const F_FLOAT &delx,
                const F_FLOAT &dely, const F_FLOAT &delz) const
{

  // The eatom and vatom arrays are duplicated for OpenMP, atomic for CUDA, and neither for Serial

  auto v_eatom = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_eatom),decltype(ndup_eatom)>::get(dup_eatom,ndup_eatom);
  auto a_eatom = v_eatom.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  auto v_vatom = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_vatom),decltype(ndup_vatom)>::get(dup_vatom,ndup_vatom);
  auto a_vatom = v_vatom.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  //Kokkos::View<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > v_eatom = k_eatom.view<DeviceType>();
  //Kokkos::View<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,typename KKDevice<DeviceType>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > v_vatom = k_vatom.view<DeviceType>();

  if (eflag_atom) {
    const E_FLOAT epairhalf = 0.5 * epair;
    a_eatom[i] += epairhalf;
    a_eatom[j] += epairhalf;
  }

  if (vflag_either) {
    const E_FLOAT v0 = delx*delx*fpair;
    const E_FLOAT v1 = dely*dely*fpair;
    const E_FLOAT v2 = delz*delz*fpair;
    const E_FLOAT v3 = delx*dely*fpair;
    const E_FLOAT v4 = delx*delz*fpair;
    const E_FLOAT v5 = dely*delz*fpair;

    if (vflag_global) {
      ev.v[0] += v0;
      ev.v[1] += v1;
      ev.v[2] += v2;
      ev.v[3] += v3;
      ev.v[4] += v4;
      ev.v[5] += v5;
    }

    if (vflag_atom) {
      a_vatom(i,0) += 0.5*v0;
      a_vatom(i,1) += 0.5*v1;
      a_vatom(i,2) += 0.5*v2;
      a_vatom(i,3) += 0.5*v3;
      a_vatom(i,4) += 0.5*v4;
      a_vatom(i,5) += 0.5*v5;

      a_vatom(j,0) += 0.5*v0;
      a_vatom(j,1) += 0.5*v1;
      a_vatom(j,2) += 0.5*v2;
      a_vatom(j,3) += 0.5*v3;
      a_vatom(j,4) += 0.5*v4;
      a_vatom(j,5) += 0.5*v5;
    }
  }
}
/*
template<class DeviceType>
struct FindMaxNumNeighs {
  typedef DeviceType device_type;
  NeighListKokkos<DeviceType> k_list;

  FindMaxNumNeighs(NeighListKokkos<DeviceType>* nl): k_list(*nl) {}
  ~FindMaxNumNeighs() {k_list.copymode = 1;}

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& ii, int& maxneigh) const {
    const int i = k_list.d_ilist[ii];
    const int num_neighs = k_list.d_numneigh[i];
    if (maxneigh < num_neighs) maxneigh = num_neighs;
  }
};
*/

namespace LAMMPS_NS {
template class PairDFTD3Kokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairDFTD3Kokkos<LMPHostType>;
#endif
}

