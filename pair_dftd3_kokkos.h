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
PairStyle(dftd3/kk,PairDFTD3Kokkos<LMPDeviceType>);
PairStyle(dftd3/kk/device,PairDFTD3Kokkos<LMPDeviceType>);
PairStyle(dftd3/kk/host,PairDFTD3Kokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_DFTD3_KOKKOS_H
#define LMP_PAIR_DFTD3_KOKKOS_H

#include "pair_dftd3.h"
#include "pair_kokkos.h"
#include "neigh_list_kokkos.h"
#include "kokkos_base.h"

struct TagPairDFTD3PackForwardComm{};
struct TagPairDFTD3UnpackForwardComm{};
struct TagPairDFTD3ComputeShortNeigh{};
struct TagPairDFTD3Computec6ab{};
struct TagPairDFTD3Initialize{};

template<int NEIGHFLAG>
struct TagPairDFTD3ComputeCN{};

template<int NEIGHFLAG, int EVFLAG>
struct TagPairDFTD3ComputeForward{};

template<int NEIGHFLAG, int EVFLAG>
struct TagPairDFTD3ComputeBackward{};

namespace LAMMPS_NS {

template<class DeviceType>
class PairDFTD3Kokkos : public PairDFTD3, public KokkosBase  {
 public:
  enum {EnabledNeighFlags=HALF|HALFTHREAD};
  enum {COUL_FLAG=0};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairDFTD3Kokkos(class LAMMPS *);
  ~PairDFTD3Kokkos() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void init_style() override;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairDFTD3PackForwardComm, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairDFTD3UnpackForwardComm, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairDFTD3Initialize, const int&) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairDFTD3ComputeForward<NEIGHFLAG,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairDFTD3ComputeForward<NEIGHFLAG,EVFLAG>, const int&) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairDFTD3ComputeBackward<NEIGHFLAG,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairDFTD3ComputeBackward<NEIGHFLAG,EVFLAG>, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairDFTD3ComputeShortNeigh, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairDFTD3Computec6ab, const int&) const;
  
  template<int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairDFTD3ComputeCN<NEIGHFLAG>, const int&) const;

  template<int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION
  void ev_tally(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &epair, const F_FLOAT &fpair, const F_FLOAT &delx,
                  const F_FLOAT &dely, const F_FLOAT &delz) const;

  int pack_forward_comm_kokkos(int, DAT::tdual_int_2d, int, DAT::tdual_xfloat_1d&,
                       int, int *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_xfloat_1d&) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;


 protected:
  typedef Kokkos::DualView<int**,DeviceType> tdual_int_2d;
  typedef typename tdual_int_2d::t_dev_const_randomread t_int_2d_randomread;
  typedef typename tdual_int_2d::t_host t_host_int_2d;
  t_int_2d_randomread d_elem2doubleparam;

  typedef Kokkos::DualView<int*,DeviceType> tdual_int_1d;
  typedef typename tdual_int_1d::t_dev_const_randomread t_int_1d_randomread;
  typedef typename tdual_int_1d::t_host t_host_int_1d;
  t_int_1d_randomread d_elem2singleparam;

  typename AT::t_int_1d_randomread d_map;

  typedef Kokkos::DualView<Singleparam*,DeviceType> tdual_singleparam_1d;
  typedef typename tdual_singleparam_1d::t_dev t_singleparam_1d;
  typedef typename tdual_int_1d::t_dev_const_randomread t_int_singleparam_1d_randomread;
  typedef typename tdual_singleparam_1d::t_host t_host_singleparam_1d;

  //t_int_singleparam_1d_randomread d_elem2singleparam;
  t_singleparam_1d d_singleparams;

  typedef Kokkos::DualView<Doubleparam*,DeviceType> tdual_doubleparam_1d;
  typedef typename tdual_doubleparam_1d::t_dev t_doubleparam_1d;
  typedef typename tdual_int_1d::t_dev_const_randomread t_int_doubleparam_1d_randomread;
  typedef typename tdual_doubleparam_1d::t_host t_host_doubleparam_1d;

  //t_int_doubleparam_1d_randomread d_elem2doubleparam;
  t_doubleparam_1d d_doubleparams;

  void setup_params() override;

  KOKKOS_INLINE_FUNCTION
  void getc6ab(const Doubleparam&, const F_FLOAT &, const F_FLOAT &, double &, double &, double &) const;

  KOKKOS_INLINE_FUNCTION
  void getpref(const Singleparam&, const Singleparam&, const Doubleparam&, const F_FLOAT &, const F_FLOAT &, const F_FLOAT &, 
  double &, double &, double &, double &) const;

  //KOKKOS_INLINE_FUNCTION
  //void getprefactor(double &, double &, double &, double &, double &, double &, double &, double &, double &) const;

  //KOKKOS_INLINE_FUNCTION
  //void twobody(const Param&, const F_FLOAT&, F_FLOAT&, const int&, F_FLOAT&) const;

  //KOKKOS_INLINE_FUNCTION
  //void threebody_kk(const Param&, const Param&, const Param&, const F_FLOAT&, const F_FLOAT&, F_FLOAT *, F_FLOAT *,
                    //F_FLOAT *, F_FLOAT *, const int&, F_FLOAT&) const;

  typename AT::t_x_array_randomread x;
  typename AT::t_f_array f;
  typename AT::t_tagint_1d tag;
  typename AT::t_int_1d_randomread type;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  int need_dup;

  using KKDeviceType = typename KKDevice<DeviceType>::value;

  template<typename DataType, typename Layout>
  using DupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterDuplicated>;

  template<typename DataType, typename Layout>
  using NonDupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterNonDuplicated>;

  DupScatterView<F_FLOAT*, typename DAT::t_ffloat_1d::array_layout> dup_cn;
  DupScatterView<F_FLOAT*, typename DAT::t_ffloat_1d::array_layout> dup_prefactor;
  DupScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout> dup_f;
  DupScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout> dup_eatom;
  DupScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout> dup_vatom;

  NonDupScatterView<F_FLOAT*, typename DAT::t_ffloat_1d::array_layout> ndup_cn;
  NonDupScatterView<F_FLOAT*, typename DAT::t_ffloat_1d::array_layout> ndup_prefactor;
  NonDupScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout> ndup_f;
  NonDupScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout> ndup_eatom;
  NonDupScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout> ndup_vatom;

  DAT::tdual_efloat_1d k_cn;
  DAT::tdual_efloat_1d k_prefactor;
  typename AT::t_efloat_1d d_cn;
  typename AT::t_efloat_1d d_prefactor;
  HAT::t_efloat_1d h_cn;
  HAT::t_efloat_1d h_prefactor;

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;
  //NeighListKokkos<DeviceType> k_list;

  int iswap;
  int first;
  typename AT::t_int_2d d_sendlist;
  typename AT::t_xfloat_1d_um v_buf;

  int neighflag,newton_pair;
  int nlocal,nall,eflag,vflag;

  int inum, maxneigh, chunk_size, chunk_offset;
  int host_flag;
  Kokkos::View<int**,DeviceType> d_neighbors_short;
  Kokkos::View<int*,DeviceType> d_numneigh_short;

  Kokkos::View<double**,DeviceType> d_c6ab;
  Kokkos::View<double**,DeviceType> d_dc6a;
  Kokkos::View<double**,DeviceType> d_dc6b;


  friend void pair_virial_fdotr_compute<PairDFTD3Kokkos>(PairDFTD3Kokkos*);
};

}

#endif
#endif

