# pair_style dftd3 command
Accelerator Variants: _dftd3/kk_
## Syntax
```
pair_style style xc cutoff cnthr
```
- style = _dftd3_
- xc = _pbe_ or _b3-lyp_
- cutoff = Cutoff radius for two-body dispersion calculations
- cnthr = Cutoff radius for coordination number and three-body calculations

## Example
```
pair_style hybrid/overlay pace dftd3 pbe 12.0 8.0  
pair_coeff * * pace Cu.yaml Cu 
pair_coeff * * dftd3 param.dftd3 Cu
```

## Description 

The _dftd3_ pair style computes the Grimme's D3 dispersion correction [1] with the Becke-Jones dumping function [2].

## Note

This implementation was tested in LAMMPS 2 Aug 2023 version.  
It may not work in some environments/versions.

[1] S. Grimme, J. Antony, S. Ehrlich, and S. Krieg, J. Chem. Phys. 132, 154104 (2010).  
[2] S. Grimme, S. Ehrlich, and L. Goerigk, J. Comput. Chem. 32, 1456 (2011).
