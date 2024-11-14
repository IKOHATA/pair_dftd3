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

```

## Description 

LAMMPS implementation of Grimme's D3 dispersion correction [1] with the Becke-Jones dumping function [2].

[1] S. Grimme, J. Antony, S. Ehrlich, and S. Krieg, J. Chem. Phys. 132, 154104 (2010).  
[2] S. Grimme, S. Ehrlich, and L. Goerigk, J. Comput. Chem. 32, 1456 (2011).
