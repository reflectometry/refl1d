
set timestamp
set bars
set title 'Current best fit'
set xlabel 'Q (inverse Angstroms)'
set ylabel 'Reflectivity'
set logscale y




plot 'fit0.dat' u 1:3:2:4 t 'Model 0' w xyerrorbars, 'fit0.dat' u 1:5 not w lines

