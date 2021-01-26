# standardised_partial_product
python code for the standardised partial product and other methods for combining p-values

Help:
./compare_combiners.py -h

gives a list of optional arguments.


Example usage:
jot -p 4 -r 5 0 1 | tr '\n' ' ' | ./compare_combiners.py -n0 100000

This generates 5 random values on [0,1], and then calculates the default p-value combination methods using null distributions derived from 100,000 samples for those methods requiring Monte Carlo simulation.

[Note that setting `-n0 0` corresponds to obtaining 0 Monte Carlo null distribution samples, and therefore correpsonds to reading in Monte Carlo samples from a file. For a selection of values of `n`, files of stored samples can be obtained online from [here](http://null-distributions.ma.ic.ac.uk).]
