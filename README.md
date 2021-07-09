# standardised_partial_product
Python code for the standardised partial product and other methods for combining *p*-values

## Partial product, complementary product and sum methods
There are individul files `PP.py`, `PCP.py` and `PS.py` to run the corresponding methods PP, PCP and PS from the command line.

Example usage: To combine five example p-values (comma or space separated),
```bash
echo 0.01 0.05 0.2 0.4 0.8 | ./PP.py
echo 0.03,0.05,0.2,0.4,0.8 | ./PCP.py
echo 0.03, 0.05, 0.2, 0.4, 0.8 | ./PS.py
```

## Comparing methods

Help:
```bash
./compare_combiners.py -h
```

gives a list of optional arguments.


Example usage:
```bash
jot -p 4 -r 5 0 1 | tr '\n' ' ' | ./compare_combiners.py -n0 100000
```

This generates 5 random values on [0,1], and then calculates the default *p*-value combination methods using null distributions derived from 100,000 samples for those methods requiring Monte Carlo simulation.

[Note that setting `-n0 0` corresponds to obtaining 0 Monte Carlo null distribution samples, and therefore implies that Monte Carlo samples should be read from a file. For a selection of values of `n`, files of stored samples can be obtained online from [here](http://null-distributions.ma.ic.ac.uk).]
