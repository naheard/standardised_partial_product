# Examples from "Standardized partial sums and products of *p*-values"

Figure 5: To obtain the results in the top left panel, for example, run the command

```bash
./compare_combiners.py -n 100 -n0 0 -n1 10000
```

Figure 6: To obtain the results in the top left panel, for example, run the command

```bash
./compare_combiners.py -n 10000 -n0 0 -n1 10000 -b 0.6
```

Figure 8: To obtain the results in the top right panel, for example, with the five methods used, run the command

```bash
./compare_combiners.py -n 10000 -n0 0 -n1 10000 -r 0.25 -h1 normal_hc -m higher_criticism standardised_product truncated_product fisher simes
```

Figure 8: To obtain the results used for the left hand panel, run the command

```bash
wget -qO- http://null-distributions.ma.ic.ac.uk/user_pvalues.txt | cut -f2 | ./compare_combiners.py -n0 10000 -m standardised_product2 standardised_product modified_berk_jones truncated_product fisher higher_criticism2 standardised_sum standardised_complementary_product higher_criticism simes
```

[Note that in the article, `-n0 100000000` was used to give smoother and more accurate power curves. The output is a 1051 x 10 matrix of p-values, where each row corresponds to a computer network user and each column corresponds to a p-value combination method, identified by the header in the first row.]
