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
