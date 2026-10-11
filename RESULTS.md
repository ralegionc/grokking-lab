# Results

Written by `scripts/report.py` from the summary of every run under `runs/`.
22 runs, 5.9 hours of CPU time in total.

`memorised` is the first evaluation at which training accuracy reached 99%.
`generalised` is the first at which held out accuracy reached 95%.

| run | task | model | train fraction | weight decay | memorised | generalised | gap | final held out | key frequencies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add_p59 | add mod 59 | mlp | 0.5 | 2 | 100 | 860 | 760 steps, 9x | 1.000 | 3, 5, 14, 15 |
| div_p59 | div mod 59 | mlp | 0.5 | 2 | 100 | 1,200 | 1,100 steps, 12x | 1.000 | none |
| headline | add mod 97 | transformer | 0.3 | 1 | 125 | 5,125 | 5,000 steps, 41x | 1.000 | 19, 29, 31, 36 |
| quick_mlp | add mod 97 | mlp | 0.4 | 2 | 140 | 1,300 | 1,160 steps, 9x | 1.000 | 14, 33, 36, 38, 44, 48 |
| seed_0 | add mod 59 | mlp | 0.5 | 2 | 100 | 860 | 760 steps, 9x | 1.000 | 3, 5, 14, 15 |
| seed_1 | add mod 59 | mlp | 0.5 | 2 | 100 | 980 | 880 steps, 10x | 1.000 | 4, 5, 12, 14, 25 |
| seed_2 | add mod 59 | mlp | 0.5 | 2 | 100 | 1,060 | 960 steps, 11x | 1.000 | 5, 9, 18, 20, 26 |
| seed_3 | add mod 59 | mlp | 0.5 | 2 | 100 | 1,080 | 980 steps, 11x | 1.000 | 6, 7, 16, 27, 28 |
| seed_4 | add mod 59 | mlp | 0.5 | 2 | 100 | 940 | 840 steps, 9x | 1.000 | 2, 6, 9, 14, 19 |
| seed_5 | add mod 59 | mlp | 0.5 | 2 | 100 | 1,140 | 1,040 steps, 11x | 1.000 | 1, 4, 16, 22, 29 |
| train_frac_0.2 | add mod 59 | mlp | 0.2 | 2 | 80 | never | - | 0.001 | none |
| train_frac_0.3 | add mod 59 | mlp | 0.3 | 2 | 100 | never | - | 0.137 | 15 |
| train_frac_0.4 | add mod 59 | mlp | 0.4 | 2 | 100 | 1,500 | 1,400 steps, 15x | 1.000 | 5, 11, 14, 15, 22 |
| train_frac_0.5 | add mod 59 | mlp | 0.5 | 2 | 100 | 860 | 760 steps, 9x | 1.000 | 3, 5, 14, 15 |
| train_frac_0.6 | add mod 59 | mlp | 0.6 | 2 | 120 | 580 | 460 steps, 5x | 1.000 | 3, 5, 15, 16 |
| weight_decay_0 | add mod 59 | mlp | 0.5 | 0 | 100 | never | - | 0.000 | none |
| weight_decay_0.25 | add mod 59 | mlp | 0.5 | 0.25 | 100 | never | - | 0.010 | 15 |
| weight_decay_0.5 | add mod 59 | mlp | 0.5 | 0.5 | 100 | never | - | 0.318 | 15 |
| weight_decay_1.0 | add mod 59 | mlp | 0.5 | 1 | 100 | 1,440 | 1,340 steps, 14x | 1.000 | 3, 5, 15 |
| weight_decay_2.0 | add mod 59 | mlp | 0.5 | 2 | 100 | 860 | 760 steps, 9x | 1.000 | 3, 5, 14, 15 |
| weight_decay_4.0 | add mod 59 | mlp | 0.5 | 4 | 120 | 580 | 460 steps, 5x | 1.000 | 3, 5, 14, 15 |
| wide_split | add mod 97 | transformer | 0.4 | 1 | 150 | 2,550 | 2,400 steps, 17x | 1.000 | 19, 29, 31, 36, 39 |
