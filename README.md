## REQUIREMENTS
At first install the next libraries using next command:
```
pip install -r requirements.txt
```
If you don't do this, the programm won't launch.
## USAGE
To launch the programm, you need to use next command:
```
python main.py --ratio <[0,1]> --c <[1, +∞]> --eta <[0,1]> --interactive
```
### CLI Flags

| Flag | Type | Description | Default |
|------|------|-------------|---------|
| `--ratio` | `float` | Proportion of data for training (0.0 to 1.0). The remainder is used for testing. | `0.75` |
| `--c` | `float` | SVM regularization parameter. Higher `C` reduces training errors but increases overfitting risk. | `100.0` |
| `--eta` | `float` | Learning rate for the gradient descent optimizer. | `0.01` |
| `--interactive` | `flag` | Enables interactive mode after training to test custom sentences. | `False` |

### Usage example
Expected input:
```
python main.py --ratio 0.8 --c 110 --eta 0.011
```
Expected output:

```
Saved 941 sentences for clean validation to test_dataset.csv
Dialect BE: 1995 features saved (Train size: 3763)
Dialect BS: 1302 features saved (Train size: 3763)
Dialect ZH: 1490 features saved (Train size: 3763)
Dialect LU: 1093 features saved (Train size: 3763)

Validating 941 sentences...

==================================================
RESULT: 58.66% accuracy
==================================================
MOST COMMON DIALECT ERRORS:
==================================================
Is BS, was identified as ZH: 73
Is LU, was identified as ZH: 65
Is BE, was identified as LU: 40
Is LU, was identified as BE: 39
Is BE, was identified as ZH: 27
Is ZH, was identified as BE: 27
Is BS, was identified as BE: 27
Is ZH, was identified as BS: 25
Is BE, was identified as BS: 22
Is LU, was identified as BS: 16
Is ZH, was identified as LU: 16
Is BS, was identified as LU: 12
==================================================

==================================================
SYSTEM READY (C=110.0, Ratio=0.8, Eta=0.011)
==================================================
```

