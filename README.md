# DETERMINING SWISS GERMAN DIALECTS USING LINEAR ALGEBRA

## Authors

- [Olesia Hapiuk](https://github.com/olkaleska)
- [Maksym Hobela](https://github.com/MaksHobela)
- [Myron Ivanytskyi](https://github.com/ivamax-create)

## Description
Swiss German has many regional dialects that sound very different from each other.
This project tries to solve a simple problem: given a sentence which dialect is it?
The program trains a custom SVM model using character-level n-grams and classifies text into one of four dialects:
- Bern (BE);
- Basel (BS);
- Luzern (LU);
- Zurich (ZH).

You can adjust training parameters, validate accuracy on test data and even try your own sentences in interactive mode.
For a full technical breakdown, see the project report. The main algorithm is implemented in ```train_data.py```.

## Requirements

Install the required libraries before running:

```
pip install -r requirements.txt
```

> ⚠️ The program will not launch without this step.

---

## Usage

```
python main.py --ratio <[0,1]> --c <[1, +∞]> --eta <[0,1]> --interactive
```

### CLI Flags

| Flag | Type | Description | Default |
|------|------|-------------|---------|
| `--ratio` | `float` | Proportion of data for training (0.0 to 1.0). The remainder is used for testing. | `0.75` |
| `--c` | `float` | SVM regularization parameter. Higher `C` reduces training errors but increases overfitting risk. | `100.0` |
| `--eta` | `float` | Learning rate for the gradient descent optimizer. | `0.01` |
| `--interactive`  | `flag` | Enables interactive mode after training to test custom sentences. | `False` |

### Usage Examples

**Example 1 — standard validation**

Input:
```
python main.py --ratio 0.8 --c 110 --eta 0.011
```

Output:
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

**Example 2 — interactive mode**

Input:
```
python main.py --ratio 0.8 --c 110 --interactive
```

Output:
```
Saved 941 sentences for clean validation to test_dataset.csv
Dialect BE: 1995 features saved (Train size: 3763)
Dialect BS: 1302 features saved (Train size: 3763)
Dialect ZH: 1490 features saved (Train size: 3763)
Dialect LU: 1093 features saved (Train size: 3763)
Validating 941 sentences...
==================================================
RESULT: 54.73% accuracy
==================================================
MOST COMMON DIALECT ERRORS:
==================================================
Is LU, was identified as BE: 96
Is ZH, was identified as BE: 87
Is BS, was identified as BE: 81
Is ZH, was identified as BS: 28
Is BE, was identified as LU: 28
Is BE, was identified as BS: 23
Is LU, was identified as ZH: 22
Is BS, was identified as ZH: 20
Is LU, was identified as BS: 13
Is ZH, was identified as LU: 11
Is BS, was identified as LU: 11
Is BE, was identified as ZH: 6
==================================================
==================================================
SYSTEM READY (C=110.0, Ratio=0.8, Eta=0.01)
==================================================
Enter Swiss sentence (or 'exit'): das isch schwirig gnue gsi
Result: BE (Score: -0.95)
Enter Swiss sentence (or 'exit'): plaage schtööre we me sim
Result: BE (Score: -0.90)
Enter Swiss sentence (or 'exit'): exit
```

---

## Videos

Short videos from each team member:

- **Olesia** — [Watch on YouTube](https://youtu.be/ZUUyzA5NPTM?si=iPeY026aJdrU_mgl)
- **Maksym** — [Watch on YouTube](https://www.youtube.com/watch?v=-wPdc0DLMuc&t=7s)
- **Myron** — [Watch on YouTube](https://youtu.be/943y6-C0Inc?si=agyytCHeWBqM3Spq)
