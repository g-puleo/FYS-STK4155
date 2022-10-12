Linear regression as an example of machine learning, with an analysis of the Franke function and the territory of Møsvatn Austfjell.
==============================
In this project we implement codes which we can use to perform linear regression, using OLS, Ridge and LASSO methods.
A detailed description of the methods and of our results can be found in the file called `report.pdf` inside this folder.

## Requirements

It is necessary to have python3 installed, in particular we tested our code with the following modules installed:

* `scikit-learn` ver. = 0.24.2
* `numpy` ver. = 1.20.2 
* `pandas` ver. = 1.2.5
* `seaborn` ver. =0.11.1
* `matplotlib` ver. =3.3.4


## Reproduction of results

To reproduce the test runs using the Franke function, run 

		python3 main.py

Optionally, the results related to this part can be also visualized using the jupyter-notebook at `test/FrankePlots.ipynb`<br>

To reproduce the regressions on terrain data, run

		python3 terrain.py

## Summary of codes

The algorithms are coded into different files contained in the `src` folder. Hereby we briefly summarize them:

1. 	`franke_fit.py` contains the definition of a `Solver` function, which is repeatedly used in `main.py` and `terrain.py`.
	In this file there are also the definitions of functions which perform OLS, Ridge, and LASSO regressions. 

2. 	`utils.py` is a tiny library of small functions we wrote to improve the readability of the code

3. 	`plotting_functions.py` contains a series of functions which are utilized to display the results of tests using the Franke function.

## Aknowledgements

[Gianmarco](https://github.com/giammy00)

[Ivar](https://github.com/ivarlon)

[Elias](https://github.com/EliasTRuud)

