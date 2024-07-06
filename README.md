# Computational model of layer 2/3 in mouse primary visual cortex

The code in this repository accompanies the article "Computational model of layer 2/3 in mouse primary visual cortex explains observed visuomotor mismatch response."

To run the code for the main article, use the following command:
```bash
python mismatch_response_simulation1.py
```
Close a figure to see the next figure. This code produces five figures. To test the variant with randomly distributed preferred values in a population code, set the variable `uniform_code` to `False` in the function `experiment`.

To run the code to compute the dependence on the speed-range offset, use the following command
```
python mismatch_response_vary_offset.py
```

To run the code for the multi-dimensional population code in the supplementary information, use the following command:
```bash
python mismatch_response_simulation2.py
```
Close a figure to see the next figure. This code produces two figures.
