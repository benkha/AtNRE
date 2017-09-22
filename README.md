# AtNRE: Adversarial training for Neural Relation Extraction

This repository is the source code for the [paper](https://people.eecs.berkeley.edu/~russell/papers/emnlp17-relation.pdf):

### Adversarial Training for Relation Extraction

*Yi Wu, David Bamman, Stuart Russell*

*University of California, Berkeley*

*Conference on Empirical Methods in Natural Language Processing (EMNLP) 2017, Copenhagen, Denmark.*

### Requirements
* Python 3
* TensorFlow

### Instructions
1. Place the pickled data in `code/pkl_data`.
2. Run `script_nyt_adv.sh`. You can adjust the parameters of the model in this bash script.
3. You will see the timestamp printed out like this
   ```
   Timestamp 1506105808
   ```
   The timestamp will determine the directory TensorBoard uses.

    To launch TensorBoard, run the following line
    ```
    tensorboard --logdir=code/tensorboard/1506105318
    ```
    replacing `1506105808` with the timestamp of the current run.

    ![TensorBoard](figures/tb.png)
4. To run the model from a saved checkpoint, use the `--warmstart` option to `bug_runner.py`.


### Datasets

For data, please refer to the references in our paper and download from the original sources of the datasets.

*Original NYT dataset ([paper](http://aclweb.org/anthology/P16-1200), [link](https://github.com/thunlp/NRE/tree/master/data))*

*Original NAACL dataset ([paper](https://aclweb.org/anthology/N/N16/N16-1104.pdf), [link](https://www.cs.washington.edu/ai/gated_instructions/naacl_data.zip))*

For reproducibility of our results, [here](https://people.eecs.berkeley.edu/~jxwuyi/data/AtNRE_processed_data.zip) is the processed pickled data used in the code. **PLEASE** (1) do not distributed and (2) refer to the original data sources for either personal use or academic purpose.

The code is under BSD-3 license.
