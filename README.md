# SCandBigFive

This is the code used for prediction of the big five personality trait scores from the brain structural connectome SC.

## Requirements
We ran this code with python 3.10. Following other packages are required:
- numpy==1.25.1
- pandas==2.0.3
- scikit_learn==1.3.0
- scipy==1.11.1

## Data Structure
The SCs used for the prediction are expected to be in the following structure:
```
DATA_DIR
    ├── sub-ID_001
    │	├── 031 (Granularity of the first parcellation)
    │   │   ├── 031_MIST_10M_mni152_count.csv
    │   │   ├── 031_MIST_10M_mni152_countsift2.csv
    │   │   ├── 031_MIST_10M_mni152_fa.csv
    │   │   └── 031_MIST_10M_mni152_md.csv
    │   ├── 038 (Granularity of the second parcellation)
    │	│   ├── 038_Craddock_10M_mni152_count.csv
    │   │   ├── 038_Craddock_10M_mni152_countsift2.csv
    │   │   ├── 038_Craddock_10M_mni152_fa.csv
    │   │   └── 038_Craddock_10M_mni152_md.csv
    .   .   .
	└── 210 (Granularity of the last parcellation)
    │   │   ├── 210_Brainnetome_10M_mni152_count.csv
    │   │   ├── 210_Brainnetome_10M_mni152_countsift2.csv
    │   │   ├── 210_Brainnetome_10M_mni152_fa.csv
    │   │   └── 210_Brainnetome_10M_mni152_md.csv
    ├── sub-ID_002
    .   .   .
    .   .   .
    .   .   .
```
### Split Files
We predefined the datasplits such that the data does not need to be re-split for each repetition of the prediction for different pipeline conditions. For each subject group (mixed sex, only males and only females) we defined 100 random 5-fold CV splits of the subjects and saved them to csv files. For each of the 100 repetitions, there were therefore ten csv files: five training sets and five test sets.
## Code Structure
Various pipeline conditions were varried for the prediction. Different feature classes are separated into different folders and scripts:
- *k* most correlated features &rarr; `corr`
- *k* principal components &rarr; `pca`
- regional connectivity profiles (rows of the matrix) &rarr; `rcp`
- upper triangle of the SC &rarr; `wholebrain`

The scripts ending with `_cv.py` determined the optimal parameter *k* and the optimal rcp in the inner loop of the nested cross validation. Scripts without this ending evaluated the test set accuracy for all considered parameters k and for all rcps.
In each of the different scripts the other pipeline conditions can be varried (parcellation, SC weighting, subject group, target). They can either be passed as arguments from the command-line or set explicitely in the script. If the pipeline condtitions are provided both as arguments in the command-line and set explicitely within the script, by default, the settings provided as arguments from the command-line will be used.

## Running Code
The scripts can be run as follows (example for the PCA feature class):
`python -m scr.pca.pca 070 /Users/amelie/Datasets/Splits/unrelated 070_DesikanKilliany_10M_mni152_count.csv log10 personality`

## Output Files
The output is saved in folders with the following naming convention:
`<SAVEDIR>/<atlas_granularity>_<weighting>_<normalization>_<feature_class>_<target>_<subject_group>`
In each folder, there are 100 csv-files named 0-100 for the 100 random splits of the data. Each file has 23 lines.
- Lines 1-5: MAE of the five outer loops for the training set
- Lines 6-10: Pearson's correlation of the five outer lopps for the training set
- Lines 11-15: MAE of the five outer loops for the test set
- Lines 16-20: Pearsons's correlation of the five outer loops for the test set
- Line 21: Pearson's correlation over all training splits
- Line 22+23: Pearson's correlation over all test splits (i.e. entire dataset) and corresponding p-value

For the wholebrain feature class and the experiments selecting the best number of features or RCP in the inner loop of the nested CV (_CV files), each csv file has only one column. For all other cases the csv file has several columns corresponding to different numbers *k* of most correlated features / principal components or the different evaluated RCPs (one column per RCP &rarr; the number of columns varries with the granularity of the parcellation)
