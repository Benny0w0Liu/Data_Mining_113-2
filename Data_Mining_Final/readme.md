# Goal
- dataset: [gene expression cancer RNA-Seq Data Set](https://drive.google.com/drive/folders/1m_ZWyapB2WLpaC7tztc4Kzx_6J_Ugzxc)
# Process
## classification
Classify the sample points(`test_data.csv`) that the classifier is less sure about into unknown classes and store to unknown ouput. If it is sure about the sample point, store to the known output.

## clustering
cluster the unknown classes from `classification/result`. There are 2 unknown class in the data set.

## summarizing
comebine classification/result and clustering/result, and also make some chart to analyze each algorithm
# File Structure
- **classification**
    - **result**
        - `KNN_known.csv` -> result of *KNN.py*, the sure classes' lables of dataset
        - `KNN_unknown.csv` -> result of *KNN.py*, the unknown classes' data of dataset
        -
        -
        -
        -
    - `KNN.py`-> classify *test_data.csv* by KNN algorithm
    -
    -
- **clustering**
    - **result**
        - `AP_result.csv`
        -
        -
    - `AP.py`
    - 
    - 
- **summarizing**
    - **result**
Out of result is code. 

# Division of work
- Classification 
    - RF @楊子嫻 
    - 以下兩者擇一使用
        - KNN @劉邦均 
        - Deep Learning (DNN, CNN, RNN) @王心妤
- Clustering 
    - DBSCAN @王心妤
    - Agglomerative @林芷榆 
    - Affinity Propagation @劉邦均