# Classification
> - RF @楊子嫻 
> 以下兩者擇一使用
> - KNN @劉邦均 Benny 
> - Deep Learning (DNN) @王心妤
## What do we do?
Classify the sample points that the classifier is less sure about into unknown classes and store to unknown ouput. If it is sure about the sample point, store to the known output.

## Output name
### unknown
1. RF -> `RF_unknown.csv`
2. KNN -> `KNN_unknown.csv`
3. Deep Learning (DNN) -> `DNN_unknown.csv`
### known
1. RF -> `RF_known.csv`
2. KNN -> `KNN_known.csv`
3. Deep Learning (DNN) -> `DNN_known.csv`

## Output format
### known
Same as `test_label.csv` but did not include the unkanown class and given them the label we predict,
### unknown
Same as `test_data.csv` but only with unknown class.
