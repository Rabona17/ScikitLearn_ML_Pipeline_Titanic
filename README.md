# ScikitLearn_ML_Pipeline_Titanic
Practice of Feature Engineering with Scikit-Learn Machine Learning Pipeline on (minimized) Titanic dataset. The training set only includes several features but the model has pretty good accuracy.

The score of this RandomForestClassifier is consistently around **82%** after using GridSearch and 5-fold Cross-Validation for parameter choosing.

## Feature Engineering
* `One-hot Encoder` for `PClass`
* `FunctionTransformer` and `One-hot encoder` for `Name` and extracted prefixes
* `Z-score Encoder` for `Age` by grouping on `P-Class`
* `Ordinal Encoder` for `Sex`
* `FunctionTransformer` for `Siblings/Spouses Aboard`,`Parents/Children Aboard`, and `Fare`

* `ColumnTransformer` allows us to input a DataFrame and make prediction using Pipeline
