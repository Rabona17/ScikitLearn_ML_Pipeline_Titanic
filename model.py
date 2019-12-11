def titanic_model(titanic):
    """
    :Example:
    >>> fp = os.path.join('data', 'titanic.csv')
    >>> data = pd.read_csv(fp)
    >>> pl = titanic_model(data)
    >>> isinstance(pl, Pipeline)
    True
    >>> from sklearn.base import BaseEstimator
    >>> isinstance(pl.steps[-1][-1], BaseEstimator)
    True
    >>> preds = pl.predict(data.drop('Survived', axis=1))
    >>> ((preds == 0)|(preds == 1)).all()
    True
    """
    def scale(fare_arr):
        def helper(fare):
            if fare[0] <= 7.91:
                return 0
            elif fare[0] <= 14.454:
                return 1
            elif fare[0] <= 31:
                return 2
            else:
                return 3
        return pd.DataFrame(fare_arr).apply(helper, axis=1).to_frame().values
    
    def prefix(ndarr):
        l=pd.DataFrame(ndarr).iloc[:, 0].apply(lambda x:re.findall('^([a-zA-Z]+\.)', x)[0])
        l = l.replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')
        l = l.replace('Mlle', 'Miss')
        l = l.replace('Ms', 'Miss')
        l = l.replace('Mme', 'Mrs')
        return l.to_frame().values
    titanic_cp = titanic.copy()
    prefix_col = ['Name']
    prefix_pl = Pipeline([('pre', FunctionTransformer(prefix, validate=False)),
                          ('onehot', OneHotEncoder(handle_unknown='ignore'))
                         ])

    onehot_cols = ['Pclass']
    onehot = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])

    age_col = ['Pclass', 'Age']
    age_z = Pipeline([('age', StdScalerByGroup())])

    ord_cols = ['Sex']
    ordi = Pipeline([('func', OrdinalEncoder())])

    keep_cols = ['Siblings/Spouses Aboard','Parents/Children Aboard']
    keep = Pipeline([('func', FunctionTransformer(lambda x:x))])

    fare_cols = ['Fare']
    fare = Pipeline([('fare', FunctionTransformer(scale))])

    prep = ColumnTransformer(transformers=[('pre', prefix_pl, prefix_col), ('onehot', onehot, onehot_cols), ('age', age_z, age_col),('sex', ordi, ord_cols),('keep', keep, keep_cols), ('fare', fare, ['Fare'])])
    pl = Pipeline(steps=[('pp', prep), ('classifier', RandomForestClassifier(max_depth=9))])
    pl_fitted = pl.fit(titanic_cp.drop('Survived', axis=1), titanic_cp.Survived)
    return pl_fitted
