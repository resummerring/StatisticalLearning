import numpy as np
import pandas as pd

from typing import List
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold


class ClusterImportance:

    """
    Compute cluster-level feature importance, which is not subject to substitution effect.
    """

    @staticmethod
    def _group_mean_std(importance: pd.DataFrame, cluster: dict) -> pd.DataFrame:
        """
        Aggregate MDI across ensemble of tress

        :param importance: pd.DataFrame, individual feature importance
        :param cluster: dict, mapping from cluster index to feature name
        """

        output = pd.DataFrame(columns=['Mean', 'Std'])
        for key, value in cluster.items():
            sub_cluster = importance[value].sum(axis=1)
            output.loc[f'Cluster_{key}', 'Mean'] = sub_cluster.mean()
            output.loc[f'Cluster_{key}', 'Std'] = sub_cluster.std() * (sub_cluster.shape[0] ** -0.5)

        return output

    @staticmethod
    def cluster_mean_decreased_impurity(model: object, features: List[str], cluster: dict) -> pd.DataFrame:
        """
        Compute cluster-level mean decreased impurity:
            Clustered MDI = sum of feature MDI in a given cluster for each tree
            Mean/Std Clustered MDI = Mean/Std for clustered MDI across ensemble of tress

        :param model: object, a tree-based ensemble classification method  with estimators_ attribute
        :param features: List[str], feature names
        :param cluster: dict, mapping from cluster index to feature name
        """

        importance = {i: tree.feature_importances_ for i, tree in enumerate(model.estimators_)}
        importance = pd.DataFrame.from_dict(importance, orient='index')
        importance.columns = features
        importance.replace(0, np.nan)

        cluster_importance = ClusterImportance._group_mean_std(importance, cluster)
        cluster_importance = cluster_importance / cluster_importance['Mean'].sum()

        return cluster_importance

    @staticmethod
    def cluster_mean_decreased_accuracy(model: object,
                                        X: pd.DataFrame,
                                        y: pd.Series,
                                        cluster: dict,
                                        n_split: int) -> pd.DataFrame:
        """
        Compute cluster-level mean decreased accuracy:
            Shuffle all the features of a given cluster at a time

        :param model: object, a fittable classification model
        :param X: pd.DataFrame, data matrix
        :param y: pd.Series, label series
        :param cluster: dict, mapping from cluster index to feature name
        :param n_split: int, number of split for cross validation
        """

        kfold = KFold(n_split)
        score, importance = pd.Series(), pd.DataFrame(columns=cluster.keys())

        for i, (train_index, test_index) in enumerate(kfold.split(X)):

            X_train, y_train = X.iloc[train_index, :], y.iloc[train_index]
            X_test, y_test = X.iloc[test_index, :], y.iloc[test_index]

            trained_model = model.fit(X=X_train, y=y_train)
            prob = trained_model.predict_proba(X_test)
            score.loc[i] = -log_loss(y_test, prob, labels=model.classes_)

            for col in importance.columns:
                data = X_test.copy(deep=True)
                for k in cluster[col]:
                    np.random.shuffle(data[k].values)
                prob = trained_model.predict_proba(data)
                importance.loc[i, col] = -log_loss(y_test, prob, labels=model.classes_)

        cluster_importance = (-1 * importance).add(score, axis=0)
        cluster_importance = cluster_importance / (-1 * importance)
        cluster_importance = pd.concat({'Mean': cluster_importance.mean(),
                                        'Std': cluster_importance.std() * cluster_importance.shape[0] ** -0.5}, axis=1)
        cluster_importance.index = [f'Cluster_{i}' for i in cluster_importance.index]

        return cluster_importance
