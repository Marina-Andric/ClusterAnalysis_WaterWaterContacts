from sklearn import mixture
from data_process import import_data
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_gmm(n_components, data):


    gmm = mixture.GaussianMixture(n_components=n_components,
                                    covariance_type='full', max_iter=500, random_state=0)
    gmm.fit(data)
    
    return gmm


def gmm(set, n_components, cols):

    data_original = import_data(set)

    data = data_original[cols]

    sc = StandardScaler()

    data_std = pd.DataFrame(sc.fit_transform(data))

    gmm = fit_gmm(n_components, data_std)

    labels = gmm.predict(data_std)

    data_original['cluster'] = labels

    return data_original
