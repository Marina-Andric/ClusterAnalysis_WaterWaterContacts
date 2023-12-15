from sklearn.metrics import silhouette_score
from gmm import gmm
from copy import copy



columns_all = ['ΔE', 'P1/P2', 'O…O', 'O-H-O', 'O_H1-O_H3', 'O_H1-O_H4', 'O_H2-O_H3',
       'O_H2-O_H4', 'min H…O', 'min H…H', 'H1…H4', 
       'H-O-H-H3', 'H-O-H-O', 'H-O-H-H4', 'O-H1-H3', 'O-H1-H4', 'H1-O-H2',
       'H3-O-H4', 'R(H1O)', 'r(H1O)', 'R(H1H3)', 'r(H1H3)', 'R(H1H4)',
       'r(H1H4)']


for set in ['all', 'all_neg', 'all_pos']:

       silhouetter_index = 0

       best_parameters = {'cluster2': {'eps' : 0, 'minpts': 0, 'features' : [], 'silhouetter_index' : 0, 'noise' : 0},
                          'cluster3' : {'eps' : 0, 'minpts': 0, 'features' : [], 'silhouetter_index' : 0, 'noise' : 0},
                          'cluster4' : {'eps' : 0, 'minpts': 0, 'features' : [], 'silhouetter_index' : 0, 'noise' : 0},
                          'cluster5' : {'eps' : 0, 'minpts': 0, 'features' : [], 'silhouetter_index' : 0, 'noise' : 0},
                          'cluster6' : {'eps' : 0, 'minpts': 0, 'features' : [], 'silhouetter_index' : 0, 'noise' : 0}}


       for n_cluster in range(2, 7):
              print("Doing clustering with " + str(n_cluster) + " clusters.")
              features = ['ΔE']
              for feature_length in range(2, len(columns_all)+1):
                     print("Feature length: ", feature_length)
                     found = 0
                     cols = copy(features)
                     remaining_features = [item for item in columns_all if item not in features]

                     if len(remaining_features) == 0:
                            break

                     current_loop_best_feature = remaining_features[0]
                     current_loop_max_silhouetter_index = {'cluster2':0, 'cluster3':0,'cluster4':0,'cluster5':0,'cluster6':0}

                     for current_feature in remaining_features:

                            cols.append(current_feature)
                            print("Current features: ", cols)

                            silhouette_indexes = []
                            calinski_harabasz_scores = []
                            davies_bouldin_scores = []
                            all_results = []

                            res_gmm = gmm(set, n_cluster, cols)

                            cluster = res_gmm['cluster']

                            silhouetter_index = silhouette_score(res_gmm[cols], res_gmm['cluster'])

                            if silhouetter_index > current_loop_max_silhouetter_index['cluster'+str(n_cluster)]:
                              
                              current_loop_max_silhouetter_index['cluster'+str(n_cluster)] = silhouetter_index
                              current_loop_best_feature = current_feature


                            if silhouetter_index > best_parameters['cluster'+str(n_cluster)]['silhouetter_index']:
                                 found += 1
                                 if found > 1:
                                        features = features[:-1]
                                 features.append(current_feature)

                                 print("Update for cluster " + str(n_cluster))
                                 print('SI: ', silhouetter_index)
                                 best_parameters['cluster'+ str(n_cluster)]['silhouetter_index'] = silhouetter_index
                                 best_parameters['cluster'+ str(n_cluster)]['features'] = copy(features)


                            cols = [item for item in cols if item != current_feature]
                     if found == 0:
                            print("No more useful features. Added feature: ", current_loop_best_feature)
                            features.append(current_loop_best_feature)


       print("Best parameters", best_parameters, flush=True)
