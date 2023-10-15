# Copyright [2024] [Center for the Study of Movement, Cognition, and Mobility. Tel-Aviv Sourasky Medical Center, Tel Aviv, Israel]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import plotly.express as px

from constants import __ALL_FEATURE_NAMES_LIST__

class Clustering():

    def __init__(self, processedData, processedlabels):

        self.X = processedData[list(set(processedData.columns).intersection(__ALL_FEATURE_NAMES_LIST__))]
        self.y = processedlabels



    def find_clusters_with_GMM(self):

        pca_comps = self.cluster()
        gmm_4 = GaussianMixture(n_components=4, random_state=0).fit_predict(pca_comps)
        gmm_3 = GaussianMixture(n_components=3, random_state=0).fit_predict(pca_comps)
        gmm_2 = GaussianMixture(n_components=2, random_state=0).fit_predict(pca_comps)
        gmm_5 = GaussianMixture(n_components=5, random_state=0).fit_predict(pca_comps)

        # colors_dict = {1: "red", 0: "blue"}

        fig = px.scatter_matrix(
            pca_comps,
            dimensions=range(2),
            color=gmm_4,
        )
        fig.update_traces(diagonal_visible=False)
        fig.show()

        self.X["gmm2"] = gmm_2
        self.X["gmm3"] = gmm_3
        self.X["gmm4"] = gmm_4
        self.X["gmm5"] = gmm_5

        return self.X

    def cluster(self, plot=False):

        pca = PCA(n_components=2)
        components = pca.fit_transform(self.X)

        # cluster = self.X["gmm4"]

        if plot:

            labels = {
                str(i): f"PC {i + 1} ({var:.1f}%)"
                for i, var in enumerate(pca.explained_variance_ratio_ * 100)
            }

            colors_dict = {1:"red", 0:"blue"}

            fig = px.scatter_matrix(
                components,
                labels=labels,
                dimensions=range(2),
                color=[colors_dict[y] for y in self.y],
            )
            fig.update_traces(diagonal_visible=False)
            fig.show()

        return components