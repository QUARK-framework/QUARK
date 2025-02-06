#  Copyright 2021 The QUARK Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import networkx as nx
import pickle
import tsplib95

# Source http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/
filename = "dsj1000.tsp"


def main():
    """
    Load a TSP problem, remove unnecessary edges, and save the reference graph.
    """
    print(f"Loading {filename}")

    # Load the problem from .tsp file
    problem = tsplib95.load(filename)
    graph = problem.get_graph()

    # We don't needed edges from e.g. node0 -> node0
    for edge in graph.edges:
        if edge[0] == edge[1]:
            graph.remove_edge(edge[0], edge[1])

    print("Loaded graph:")
    print(nx.info(graph))

    with open("reference_graph.gpickle", "wb") as file:
        pickle.dump(graph, file, pickle.HIGHEST_PROTOCOL)

    print("Saved graph as reference_graph.gpickle")


if __name__ == '__main__':
    main()
