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

# Read in the original graph
graph = nx.MultiDiGraph()

with open("reference_data.txt") as infile:
    for line in infile:
        line_elements = line.split(" ")

        print(line_elements)

        r_start = int(line_elements[1])
        s_start = int(line_elements[2])
        n_start = int(line_elements[3])
        c_start = int(line_elements[4])
        t_start = int(line_elements[5])
        l_start = int(line_elements[6])

        r_end = int(line_elements[8])
        s_end = int(line_elements[9])
        n_end = int(line_elements[10])
        c_end = int(line_elements[11])
        t_end = int(line_elements[12])
        l_end = int(line_elements[13])

        duration = float(line_elements[15])

        if s_start == -1:
            s_start = 0
            t_start = 1  # TODO except of picking a hardcoded value here we should select 1 from the dataset itself
            c_start = 1
        if s_end == -1:
            s_end = 0
            t_end = 1
            c_end = 1
        if n_start == -1:
            n_start = 0
        if n_end == -1:
            n_end = 0
        # c_start, t_start, c_end, t_end

        if c_end < 3 and c_start < 3 and t_start < 2 and t_end < 2:
            # Let's reduce the number of tools and configs for now
            graph.add_edge((s_start, n_start), (s_end, n_end), c_start=c_start, t_start=t_start, c_end=c_end,
                           t_end=t_end, weight=duration)

nx.write_gpickle(graph, f"reference_graph.gpickle")
