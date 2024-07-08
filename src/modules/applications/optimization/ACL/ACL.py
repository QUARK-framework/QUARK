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

from typing import TypedDict

import pandas as pd
import numpy as np
import pulp

from modules.applications.Application import *
from modules.applications.optimization.Optimization import Optimization
from utils import start_time_measurement, end_time_measurement


class ACL(Optimization):
    """
    The distribution of passenger vehicles is a complex task and a high cost factor for automotive original
equipment manufacturers (OEMs). On the way from the production plant to the customer, vehicles travel
long distances on different carriers such as ships, trains, and trucks. To save costs, OEMs and logistics service
providers aim to maximize their loading capacities. Modern auto carriers are extremely flexible. Individual
platforms can be rotated, extended, or combined to accommodate vehicles of different shapes and weights
and to nest them in a way that makes the best use of the available space. In practice, finding feasible
combinations is done with the help of simple heuristics or based on personal experience. In research, most
papers that deal with auto carrier loading focus on route or cost optimization. Only a rough approximation
of the loading sub-problem is considered. We formulate the problem as a mixed integer quadratically constrained
assignment problem.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__("ACL")
        self.submodule_options = ["MIPsolverACL", "Ising", "QUBO"]
        self.application = None

    @staticmethod
    def get_requirements() -> list:
        return [
            {
                "name": "pulp",
                "version": "2.7.0"
            },
            {
                "name": "pandas",
                "version": "2.1.0"
            },
            {
                "name": "numpy",
                "version": "1.23.5"
            },
            {
                "name": "openpyxl",
                "version": "3.1.2"
            }

        ]

    def get_default_submodule(self, option: str) -> Core:
        if option == "MIPsolverACL":
            from modules.solvers.MIPsolverACL import MIPaclp  # pylint: disable=C0415
            return MIPaclp()
        elif option == "Ising":
            from modules.applications.optimization.ACL.mappings.ISING import Ising  # pylint: disable=C0415
            return Ising()
        elif option == "QUBO":
            from modules.applications.optimization.ACL.mappings.QUBO import Qubo  # pylint: disable=C0415
            return Qubo()
        else:
            raise NotImplementedError(f"Submodule Option {option} not implemented")

    def get_parameter_options(self):
        return {
            "model_select": {
                "values": list(["Full", "Small", "Tiny"]),
                "description": "Do you want the full model or a simplified (QUBO friendly) one?"
            }
        }

    class Config(TypedDict):
        model_select: str

    @staticmethod
    def intersectset(p1, p2):
        return np.intersect1d(p1, p2).tolist()

    @staticmethod
    def diffset(p1, p2):
        return np.setdiff1d(p1, p2).tolist()

    def generate_problem(self, config: Config):  # pylint: disable=R0915
        """
        This function includes three models: Full, small and tiny. Full refers to the original model with all of its
        constraints. Small refers to the simplified model which is more suitable for solving it with QC methods.
        The simplified model does not consider the possibility that vehicles can be angled or that they can be
        oriented forwards or backwards in relation to the auto carrier. For the tiny version we do not consider split
        platforms and consider only weight constraints.
        """
        # Enter vehicles to load (BMW model codes)
        vehicles = ["G20", "G20", "G20", "G20", "G07", "G20"]
        # Import vehicle data
        df = pd.read_excel(os.path.join(os.path.dirname(__file__), "Vehicle_data_QUARK.xlsx"))
        model_select = config['model_select']

        # All the parameters are given in decimeters -> 4m == 400 cm == 40 dm or decitons -> 2 tons -> 20 dt
        # Below are the model specific parameters, constraints and objectives for the tiny, small and the full model

        if model_select == "Tiny":
            # Weight parameters
            # max. total weight on truck / trailer
            wt = [100]
            # wt = [10]
            # max. weight on the four levels
            wl = [50, 60]
            # wl = [5, 6]
            # max. weights on platforms p, if not angled
            wp = [23, 23, 23, 26, 17]
            # wp = [2, 2, 2, 2, 1]

            # Create empty lists for different vehicle parameters. This is required for proper indexing in the model
            weight_list = [0] * (len(vehicles))

            for i in set(range(len(vehicles))):
                df_new = df.loc[df['Type'] == vehicles[i]]
                # df_new = (df.loc[df['Type'] == vehicles[i]])
                weight_list[i] = int(df_new["Weight"].iloc[0])
                # weight_list[i] = int(int(df_new["Weight"].iloc[0])/10)

            # Construct sets
            # Set of available cars
            vecs = set(range(len(vehicles)))
            # Set of available platforms
            platforms_array = np.array([0, 1, 2, 3, 4])
            plats = set(range(len(platforms_array)))

            # Set of platforms that have a limitation on allowed weight
            platforms_level_array = np.array([[0, 1, 2], [3, 4]], dtype=object)
            plats_l = set(range(len(platforms_level_array)))

            # Set of platforms that form trailer and truck
            platforms_truck_trailer_array = np.array([[0, 1, 2, 3, 4]], dtype=object)
            plats_t = set(range(len(platforms_truck_trailer_array)))

            # Create decision variables
            # Vehicle v assigned to p
            x = pulp.LpVariable.dicts('x', ((p, v) for p in plats for v in vecs), cat='Binary')

            # Create the 'prob' variable to contain the problem data
            prob = pulp.LpProblem("ACL", pulp.LpMaximize)

            # Objective function
            # Maximize number of vehicles on the truck
            prob += pulp.lpSum(x[p, v] for p in plats for v in vecs)

            # Constraints
            # Assignment constraints
            # (1) Every vehicle can only be assigned to a single platform
            for p in plats:
                prob += pulp.lpSum(x[p, v] for v in vecs) <= 1

            # (2) Every platform can only hold a single vehicle
            for v in vecs:
                prob += pulp.lpSum(x[p, v] for p in plats) <= 1

            # (3) Weight limit for every platform
            for p in plats:
                for v in vecs:
                    prob += weight_list[v] * x[p, v] <= wp[p]

            # (4) Weight constraint for every level
            for p_l in plats_l:
                prob += pulp.lpSum(weight_list[v] * x[p, v] for p in platforms_level_array[p_l] for v in vecs) <= \
                        wl[p_l]

            # (5) Weight constraint for truck and trailer
            for t in plats_t:
                prob += pulp.lpSum(
                    weight_list[v] * x[p, v] for p in platforms_truck_trailer_array[t] for v in vecs) <= wt[t]

        elif model_select == "Small":

            # For the small model, we only consider two levels with 3 and 2 platforms each

            # Length parameters
            # platform lengths, extension, bounds on extension
            # Level 1 (Truck up), 2 (Truck down), 3 (Trailer up), 4 (Trailer down)
            # Consider maximum length of 20750, drivers cab 2350, distance between truck and trailer of 500
            # We do not consider continuous extension of the loading planes
            lmax_l = [97, 79]

            # Height parameters
            # Considers base truck height and height distance between vehicles (~10cm)
            hmax_truck = [34, 34, 33, 36, 32, 36]
            # [0, 3], [1, 3], [2, 3], [0, 4], [1, 4], [2, 4]

            # Weight parameters
            # max. total weight on truck / trailer
            wt = [100]
            # max. weight on the two levels
            wl = [65, 50]
            # max. weights on platforms p, if not angled
            wp = [23, 23, 23, 26, 17]
            # max. weight on p, if sp is used
            wsp = [28, 28, 28]

            # Create empty lists for different vehicle parameters. This is required for proper indexing in the model
            class_list = [0] * (len(vehicles))
            length_list = [0] * (len(vehicles))
            height_list = [0] * (len(vehicles))
            weight_list = [0] * (len(vehicles))

            for i in set(range(len(vehicles))):
                df_new = df.loc[df['Type'] == vehicles[i]]
                class_list[i] = int(df_new["Class"].iloc[0])
                length_list[i] = int(df_new["Length"].iloc[0])
                height_list[i] = int(df_new["Height"].iloc[0])
                weight_list[i] = int(df_new["Weight"].iloc[0])

            # Construct sets
            # Set of available cars
            vecs = set(range(len(vehicles)))
            # Set of available platforms
            platforms_array = np.array([0, 1, 2, 3, 4])
            plats = set(range(len(platforms_array)))

            # Set of possible split platforms
            split_platforms_array = np.array([[0, 1], [1, 2], [3, 4]], dtype=object)
            plats_sp = set(range(len(split_platforms_array)))

            # Set of platforms that have a limitation on allowed length and weight because they are on the same level
            platforms_level_array = np.array([[0, 1, 2], [3, 4]], dtype=object)
            plats_l = set(range(len(platforms_level_array)))

            # Set of platforms that form trailer and truck
            platforms_truck_trailer_array = np.array([[0, 1, 2, 3, 4]], dtype=object)
            plats_t = set(range(len(platforms_truck_trailer_array)))

            # Set of platforms that have a limitation on allowed height
            platforms_height_array_truck = np.array([[0, 3], [1, 3], [2, 3], [0, 4], [1, 4], [2, 4]],
                                                    dtype=object)
            plats_h1 = set(range(len(platforms_height_array_truck)))

            # Create decision variables
            # Vehicle v assigned to p
            x = pulp.LpVariable.dicts('x', ((p, v) for p in plats for v in vecs), cat='Binary')
            # Usage of split platform
            sp = pulp.LpVariable.dicts('sp', (q for q in plats_sp), cat='Binary')
            # Auxiliary variable for linearization of quadratic constraints
            gamma = pulp.LpVariable.dicts('gamma', (p for p in plats_sp), cat='Binary')

            # Create the 'prob' variable to contain the problem data
            prob = pulp.LpProblem("ACL", pulp.LpMaximize)

            # Objective function
            # Maximize number of vehicles on the truck
            prob += pulp.lpSum(x[p, v] for p in plats for v in vecs)

            # Constraints
            # Assignment constraints
            # (1) Every vehicle can only be assigned to a single platform
            for p in plats:
                prob += pulp.lpSum(x[p, v] for v in vecs) <= 1

            # (2) Every platform can only hold a single vehicle
            for v in vecs:
                prob += pulp.lpSum(x[p, v] for p in plats) <= 1

            # (3) If a split platform q in plats_sp is used, only one of its "sub platforms" can be used
            for q in plats_sp:
                prob += pulp.lpSum(x[p, v] for p in split_platforms_array[q] for v in vecs) \
                        <= len(split_platforms_array[q]) * (1 - sp[q]) + sp[q]

            # (4) It is always only possible to use a single split-platform for any given p
            for q in plats_sp:
                for p in plats_sp:
                    if p != q:
                        z = bool(set(split_platforms_array[q]) & set(split_platforms_array[p]))
                        if z is True:
                            prob += sp[q] + sp[p] <= 1

            # (5) Length constraint
            # Checks that vehicles v on platforms p that belong to level L are shorter than the maximum available length
            for L in plats_l:
                prob += (pulp.lpSum(x[p, v] * length_list[v] for p in platforms_level_array[L] for v in vecs)
                         <= lmax_l[L])

            # (6) Height constraints for truck and trailer, analogue to length constraints
            # Truck
            for h in plats_h1:
                prob += pulp.lpSum(x[p, v] * height_list[v] for p in platforms_height_array_truck[h] for v in vecs) \
                        <= hmax_truck[h]

            # (7) Linearization constraint -> gamma == 1, if split platform is used
            for q in plats_sp:
                prob += pulp.lpSum(
                    sp[q] + x[p, v] for p in self.intersectset(split_platforms_array[q], platforms_array)
                    for v in vecs) >= 2 * gamma[q]

            # (8) Weight limit for every platform
            for p in plats:
                for v in vecs:
                    prob += weight_list[v] * x[p, v] <= wp[p]

            # (9) If a split platform is used, weight limit == wsp, if not, then weight limit == wp
            for q in plats_sp:
                for p in split_platforms_array[q]:
                    prob += pulp.lpSum(weight_list[v] * x[p, v] for v in vecs) <= gamma[q] * wsp[q] \
                            + (1 - gamma[q]) * wp[p]

            # (10) Weight constraint for every level
            for p_l in plats_l:
                prob += pulp.lpSum(weight_list[v] * x[p, v] for p in platforms_level_array[p_l] for v in vecs) <= \
                        wl[p_l]

            # (11) Weight constraint for truck and trailer
            for p_t in plats_t:
                prob += pulp.lpSum(
                    weight_list[v] * x[p, v] for p in platforms_truck_trailer_array[p_t] for v in vecs) <= wt[p_t]
        else:
            # Horizontal Coefficients: Length reduction
            # 1 = forward, 0 = backward
            # [0:1, 1:1, 0:0, 1:0]
            v_coef = np.array([[0.20, 0.15, 0.14, 0.19],
                               [0.22, 0.22, 0.22, 0.22],
                               [0.22, 0.13, 0.12, 0.17]])

            # Vertical Coefficients: Height increase
            # [0:1, 1:1, 0:0, 1:0]
            h_coef = np.array([[0.40, 1, 1, 1],
                               [0.17, 0.22, 0.21, 0.22],
                               [0.17, 0.38, 0.32, 0.32]])

            # Length parameters
            # platform lengths, extension, bounds on extension
            # Level 1 (Truck up), 2 (Truck down), 3 (Trailer up), 4 (Trailer down)
            # Consider maximum length of 20750, drivers cab 2350, distance between truck and trailer of 500
            # We do not consider continuous extension of the loading planes
            lmax_l = [97, 79, 97, 97]

            # Height parameters
            # Considers base truck height and height distance between vehicles (~10cm)
            hmax_truck = [34, 34, 33, 36, 32, 36]
            # [0, 3], [1, 3], [2, 3], [0, 4], [1, 4], [2, 4]
            hmax_trailer = [36, 32, 32, 34]
            # [5, 7], [5, 8], [6, 8], [6, 9]

            # Weight parameters
            # max. total weight
            wmax = 180
            # max. total weight on truck / trailer
            wt = [100, 100]
            # max. weight on the four levels
            wl = [50, 60, 50, 90]
            # max. weights on platforms p, if not angled
            wp = [23, 23, 23, 26, 17, 26, 26, 26, 23, 26]
            # max. weights on p, angled (if possible: 1, 2, 4, 7, 8, 9):
            wpa = [20, 22, 17, 18, 19, 22]
            # max. weight on p, if sp is used
            wsp = [28, 28, 28, 28, 28, 28]

            # Create empty lists for different vehicle parameters. This is required for proper indexing in the model
            class_list = [0] * (len(vehicles))
            length_list = [0] * (len(vehicles))
            height_list = [0] * (len(vehicles))
            weight_list = [0] * (len(vehicles))

            for i in set(range(len(vehicles))):
                df_new = df.loc[df['Type'] == vehicles[i]]
                class_list[i] = int(df_new["Class"].iloc[0])
                length_list[i] = int(df_new["Length"].iloc[0])
                height_list[i] = int(df_new["Height"].iloc[0])
                weight_list[i] = int(df_new["Weight"].iloc[0])

            # Construct sets
            # Set of available cars
            vecs = set(range(len(vehicles)))
            # Set of available platforms
            platforms_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            plats = set(range(len(platforms_array)))

            # Set of platforms that can be angled
            platforms_angled_array = [1, 2, 4, 7, 8]
            vp = [0, 1, 3, 8, 9]  # Platforms "under" a_p
            plats_a = set(range(len(platforms_angled_array)))

            # Set of possible split platforms
            split_platforms_array = np.array([[0, 1], [1, 2], [3, 4], [5, 6], [7, 8], [8, 9]], dtype=object)
            plats_sp = set(range(len(split_platforms_array)))

            # Set of platforms that have a limitation on allowed length and weight because they are on the same level
            platforms_level_array = np.array([[0, 1, 2], [3, 4], [5, 6], [7, 8, 9]], dtype=object)
            plats_l = set(range(len(platforms_level_array)))

            # Set of platforms that form trailer and truck
            platforms_truck_trailer_array = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=object)
            plats_t = set(range(len(platforms_truck_trailer_array)))

            # Set of platforms that have a limitation on allowed height
            platforms_height_array_truck = np.array([[0, 3], [1, 3], [2, 3], [0, 4], [1, 4], [2, 4]],
                                                    dtype=object)
            platforms_height_array_trailer = np.array([[5, 7], [5, 8], [6, 8], [6, 9]], dtype=object)

            plats_h1 = set(range(len(platforms_height_array_truck)))
            plats_h2 = set(range(len(platforms_height_array_trailer)))

            # Create the 'prob' variable to contain the problem data
            prob = pulp.LpProblem("ACL", pulp.LpMaximize)

            # Create decision variables
            # Vehicle v assigned to p
            x = pulp.LpVariable.dicts('x', ((p, v) for p in plats for v in vecs), cat='Binary')
            # Usage of split platform
            sp = pulp.LpVariable.dicts('sp', (q for q in plats_sp), cat='Binary')
            # Direction of vehicle on p
            d = pulp.LpVariable.dicts('d', (p for p in plats), cat='Binary')
            # State of platform p in PA - angled == 1, not angled == 0
            a_p = pulp.LpVariable.dicts('a_p', (p for p in plats_a), cat='Binary')

            # Create auxiliary variables for linearization of quadratic constraints
            y1 = pulp.LpVariable.dicts('y1', (p for p in plats_a), cat='Binary')
            y2 = pulp.LpVariable.dicts('y2', (p for p in plats_a), cat='Binary')
            y3 = pulp.LpVariable.dicts('y3', (p for p in plats_a), cat='Binary')
            y4 = pulp.LpVariable.dicts('y4', (p for p in plats_a), cat='Binary')
            ay1 = pulp.LpVariable.dicts('ay1', (p for p in plats_a), cat='Binary')
            ay2 = pulp.LpVariable.dicts('ay2', (p for p in plats_a), cat='Binary')
            ay3 = pulp.LpVariable.dicts('ay3', (p for p in plats_a), cat='Binary')
            ay4 = pulp.LpVariable.dicts('ay4', (p for p in plats_a), cat='Binary')
            # Weight for split-platforms
            gamma = pulp.LpVariable.dicts('gamma', (p for p in plats_sp), cat='Binary')

            # Here the model starts, including objective and constraints

            # Objective function
            # Maximize number of vehicles on the truck
            prob += pulp.lpSum(x[p, v] for p in plats for v in vecs)

            # Constraints
            # Assignment constraints
            # (1) Every vehicle can only be assigned to a single platform
            for p in plats:
                prob += pulp.lpSum(x[p, v] for v in vecs) <= 1

            # (2) Every platform can only hold a single vehicle
            for v in vecs:
                prob += pulp.lpSum(x[p, v] for p in plats) <= 1

            # (3) If a split platform q in plats_sp is used, only one of its "sub platforms" can be used
            for q in plats_sp:
                prob += pulp.lpSum(x[p, v] for p in split_platforms_array[q] for v in vecs)\
                        <= len(split_platforms_array[q]) * (1 - sp[q]) + sp[q]

            # (3.1) It is always only possible to use a single split-platform for any given p
            for q in plats_sp:
                for p in plats_sp:
                    if p != q:
                        z = bool(set(split_platforms_array[q]) & set(split_platforms_array[p]))
                        if z is True:
                            prob += sp[q] + sp[p] <= 1

            # (3.2) It is not allowed to angle platforms next to empty platforms
            for i, p in enumerate(platforms_angled_array):
                prob += pulp.lpSum(x[p, v] + x[vp[i], v] for v in vecs) >= 2 * a_p[i]

            # Linearization constraints
            # Linearization of d_p and d_v(p) -> orientations of two neighboring cars
            for p in platforms_angled_array:
                z = platforms_angled_array.index(p)
                v_p = vp[z]
                prob += (1 - d[p]) + d[v_p] >= 2 * y1[z]
                prob += d[p] + d[v_p] >= 2 * y2[z]
                prob += (1 - d[p]) + (1 - d[v_p]) >= 2 * y3[z]
                prob += d[p] + (1 - d[v_p]) >= 2 * y4[z]

            # Linearization of a_p with y1 - y4 -> linear combination of angle and orientations
            for p in platforms_angled_array:
                z = platforms_angled_array.index(p)
                prob += a_p[z] + y1[z] >= 2 * ay1[z]
                prob += a_p[z] + y2[z] >= 2 * ay2[z]
                prob += a_p[z] + y3[z] >= 2 * ay3[z]
                prob += a_p[z] + y4[z] >= 2 * ay4[z]

            # Linearization of x * ay -> linear combination of assignment and orientation/angle
            xay1 = pulp.LpVariable.dicts('xay1', ((p, v) for p in plats_a for v in vecs), cat='Binary')
            xay2 = pulp.LpVariable.dicts('xay2', ((p, v) for p in plats_a for v in vecs), cat='Binary')
            xay3 = pulp.LpVariable.dicts('xay3', ((p, v) for p in plats_a for v in vecs), cat='Binary')
            xay4 = pulp.LpVariable.dicts('xay4', ((p, v) for p in plats_a for v in vecs), cat='Binary')
            for p in platforms_angled_array:
                z = platforms_angled_array.index(p)
                for v in vecs:
                    prob += ay1[z] + x[z, v] >= 2 * xay1[z, v]
                    prob += ay2[z] + x[z, v] >= 2 * xay2[z, v]
                    prob += ay3[z] + x[z, v] >= 2 * xay3[z, v]
                    prob += ay4[z] + x[z, v] >= 2 * xay4[z, v]

            # Making sure always only 1 case applies
            for p in platforms_angled_array:
                z = platforms_angled_array.index(p)
                prob += ay1[z] + ay2[z] + ay3[z] + ay4[z] <= 1
                prob += y1[z] + y2[z] + y3[z] + y4[z] <= 1

            # (4) Length constraint
            # Checks that vehicles v on platforms p that belong to level L are shorter than the maximum available length
            # The length of the vehicles depends on whether they are angled or not and which vehicle is standing on
            # platform v(p)
            for L in plats_l:
                prob += pulp.lpSum(x[p, v] * length_list[v]
                                   - xay1[platforms_angled_array.index(p), v] *
                                   int(v_coef[class_list[v]][0]*length_list[v])
                                   - xay2[platforms_angled_array.index(p), v] *
                                   int(v_coef[class_list[v]][1]*length_list[v])
                                   - xay3[platforms_angled_array.index(p), v] *
                                   int(v_coef[class_list[v]][2]*length_list[v])
                                   - xay4[platforms_angled_array.index(p), v]
                                   * int(v_coef[class_list[v]][3]*length_list[v])
                                   for p in self.intersectset(platforms_angled_array, platforms_level_array[L])
                                   for v in vecs)\
                        + pulp.lpSum(x[p, v] * length_list[v]
                                     for p in self.diffset(platforms_level_array[L], platforms_angled_array)
                                     for v in vecs) \
                        <= lmax_l[L]

            # (5) Platforms can not be angled, if they are part of a split platform
            for q in plats_sp:
                prob += pulp.lpSum(a_p[platforms_angled_array.index(p)]
                                   for p in self.intersectset(platforms_angled_array, split_platforms_array[q]))\
                        <= len(split_platforms_array[q]) * (1 - sp[q])

            # (6) Weight constraint if split platform is used, gamma == 1
            for q in plats_sp:
                prob += pulp.lpSum(sp[q] + x[p, v] for p in self.intersectset(split_platforms_array[q], platforms_array)
                                   for v in vecs) >= 2 * gamma[q]

            # If split platform is used, weight limit == wsp, if not, then weight limit == wp
            for q in plats_sp:
                for p in split_platforms_array[q]:
                    prob += (pulp.lpSum(weight_list[v] * x[p, v] for v in vecs) <= gamma[q] * wsp[q] + (1 - gamma[q]) *
                             wp[p])

            # (7) If a platform that can be angled is angled, weight limit == wpa
            # Need another linearization for that:
            apx = pulp.LpVariable.dicts('apx', ((p, v) for p in plats_a for v in vecs), cat='Binary')
            for p in platforms_angled_array:
                z = platforms_angled_array.index(p)
                for v in vecs:
                    prob += a_p[z] + x[z, v] >= 2 * apx[z, v]

            for p in platforms_angled_array:
                prob += pulp.lpSum(weight_list[v] * apx[platforms_angled_array.index(p), v] for v in vecs) \
                        <= wpa[platforms_angled_array.index(p)]

            # (8) Weight constraint for every level
            for p_l in plats_l:
                prob += (pulp.lpSum(weight_list[v] * x[p, v] for p in platforms_level_array[p_l] for v in vecs) <=
                         wl[p_l])

            # (9) Weight constraint for truck and trailer
            for p_t in plats_t:
                prob += (pulp.lpSum(weight_list[v] * x[p, v] for p in platforms_truck_trailer_array[p_t] for v in vecs)
                         <= wt[p_t])

            # (10) Weight constraint for entire auto carrier
            prob += pulp.lpSum(weight_list[v] * x[p, v] for p in plats for v in vecs) <= wmax

            # (11) Height constraints for truck and trailer, analogue to length constraints
            # Truck
            for h in plats_h1:
                prob += pulp.lpSum(x[p, v] * height_list[v]
                                   - xay1[platforms_angled_array.index(p), v] *
                                   int(h_coef[class_list[v]][0]*height_list[v])
                                   - xay2[platforms_angled_array.index(p), v] *
                                   int(h_coef[class_list[v]][1]*height_list[v])
                                   - xay3[platforms_angled_array.index(p), v] *
                                   int(h_coef[class_list[v]][2]*height_list[v])
                                   - xay4[platforms_angled_array.index(p), v] *
                                   int(h_coef[class_list[v]][3]*height_list[v])
                                   for p in self.intersectset(platforms_angled_array, platforms_height_array_truck[h])
                                   for v in vecs)\
                        + pulp.lpSum(x[p, v] * height_list[v]
                                     for p in self.diffset(platforms_height_array_truck[h], platforms_angled_array)
                                     for v in vecs) \
                        <= hmax_truck[h]
            # Trailer
            for h in plats_h2:
                prob += pulp.lpSum(x[p, v] * height_list[v]
                                   - xay1[platforms_angled_array.index(p), v] *
                                   int(h_coef[class_list[v]][0]*height_list[v])
                                   - xay2[platforms_angled_array.index(p), v] *
                                   int(h_coef[class_list[v]][1]*height_list[v])
                                   - xay3[platforms_angled_array.index(p), v] *
                                   int(h_coef[class_list[v]][2]*height_list[v])
                                   - xay4[platforms_angled_array.index(p), v] *
                                   int(h_coef[class_list[v]][3]*height_list[v])
                                   for p in self.intersectset(platforms_angled_array, platforms_height_array_trailer[h])
                                   for v in vecs)\
                        + pulp.lpSum(x[p, v] * height_list[v]
                                     for p in self.diffset(platforms_height_array_trailer[h], platforms_angled_array)
                                     for v in vecs) \
                        <= hmax_trailer[h]

        # Set the problem sense and name
        problem_instance = prob.to_dict()
        self.application = problem_instance
        return self.application

    def validate(self, solution:any) -> (bool, float):
        """
        Checks if the solution is a valid solution
:
        :param solution: Proposed solution
        :type solution: any
        :return: bool value if solution is valid and the time it took to validate the solution
        :rtype: tuple(bool, float)
        """

        start = start_time_measurement()
        status = solution["status"]
        if status == 'Optimal':
            return True, end_time_measurement(start)
        else:
            return False, end_time_measurement(start)

    def get_solution_quality_unit(self) -> str:
        return "Number of loaded vehicles"

    def evaluate(self, solution: any) -> (float, float):
        """
        Checks how good the solution is

        :param solution: Provided solution
        :type solution: any
        :return: Evaluation and the time it took to create it
        :rtype: tuple(any, float)

        """
        start = start_time_measurement()
        objective_value = solution["obj_value"]
        logging.info("Loading successful!")
        logging.info(str(objective_value)+" cars will fit on the auto carrier.")
        variables = solution["variables"]
        assignments = []
        # Check which decision variables are equal to 1
        for key in variables:
            if variables[key] > 0:
                assignments.append(key)
        logging.info("vehicle to platform assignments (platform, vehicle): "+ str(assignments))
        return objective_value, end_time_measurement(start)

    def save(self, path: str, iter_count: int) -> None:
        # Convert our problem instance from Dict to an LP problem and then to json
        _, problem_instance = pulp.LpProblem.from_dict(self.application)
        # Save problem instance to json
        problem_instance.to_json(f"{path}/ACL_instance.json")
