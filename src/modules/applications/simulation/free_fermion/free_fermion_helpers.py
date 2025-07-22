import numpy as np
import logging
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from scipy.stats import chi2
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

logger = logging.getLogger()


def coordinates(x: int, y: int, lx: int, ly: int) -> int:
    """Coordinate of site (x, y)"""
    return (x % lx) + (y % ly) * lx


def create_couplings(lx, ly) -> list[list[int]]:
    """Creates the list of couplings."""

    l_tot = lx * ly
    couplings_e: list[list[int]] = []
    for j in range(l_tot):
        # runs through all the lattice sites. j//Lx is the vertical coordinate
        # (<Ly) and j%Lx the horizontal coordinate (<Lx)
        j_test = (j // lx) % 2 == j % 2
        # k for +1 in horizontal coordinate
        k_horizontal = (j // lx) * lx + ((j % lx) + 1) % lx
        v_horizontal = j // 2 if j_test else (j // 2 - lx // 2) % (l_tot // 2)
        w_horizontal = 1
        couplings_e.append([j, k_horizontal, l_tot + v_horizontal, w_horizontal])
        # k +1 in vertical coordinate
        k_vertical = (((j // lx) + 1) % ly) * lx + j % lx
        v_vertical = j // 2 if j_test else ((j // lx) * lx + ((j % lx) - 1) % lx) // 2
        w_vertical = 0
        couplings_e.append([j, k_vertical, l_tot + v_vertical, w_vertical])

    return couplings_e


class FreeFermionSolver:
    def __init__(self, j, k, s, lx, ly, bound_hor, bound_vert, n):
        self.j = j
        self.k = k
        self.s = s
        self.lx = lx
        self.ly = ly
        self.n = n
        self.sig = 1
        if j < k:
            abs_jk = abs(j - k)
            if bound_hor == 1 and abs_jk == lx - 1:
                self.sig *= -1
            if bound_vert == 1 and abs_jk >= lx:
                self.sig *= -1

    def dc(self, c: np.array, d: np.array):  # derivative of evolution of c=<c_i^\dagger c_j>
        deriv: np.array = np.zeros((self.n, self.n)) * 1j
        deriv[self.j, :] += 1j * (c[self.k, :] - self.s * d[self.k, :]) * self.sig
        deriv[self.k, :] += 1j * (c[self.j, :] + self.s * d[self.j, :]) * self.sig
        deriv[:, self.j] += 1j * (-c[:, self.k] + self.s * np.conj(d[self.k, :])) * self.sig
        deriv[:, self.k] += 1j * (-c[:, self.j] - self.s * np.conj(d[self.j, :])) * self.sig
        return deriv

    def dd(self, c: np.array, d: np.array) -> np.array:  # derivative of evolution of D=<c_i c_j>
        deriv: np.array = np.zeros((self.n, self.n)) * 1j
        deriv[self.j, :] += 1j * (-d[self.k, :] + self.s * c[self.k, :]) * self.sig
        deriv[self.k, :] += 1j * (-d[self.j, :] - self.s * c[self.j, :]) * self.sig
        deriv[:, self.j] += 1j * (-d[:, self.k] - self.s * c[self.k, :]) * self.sig
        deriv[self.k, self.j] += 1j * self.s * self.sig
        deriv[:, self.k] += 1j * (-d[:, self.j] + self.s * c[self.j, :]) * self.sig
        deriv[self.j, self.k] += -1j * self.s * self.sig
        return deriv

    def diff(self, t: float, cvec: np.array):  # function to call for the differential equation
        """Function to pass to solve_ivp"""
        call: np.array = cvec.reshape((2 * self.n, self.n))
        return (np.concatenate((self.dc(call[:self.n], call[self.n:]),
                                self.dd(call[:self.n], call[self.n:])))).reshape(2 * self.n ** 2)


def exact_values_and_variance(n_trot: int, dt: float, lx: int, ly: int):
    l = lx * ly
    n = 2 * l
    # index 0: number of steps; index 1: expectation value of imbalance; index
    # 2: expectation value of square of imbalance
    res: np.array = np.zeros((n_trot, 3))

    for bb in [[0, 0], [0, 1], [1, 0], [1, 1]]:  # loops over the 4 boundary conditions
        boundary_vert = bb[0]
        boundary_hor = bb[1]

        c: np.array = np.zeros((n, n)) * 1j
        d: np.array = np.zeros((n, n)) * 1j

        for j in range(l // 2):  # initialize in the product state
            c[j, j] = 1

        res[0, 1] += -1
        res[0, 2] += 1

        # applies horizontal XX, horizontal YY, vertical XX, vertical YY
        order = [[1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]]
        f: list = [1 / l] * (l // 2) + [-1 / l] * (l // 2)  # observable in Eq10-11
        for t in range(n_trot - 1):  # loop over Trotter steps
            for o in order:  # loop over the 4 edges configurations
                for k in range(ly):  # loop over vertical coordinate
                    for j in range(lx):  # loop over horizontal coordinate
                        jcur = coordinates(j, k, lx, ly)
                        kcur = coordinates(j + o[0], k + o[1], lx, ly)
                        scur = o[2]
                        solver = FreeFermionSolver(jcur, kcur, scur, lx, ly, boundary_hor, boundary_vert, n)
                        cc: np.array = (solve_ivp(solver.diff, [0, dt / 2], np.concatenate((c, d)).reshape(2 * n ** 2),
                                                  atol=1e-9, rtol=1e-9).y)[:, -1].reshape((2 * n, n))
                        c = cc[:n]
                        d = cc[n:]
            a: float = np.sum([f[j] * (1 - 2 * c[j, j]) for j in range(l)])
            var: float = 4 * np.sum([f[i] * f[j] * c[i, i] * c[j, j] for i in range(l) for j in range(l)])
            var += -4 * np.sum([f[i] * f[j] * c[i, j] * c[j, i] for i in range(l) for j in range(l)])
            var += 4 * np.sum([f[i] ** 2 * c[i, i] for i in range(l)])
            var += 4 * np.sum([f[i] * f[j] * abs(d[i, j]) ** 2 for i in range(l) for j in range(l)])
            res[t + 1, 0] += t + 1
            res[t + 1, 1] += np.real(a)
            res[t + 1, 2] += np.real(var)

    res = res / 4
    res[:, 2] = np.sqrt(res[:, 2] - res[:, 1] ** 2)  # standard deviation per shot
    return res


def state_preparation(u, lx: int, ly: int):
    l = lx * ly

    for j in range(l // 2 - 2 * (lx // 2)):  # toric code ground state preparation on the ancillas
        if (j // (lx // 2)) % 2 == 0:
            k = l // 2 - 2 * (lx // 2) - j - 1
            f1 = l + (k % (l // 2))
            f2 = l + ((k // (lx // 2)) + 1) * lx // 2 + (k % (lx // 2))
            f3 = l + ((k // (lx // 2)) + 1) * lx // 2 + (((k % (lx // 2)) + 1) % (lx // 2))
            f4 = l + ((k // (lx // 2)) + 2) * lx // 2 + (k % (lx // 2))
            u.h(f1)
            u.cx(f1, f2)
            u.cx(f1, f3)
            u.cx(f1, f4)
    for j in range(lx // 2 - 1):
        k = lx // 2 - 2 - j
        f1 = l + k
        f2 = l + ((k // (lx // 2)) + 1) * lx // 2 + (k % (lx // 2))
        f3 = l + ((k // (lx // 2)) + 0) * lx // 2 + (((k % (lx // 2)) + 1) % (lx // 2))
        f4 = l + (((k // (lx // 2)) - 1) % ly) * lx // 2 + (k % (lx // 2))
        u.h(f1)
        u.cx(f1, f2)
        u.cx(f1, f3)
        u.cx(f1, f4)
    for j in range(l // 2):  # change of basis of the toric code
        if (j // (lx // 2)) % 2 == 1:
            u.sdg(l + j)
            u.h(l + j)
        if (j // (lx // 2)) % 2 == 0:
            u.s(l + j)
            u.h(l + j)
            u.s(l + j)


def trotter_step(u, dt: float, lx: int, e: list):
    for ind2 in [1, 0]:
        # ind2=0 does vertical edges, and ind2=1 does horizontal edges.
        # Implements the difference horizontal/vertical in Eq7
        for ind in [0, 1]:
            # ind=0 implements XX on even rows/columns and YY on odd rows/columns.
            # ind=1 implements the other way around. Implements the difference 1/2 in Eq7
            for c in e:  # loops over all edges
                if c[3] == ind2:  # selects horizontal or vertical edges
                    sig: int = 1
                    if c[3] == 0 and (c[0] % 2) == 0:
                        # implements the -1 in the fermionic encoding that occurs only for even columns
                        sig *= -1
                    # the sequence of H's and Sdg's  are conjugating the central ZZZ rotation
                    # into some rotations like XXY
                    if (c[3] == 0 and c[0] % 2 == 1 - ind) or (c[3] == 1 and c[0] // lx) % 2 == 1 - ind:
                        # if c is a column (line), apply Y only when the parity of the column (line) is 1-ind.
                        u.sdg(c[0])
                    u.h(c[0])
                    if (c[3] == 0 and c[0] % 2 == 1 - ind) or (c[3] == 1 and (c[0] // lx) % 2 == 1 - ind):  # same
                        u.sdg(c[1])
                    u.h(c[1])
                    if c[3] == 1:  # apply Y on the ancilla only for horizontal edges
                        u.sdg(c[2])
                    u.h(c[2])

                    u.cx(c[0], c[1])  # Pauli gadget that implements a ZZZ rotation on qubits c[0], c[1], c[2]
                    u.rzz(-2 * dt * sig / 2, c[1], c[2])
                    u.cx(c[0], c[1])

                    u.h(c[2])
                    if c[3] == 1:
                        u.s(c[2])
                    u.h(c[1])
                    if (c[3] == 0 and c[0] % 2 == 1 - ind) or (c[3] == 1 and (c[0] // lx) % 2 == 1 - ind):
                        u.s(c[1])
                    u.h(c[0])
                    if (c[3] == 0 and c[0] % 2 == 1 - ind) or (c[3] == 1 and (c[0] // lx) % 2 == 1 - ind):
                        u.s(c[0])


def create_circuit(lx: int, ly: int, dt: float, n_trot: int) -> np.array:
    logger.info(f"Creating simulation circuit for {n_trot} Trotter steps")
    e = create_couplings(lx, ly)
    u = QuantumCircuit(lx * ly * 3 // 2)
    state_preparation(u, lx, ly)
    for j in range(lx * ly // 2):
        u.x(j)  # applies X where there is a fermion. The state has to satisfy the constraint that
        # there is an even number of fermions per face
    for t in range(n_trot):
        trotter_step(u, dt, lx, e)
    u.measure_all()
    return u


def extract_simulation_results(dt: float, lx: int, ly: int, n_shots: int, counts_per_circuit: list[dict[str, int]]) \
        -> list[tuple[float, float, float]]:
    """Returns the simulation results.

    For every time step returns the time, expectation value and standard deviation as a tuple for that step.
    """
    l_tot = lx * ly
    results = []
    for n, counts in enumerate(counts_per_circuit):
        res: float = 0
        var: float = 0
        for s in counts:
            a: float = 0
            for j in range(l_tot // 2):
                if s[l_tot * 3 // 2 - 1 - j] == '1':
                    a += -1 / l_tot
                else:
                    a += 1 / l_tot
                if s[l_tot * 3 // 2 - 1 - j - l_tot // 2] == '1':
                    a += 1 / l_tot
                else:
                    a += -1 / l_tot
            res += a * counts[s]
            var += a ** 2 * counts[s]
        res = res / n_shots
        var = var / n_shots
        results.append((dt * n, res, np.sqrt(var - res ** 2) / np.sqrt(n_shots)))
    return results


def computes_score_values(delta: np.array, std_exp: np.array, std: np.array, l: int) -> tuple[int, int, int]:
    """ Computes score values.

        Returns the score in terms of
         1.) Number of gates
         2.) Number of shots
         3.) Number of Trotter steps
    """
    n: int = len(delta)
    delta_corrected = np.zeros(n)
    for j in range(n):
        if std[j] > 0:
            delta_corrected[j] = max(abs(delta[j]), std_exp[j]) / std[j]
            # redefines delta as the maximum between the experimental standard deviation and the measured value,
            # normalized by the theoretical standard deviation
    rewards: float = float(delta_corrected[0]) ** 2
    opt: int = 0
    for j in range(1, n):  # looks for the time point opt with maximal reward
        temp: float = float(delta_corrected[j]) ** 2 / (j + 1)
        if temp > rewards:
            rewards = temp
            opt = j

    def ff(x): return chi2.cdf(delta_corrected[opt] ** 2 * x, df=n - 1) - 0.997
    x: float = fsolve(ff, n / delta_corrected[opt] ** 2)[0]
    # looks for x such that chi2.cdf(delta[opt]**2*x*L,df=1)=0.997

    return 6 * int(np.floor(x) + 1) * (opt + 1) * l, int(np.floor(x) + 1), int(np.floor(x) + 1) * (opt + 1)
