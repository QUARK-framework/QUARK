import numpy as np
import logging
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from scipy.stats import chi2
from scipy.optimize import fsolve

logger = logging.getLogger()

def createCouplings(Lx, Ly) -> tuple:  # returns a list of couplings and a list of faces
    L = Lx * Ly
    # creation of the list of couplings
    E: list = []
    for j in range(
            # runs through all the lattice sites. j//Lx is the vertical coordinate
            # (<Ly) and j%Lx the horizontal coordinate (<Lx)
            L):
        for k in [(j // Lx) * Lx + ((j % Lx) + 1) % Lx, (((
                j // Lx) + 1) % Ly) * Lx + j % Lx]:  # runs through the two neighbours of j, one with +1 for horizontal coordinate, and one with +1 for vertical coordinate
            if (j // Lx == k // Lx and (j // Lx) % 2 == j % 2):  # horizontal edge on the left of the ancilla
                v: int = j // 2  # computes the ancilla number (minus L) that touches the edge (j,k)
                w: int = 1  # w=1 means horizontal edge and w=0 means vertical edge
            if (j // Lx == k // Lx and (j // Lx) % 2 != j % 2):  # horizontal edge on the right of the ancilla
                v: int = (j // 2 - Lx // 2) % (L // 2)
                w: int = 1

            if (j % Lx == k % Lx and (j // Lx) % 2 == j % 2):  # vertical edge below the ancilla
                v: int = j // 2
                w: int = 0
            if (j % Lx == k % Lx and (j // Lx) % 2 != j % 2):  # vertical edge above the ancilla
                v: int = ((j // Lx) * Lx + ((j % Lx) - 1) % Lx) // 2
                w: int = 0

            E.append([j, k, L + v, w])

    # creation of the list of faces
    F: list = []
    for f in range(L // 2):  # runs through all the faces of the lattice without ancillas
        f1: int = L + f  # ancilla on left
        f2: int = L + (f // (Lx // 2)) * (Lx // 2) + ((f + 1) % (Lx // 2))  # ancilla on right
        f3: int = L + ((f // (Lx // 2) + 1) % Ly) * (Lx // 2) + (
            (f + ((f // (Lx // 2)) % 2)) % (Lx // 2))  # ancilla above
        f4: int = L + ((f // (Lx // 2) - 1) % Ly) * (Lx // 2) + (
            (f + ((f // (Lx // 2)) % 2)) % (Lx // 2))  # ancilla below
        f5: int = (((2 * f) % Lx) + 1 + (((2 * f) // Lx) % 2)) % Lx + ((2 * f) // Lx) * Lx  # site bottom left
        f6: int = (((2 * f) % Lx) + 2 + (((2 * f) // Lx) % 2)) % Lx + ((2 * f) // Lx) * Lx  # site bottom right
        f7: int = (((f5 + Lx) // Lx) % Ly) * Lx + (f5 % Lx)  # site top left
        f8: int = (((f6 + Lx) // Lx) % Ly) * Lx + (f6 % Lx)  # site top right
        F.append([f1, f2, f3, f4, f5, f6, f7, f8])

    return E, F


def get_pauli_string(N: int,
                     # takes as argument a list of the form [['Z',1]] and returns a Pauli
                     # string IZIII compatible with qiskit
                     lst: list) -> str:
    s: str = ''
    for j in range(N):
        flag: int = 0
        for op in lst:
            if op[1] == j:
                s = s + op[0]
                flag = 1
                break
        if flag == 0:
            s = s + 'I'
    return s


def exactTrotter(dt: float, n: int, decalx: float, decaly: float, Lx: int, Ly: int) -> float:
    # returns exact value after n Trotter steps of size dt. decalx/decaly=0 for periodic boundary
    # conditions in the horizontal/vertical direction, and =0.5 for anti-periodic boundary conditions,
    # they are the phix/phiy in Eq12.
    # Values from the quantum circuits are (exactTrotter(dt,n,0,0.5)+exactTrotter(dt,n,0.5,0))/2
    L = Lx * Ly
    def coseps(x, y): return 1 - 2 * np.sin(dt) ** 2 * (np.cos(x) + np.cos(y)) ** 2 + 4 * np.sin(dt) ** 4 * np.cos(
        x) * np.cos(y) * (1 + np.cos(x + y))  # cos of Eq17 in Appendix

    def sineps(x, y): return np.sqrt(1 - coseps(x, y) ** 2)  # sin of Eq17 in Appendix
    def eit(x, y): return coseps(x, y) + 1j * sineps(x, y)  # exp(i\eps) with \eps=Eq17 in Appendix

    def alpha1(x, y): return -1j * np.sin(2 * dt) * (np.cos(x) + np.cos(y)) + 2 * 1j * np.sin(2 * dt) * np.sin(
        dt) ** 2 * np.cos(x) * np.cos(y) * (np.cos(x) + np.cos(y))  # alpha in Eq18 in Appendix, for n=1

    def beta1(x, y): return 1j * (1 - 2 * np.sin(dt) ** 2 * np.cos(x) ** 2 - 1j * np.sin(2 * dt) * np.cos(x)) * np.sin(
        dt) ** 2 * np.sin(2 * y) + 1j * (1 - 2 * np.sin(dt) ** 2 * np.cos(y) ** 2 + 1j * np.sin(2 * dt) * np.cos(
            y)) * np.sin(dt) ** 2 * np.sin(2 * x)  # beta in Eq 18 in Appendix, for n=1

    def aa(x, y): return (alpha1(x, y) + 1j * sineps(x, y))
    def bb(x, y): return beta1(x, y)

    def alphan(x, y): return eit(x, y) ** (-n) + aa(x, y) * np.sum([eit(x, y) ** (n - 1 - 2 * m) for m in range(n)],
                                                                   axis=0)  # alpha in Eq18 in Appendix
    def betan(x, y): return bb(x, y) * np.sum([eit(x, y) ** (n - 1 - 2 * m) for m in range(n)],
                                              axis=0)  # beta in Eq18 in Appendix

    def momentx(x): return 2 * np.pi * (x + decalx) / Lx  # momenta in K in Eq12
    def momenty(y): return 2 * np.pi * (y + decaly) / Ly

    f: np.array = np.array([[-1 + 0 * 1j if (i < Ly // 2) else 1 for i in range(Ly)] for j in range(Lx)]) / (
        L // 2)  # observable in Eq9-10

    x: np.array = np.tensordot(np.array([j for j in range(Lx)]), np.array([1 for j in range(Ly)]), 0)
    y: np.array = np.tensordot(np.array([1 for j in range(Lx)]), np.array([j for j in range(Ly)]), 0)
    exp: np.array = np.exp(2 * np.pi * 1j * (np.tensordot(x, x, 0) / Lx + np.tensordot(y, y, 0) / Ly))
    ftilde: np.array = np.sum(np.tensordot(f, np.ones((Lx, Ly)), 0) * exp,
                              axis=(0, 1)) / L  # computation of f^hat Fourier transform of f

    nn: np.array = np.array([[1 + 0 * 1j if (i < Ly // 2) else 0 for i in range(Ly)] for j in
                             range(Lx)])  # initial state of the system qubits in Eq3
    ntilde: np.array = np.sum(np.tensordot(nn, np.ones((Lx, Ly)), 0) * np.conj(exp),
                              axis=(0, 1)) / L  # computation of n^hat Fourier transform of n

    kx: np.array = np.tensordot(np.array([j + 0 * 1j for j in range(Lx)]), np.ones((Ly, Lx, Ly)),
                                0)  # k and k-q indices in the sum Eq21 in the appendix
    ky: np.array = np.tensordot(np.ones(Lx),
                                np.tensordot(np.array([j + 0 * 1j for j in range(Ly)]), np.ones((Lx, Ly)), 0), 0)
    qx: np.array = np.tensordot(np.ones((Lx, Ly)),
                                np.tensordot(np.array([j + 0 * 1j for j in range(Lx)]), np.ones(Ly), 0), 0)
    qy: np.array = np.tensordot(np.ones((Lx, Ly, Lx)), np.array([j + 0 * 1j for j in range(Ly)]), 0)

    total: np.array = np.tensordot(np.ones((Lx, Ly)), ftilde * ntilde,
                                   0)  # product of ftilde and ntilde in Eq21 in Appendix
    total2: np.array = np.tensordot(np.ones((Lx, Ly)), ftilde,
                                    0)  # ftilde in same format as above. To implement second line ofEq21 in Appendix

    aterm: np.array = alphan(momentx(kx + qx), momenty(ky + qy))  # product of alpha and alpha* in Eq21 in Appendix
    aterm *= np.conj(alphan(momentx(kx), momenty(ky)))

    bterm: np.array = betan(momentx(kx + qx), momenty(ky + qy))  # product of beta and beta* in Eq21 in Appendix
    bterm *= np.conj(betan(momentx(kx), momenty(ky)))

    ident: np.array = np.array(
        [[[[1 if (i1 == 0 and i2 == 0) else 0 for i1 in range(Ly)] for i2 in range(Lx)] for i3 in range(Ly)] for i4 in
         range(Lx)])  # identity term

    res: complex = np.sum(total * (aterm - bterm) + total2 * bterm * ident)  # Eq21 in Appendix

    return np.real(res)


def state_preparation(U, Lx: int, Ly: int):
    L = Lx * Ly

    for j in range(L // 2 - 2 * (Lx // 2)):  # toric code ground state preparation on the ancillas
        if ((j // (Lx // 2)) % 2 == 0):
            k = L // 2 - 2 * (Lx // 2) - j - 1
            f1 = L + (k % (L // 2))
            f2 = L + ((k // (Lx // 2)) + 1) * Lx // 2 + (k % (Lx // 2))
            f3 = L + ((k // (Lx // 2)) + 1) * Lx // 2 + (((k % (Lx // 2)) + 1) % (Lx // 2))
            f4 = L + ((k // (Lx // 2)) + 2) * Lx // 2 + (k % (Lx // 2))
            U.h(f1)
            U.cx(f1, f2)
            U.cx(f1, f3)
            U.cx(f1, f4)
    for j in range(Lx // 2 - 1):
        k = Lx // 2 - 2 - j
        f1 = L + k
        f2 = L + ((k // (Lx // 2)) + 1) * Lx // 2 + (k % (Lx // 2))
        f3 = L + ((k // (Lx // 2)) + 0) * Lx // 2 + (((k % (Lx // 2)) + 1) % (Lx // 2))
        f4 = L + (((k // (Lx // 2)) - 1) % Ly) * Lx // 2 + (k % (Lx // 2))
        U.h(f1)
        U.cx(f1, f2)
        U.cx(f1, f3)
        U.cx(f1, f4)
    for j in range(L // 2):  # change of basis of the toric code
        if ((j // (Lx // 2)) % 2 == 1):
            U.sdg(L + j)
            U.h(L + j)
        if ((j // (Lx // 2)) % 2 == 0):
            U.s(L + j)
            U.h(L + j)
            U.s(L + j)


def trotter_step(U, dt: float, Lx: int, E: list):
    for ind2 in [1,
                 0]:  # ind2=0 does vertical edges, and ind2=1 does horizontal edges. Implements the difference horizontal/vertical in Eq7
        for ind in [0,
                    1]:  # ind=0 implements XX on even rows/columns and YY on odd rows/columns. ind=1 implements the other way around. Implements the difference 1/2 in Eq7
            for c in E:  # loops over all edges
                if (c[3] == ind2):  # selects horizontal or vertical edges
                    sig: int = 1
                    if (c[3] == 0 and (c[
                            0] % 2) == 0):  # implements the -1 in the fermionic encoding that occurs only for even columns
                        sig *= -1
                    # the sequence of H's and Sdg's  are conjugating the central ZZZ rotation
                    # into some rotations like XXY
                    if ((c[3] == 0 and c[0] % 2 == 1 - ind) or (c[3] == 1 and (c[
                            # if c is a column (line), apply Y only when the parity of the column (line) is 1-ind.
                            0] // Lx) % 2 == 1 - ind)):
                        U.sdg(c[0])
                    U.h(c[0])
                    if ((c[3] == 0 and c[0] % 2 == 1 - ind) or (c[3] == 1 and (c[0] // Lx) % 2 == 1 - ind)):  # same
                        U.sdg(c[1])
                    U.h(c[1])
                    if (c[3] == 1):  # apply Y on the ancilla only for horizontal edges
                        U.sdg(c[2])
                    U.h(c[2])

                    U.cx(c[0], c[1])  # Pauli gadget that implements a ZZZ rotation on qubits c[0], c[1], c[2]
                    U.rzz(-2 * dt * sig / 2, c[1], c[2])
                    U.cx(c[0], c[1])

                    U.h(c[2])
                    if (c[3] == 1):
                        U.s(c[2])
                    U.h(c[1])
                    if ((c[3] == 0 and c[0] % 2 == 1 - ind) or (c[3] == 1 and (c[0] // Lx) % 2 == 1 - ind)):
                        U.s(c[1])
                    U.h(c[0])
                    if ((c[3] == 0 and c[0] % 2 == 1 - ind) or (c[3] == 1 and (c[0] // Lx) % 2 == 1 - ind)):
                        U.s(c[0])


def create_circuit(Lx: int, Ly: int, dt: float, Ntrot: int) -> np.array:
    logger.info(f"Creating simulation circuit for {Ntrot} trotter steps")
    E, F = createCouplings(Lx, Ly)
    U = QuantumCircuit(Lx * Ly * 3 // 2)
    state_preparation(U, Lx, Ly)
    for j in range(Lx * Ly // 2):
        U.x(j)  # applies X where there is a fermion. The state has to satisfy the constraint that
        # there is an even number of fermions per face
    for t in range(Ntrot):
        trotter_step(U, dt, Lx, E)
    U.measure_all()
    return U


def exact_values(Ntrot: int, dt: float, Lx: int, Ly: int) -> list[float]:
    exactList: list[float] = []
    for u in range(Ntrot):
        logger.info(f"Calculating exact value for {u} trotter steps")
        exactList.append([
            u,
            (exactTrotter(dt, u, 0, 0., Lx, Ly)
             + exactTrotter(dt, u, 0.5, 0.5, Lx, Ly)
             + exactTrotter(dt, u, 0,0.5, Lx, Ly)
             + exactTrotter(dt, u, 0.5, 0, Lx, Ly)) / 4]
        )
    return exactList


def score_minimal(delta: np.array, L: int) -> float:  # returns the score, given differences delta=measured-exact
    n: int = len(delta)
    rewards: float = delta[0] ** 2
    opt: int = 0
    for j in range(1, n):  # looks for the time point opt with maximal reward
        temp: float = delta[j] ** 2 / (j + 1)
        if (temp > rewards):
            rewards = temp
            opt = j

    def ff(x): return chi2.cdf(delta[opt] ** 2 * x * L, df=1) - 0.997
    x: float = fsolve(ff, n / delta[opt] ** 2 / L)[0]  # looks for x such that chi2.cdf(delta[opt]**2*x*L,df=1)=0.997
    return 6 * x * (opt + 1) * L


def score_minimal_mean(delta: np.array, var: np.array,
                       L: int) -> tuple:  # returns the score+variance,given differences delta and associated standard deviations
    res: float = 0
    var2: float = 0
    for j in range(1000):  # loops over 1000 (this number is arbitrary) samples to estimate the variance and mean
        toAdd: float = np.log(score_minimal(delta + var * np.random.normal(size=len(delta)), L)) / np.log(10)
        res += toAdd
        var2 += toAdd ** 2
    res = res / 1000
    var2 = np.sqrt(var2 / 1000 - res ** 2)
    return res, var2

def extract_simulation_results(
        lx: int, ly: int, n_shots: int,
        counts_per_circuit: list[dict[str, int]]
) -> list[tuple[int, float, float]]:
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
        results.append((n, res, np.sqrt(var - res ** 2) / np.sqrt(n_shots)))
    return results
