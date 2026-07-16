"""
triples_sigfast.hep.mc
----------------------
Monte Carlo tools for high-energy physics.
Includes a Numba-compiled RAMBO (Random Momenta Beautifully Organized)
phase space generator for N-body decays/scattering.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

__all__ = [
    "generate_phase_space",
    "rambo",
]


@njit(fastmath=True, cache=True, parallel=True)
def generate_phase_space(
    n_events: int, e_cm: float, masses: np.ndarray
) -> tuple[np.ndarray, ...]:
    """
    Generate n-body flat phase space events using the RAMBO algorithm.

    Parameters
    ----------
    n_events : int
        Number of events to generate.
    e_cm : float
        Center of mass energy in GeV.
    masses : np.ndarray
        1D array of particle masses in GeV.

    Returns
    -------
    tuple of np.ndarray
        Tuple of `n_particles` arrays, each of shape `(n_events, 4)`.
        Each array corresponds to the 4-momenta [E, px, py, pz] of a particle across all events.
    """
    n_particles = len(masses)

    # We allocate a 3D array internally for convenience, then unpack it to a tuple of 2D arrays.
    out = np.empty((n_particles, n_events, 4), dtype=np.float64)

    # Validate masses
    total_mass = 0.0
    for m in masses:
        total_mass += m

    if total_mass > e_cm:
        # Kinematically forbidden
        for p_idx in range(n_particles):
            for ev in prange(n_events):
                for j in range(4):
                    out[p_idx, ev, j] = 0.0
    else:
        for ev in prange(n_events):
            # 1. Generate massless momenta
            Q0 = 0.0
            Q1 = 0.0
            Q2 = 0.0
            Q3 = 0.0

            q = np.empty((n_particles, 4), dtype=np.float64)
            for i in range(n_particles):
                r1 = np.random.rand()
                r2 = np.random.rand()
                r3 = np.random.rand()
                r4 = np.random.rand()

                # To avoid log(0)
                if r1 == 0.0:
                    r1 = 1e-10
                if r2 == 0.0:
                    r2 = 1e-10

                c = 2.0 * r3 - 1.0
                s = math.sqrt(max(1.0 - c * c, 0.0))
                phi = 2.0 * math.pi * r4

                e = -math.log(r1 * r2)
                q[i, 0] = e
                q[i, 1] = e * s * math.cos(phi)
                q[i, 2] = e * s * math.sin(phi)
                q[i, 3] = e * c

                Q0 += q[i, 0]
                Q1 += q[i, 1]
                Q2 += q[i, 2]
                Q3 += q[i, 3]

            M_Q2 = Q0 * Q0 - Q1 * Q1 - Q2 * Q2 - Q3 * Q3
            if M_Q2 <= 0.0:
                for i in range(n_particles):
                    out[i, ev, 0] = 0.0
                    out[i, ev, 1] = 0.0
                    out[i, ev, 2] = 0.0
                    out[i, ev, 3] = 0.0
                continue
            M_Q = math.sqrt(M_Q2)

            b0 = -Q1 / M_Q
            b1 = -Q2 / M_Q
            b2 = -Q3 / M_Q
            gamma = Q0 / M_Q
            a = 1.0 / (1.0 + gamma)
            x = e_cm / M_Q

            p = np.empty((n_particles, 4), dtype=np.float64)
            for i in range(n_particles):
                b_dot_q = b0 * q[i, 1] + b1 * q[i, 2] + b2 * q[i, 3]
                p[i, 0] = x * (gamma * q[i, 0] + b_dot_q)
                p[i, 1] = x * (q[i, 1] + b0 * q[i, 0] + a * b_dot_q * b0)
                p[i, 2] = x * (q[i, 2] + b1 * q[i, 0] + a * b_dot_q * b1)
                p[i, 3] = x * (q[i, 3] + b2 * q[i, 0] + a * b_dot_q * b2)

            # 2. Check if massive
            if total_mass == 0.0:
                for i in range(n_particles):
                    out[i, ev, 0] = p[i, 0]
                    out[i, ev, 1] = p[i, 1]
                    out[i, ev, 2] = p[i, 2]
                    out[i, ev, 3] = p[i, 3]
                continue

            # 3. Massive adjustment via Newton-Raphson
            xi = math.sqrt(max(1.0 - (total_mass / e_cm) ** 2, 0.0))
            for _ in range(20):
                f = -e_cm
                df = 0.0
                for i in range(n_particles):
                    p2 = p[i, 1] * p[i, 1] + p[i, 2] * p[i, 2] + p[i, 3] * p[i, 3]
                    E_i = math.sqrt(masses[i] * masses[i] + xi * xi * p2)
                    f += E_i
                    if E_i > 0.0:
                        df += (xi * p2) / E_i
                if abs(f) < 1e-9 or df == 0.0:
                    break
                xi = xi - f / df

            # 4. Final momenta
            for i in range(n_particles):
                p2 = p[i, 1] * p[i, 1] + p[i, 2] * p[i, 2] + p[i, 3] * p[i, 3]
                out[i, ev, 0] = math.sqrt(masses[i] * masses[i] + xi * xi * p2)
                out[i, ev, 1] = xi * p[i, 1]
                out[i, ev, 2] = xi * p[i, 2]
                out[i, ev, 3] = xi * p[i, 3]

    # Return as tuple of 2D arrays (N x 4)
    # Numba doesn't support dynamically sized tuples, so we return a list and wrap it
    return out


def rambo(
    n_events: int, e_cm: float, masses: np.ndarray | list[float]
) -> tuple[np.ndarray, ...]:
    """
    Generate n-body flat phase space events using the RAMBO algorithm.
    Wrapper around the JIT-compiled generator.

    Parameters
    ----------
    n_events : int
        Number of events.
    e_cm : float
        Center of mass energy in GeV.
    masses : np.ndarray or list of floats
        Masses of the final state particles.

    Returns
    -------
    tuple of np.ndarray
        Tuple of `n_particles` arrays, each of shape `(n_events, 4)`.
    """
    m_arr = np.array(masses, dtype=np.float64)
    out_3d = generate_phase_space(n_events, float(e_cm), m_arr)
    return tuple(out_3d[i] for i in range(len(masses)))
