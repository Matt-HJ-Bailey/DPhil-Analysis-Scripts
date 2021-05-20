#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:06:09 2021

@author: matthew-bailey
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from collections import defaultdict

K_BOND = 0.16
R_EQM = 50.0
BURD_K_BOND = 1.5490e6 / 6
R_STIFF = 1.347 * R_EQM
SIGMA = 50.0
EPSILON = 4 * 4.142


def quartquad_force(
    current_r: float, r_eqm: float, r_stiff: float, quadratic_k: float, quartic_k: float
) -> float:
    """
    The force corresponding to a quartic-quadratic potential.

    Parameters
    ----------
    current_r
        The separation to evaluate the force at
    r_eqm
        The equilibrium separation
    r_stiff
        A length scale over which the potential transitions from quadratic to quartic
    quadratic_k
        Force constant for quadratic term in Nm^-2
    quartic_k
        Force constant for quartic term in Nm^-4
    Returns
    -------
    force
        The force at this separation
    """
    erf_scale = (current_r - r_eqm) ** 2 / (r_stiff) ** 2
    erf_term = scipy.special.erf(erf_scale)
    erfc_term = 1.0 - erf_term
    exp_sq = np.exp(-(erf_scale ** 2))

    linear_term = 2 * quadratic_k * (current_r - r_eqm) * erfc_term
    cubic_exp = (
        -4 * quadratic_k * (current_r - r_eqm) * erf_scale * exp_sq / np.sqrt(np.pi)
    )

    cubic_term = 4 * quartic_k * (current_r - r_eqm) ** 3 * erf_term
    quintic_exp = (
        4 * quartic_k * erf_scale * (current_r - r_eqm) ** 3 * exp_sq / np.sqrt(np.pi)
    )

    return -(linear_term + cubic_exp + cubic_term + quintic_exp)


def quartquad(
    current_r: float, r_eqm: float, r_stiff: float, quadratic_k: float, quartic_k: float
) -> float:
    """
    The energy corresponding to a quartic-quadratic potential.

    Parameters
    ----------
    current_r
        The separation to evaluate the energy at
    r_eqm
        The equilibrium separation
    r_stiff
        A length scale over which the potential transitions from quadratic to quartic
    quadratic_k
        Force constant for quadratic term in Nm^-2
    quartic_k
        Force constant for quartic term in Nm^-4
    Returns
    -------
    force
        The energy at this separation
    """
    exp_contents = np.abs((current_r - r_eqm) / r_stiff)

    harm_term = quadratic_k * (current_r - r_eqm) ** 2
    quartic_term = quartic_k * (current_r - r_eqm) ** 4

    erf_scale = (current_r - r_eqm) ** 2 / (r_stiff) ** 2

    erf_term = scipy.special.erf(erf_scale)
    erfc_term = 1.0 - erf_term
    return (erfc_term * harm_term) + (erf_term * quartic_term)


def harm_exp_r(current_r):
    harm_term = (current_r - R_STIFF) ** 2
    stiff_eqm = (R_STIFF - R_EQM) ** 2
    exp_contents = -(current_r - R_EQM) / (R_STIFF - R_EQM)
    prefactor = BURD_K_BOND
    energy = prefactor * (
        harm_term + (stiff_eqm * (1.0 - (2.0 * np.exp(exp_contents))))
    )
    energy[R_EQM > current_r] *= -1
    return energy


def harm_exp_force(current_r):
    harm_term = current_r - R_STIFF
    stiff_eqm = R_STIFF - R_EQM
    exp_contents = -(current_r - R_EQM) / (R_STIFF - R_EQM)
    prefactor = 2.0 * BURD_K_BOND
    force = prefactor * (harm_term + (stiff_eqm * np.exp(exp_contents)))
    force[R_EQM > current_r] *= -1
    return np.sign(current_r - R_STIFF) * force


def lj_12_4(current_r, epsilon=EPSILON, sigma=SIGMA):
    return EPSILON * ((SIGMA / current_r) ** 12 - (SIGMA / current_r) ** 4)


def lj_12_4_force(current_r, epsilon=EPSILON, sigma=SIGMA):
    return -(EPSILON / SIGMA) * (
        -12 * (SIGMA / current_r) ** 13 + 4 * (SIGMA / current_r) ** 5
    )


def shifted_morse_potential(current_r, D_e=0.1657, r_eqm=50.0, alpha=0.15):
    """
    Morse Potential, given by
    V(x) = D_e( 1 - exp(alpha*(r - r_eqm)))^2 - D_e

    Parameters
    ----------
    current_r
        The distances at which to evaluate the potential
    d_e
        Dissociation energy of the bond / well depth
    r_eqm
        Position of the minimum
    alpha
        1/distance units, represents steepness of well
    """
    return (D_e * (1.0 - np.exp(-alpha * (current_r - r_eqm))) ** 2) - D_e


def shifted_morse_potential_force(current_r, D_e=0.1657, r_eqm=50.0, alpha=0.15):
    """
    Morse Potential, given by
    V(x) = D_e( 1 - exp(alpha*(r - r_eqm)))^2 - D_e

    Parameters
    ----------
    current_r
        The distances at which to evaluate the potential
    d_e
        Dissociation energy of the bond / well depth
    r_eqm
        Position of the minimum
    alpha
        1/distance units, represents steepness of well
    """
    exp_term = np.exp(-alpha * (current_r - r_eqm))
    return -2.0 * alpha * D_e * exp_term * (1.0 - exp_term)


def check_force_function(func, *args):
    xs = np.linspace(1, 1000, 101)

    scipy_forces = -scipy.misc.derivative(func, xs, dx=0.01, args=args)
    analytic_forces = ENERGY_TO_FORCE[func](xs, *args)

    diffs = np.abs(analytic_forces - scipy_forces)
    fractional_diffs = diffs / np.minimum(np.abs(scipy_forces), np.abs(analytic_forces))
    print(fractional_diffs)
    within_diff = np.isclose(analytic_forces, scipy_forces)
    plt.plot(xs, scipy_forces, label="scipy")
    plt.plot(xs, analytic_forces, label="Analytic")
    plt.legend()
    plt.show()
    return np.all(within_diff)


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


ENERGY_TO_FORCE = keydefaultdict(
    lambda func: lambda x0, *args: -scipy.misc.derivative(func, x0, dx=0.01, args=args)
)
ENERGY_TO_FORCE[harm_exp_r] = harm_exp_force
ENERGY_TO_FORCE[lj_12_4] = lj_12_4_force
ENERGY_TO_FORCE[quartquad] = quartquad_force
ENERGY_TO_FORCE[shifted_morse_potential] = shifted_morse_potential_force


def write_bond_table(
    func, bond_xs: np.array, label: str, filename: str, comment: str = None, r_eqm=None
):
    """
    Write out a bond table in the lammps table format.
    Checks ENERGY_TO_FORCE to find a force function corresponding to func and uses scipy otherwise.

    Parameters
    ----------
    func
        The energy function to evaluate
    bond_xs
        Numpy array of positions to evaluate the energies.
    label
        The LAMMPS label within this file
    filename
        The name of the file to write to
    comment
        A comment string to describe this
    r_eqm
        The equilibrium distance of this bond; found as the minimum if not specified.
    """
    if not check_force_function(func):
        print("Warning: calculated derivatives do not match.")

    energies = func(bond_xs)
    forces = ENERGY_TO_FORCE[func](bond_xs)
    if r_eqm is None:
        r_eqm = bond_xs[np.argmin(energies)]
    with open(f"{filename}", "w") as fi:
        fi.write(f"# {comment}\n")
        fi.write("\n")
        fi.write(f"{label}\n")
        fi.write(f"N {bond_xs.shape[0]} EQ {r_eqm}\n")
        fi.write("\n")
        for idx in range(bond_xs.shape[0]):
            fi.write(f"{idx+1} {bond_xs[idx]} {energies[idx]} {forces[idx]}\n")


def write_pair_table(
    func, pair_xs: np.array, label: str, filename: str, comment: str = None, r_eqm=None
):
    """
    Write out a pair table in the lammps table format, which are slightly different to bond tables.
    Checks ENERGY_TO_FORCE to find a force function corresponding to func and uses scipy otherwise.

    Parameters
    ----------
    func
        The pair energy function to evaluate
    bond_xs
        Numpy array of positions to evaluate the energies.
    label
        The LAMMPS label within this file
    filename
        The name of the file to write to
    comment
        A comment string to describe this
    r_eqm
        The equilibrium distance of this bond; found as the minimum if not specified.
    """

    energies = func(pair_xs)
    forces = ENERGY_TO_FORCE[func](pair_xs)
    with open(f"{filename}", "w") as fi:
        fi.write(f"# {comment}\n")
        fi.write("\n")
        fi.write(f"{label}\n")
        fi.write(f"N {pair_xs.shape[0]}\n")
        fi.write("\n")
        for idx in range(pair_xs.shape[0]):
            fi.write(f"{idx+1} {pair_xs[idx]} {energies[idx]} {forces[idx]}\n")


def main():
    write_pair_table(
        lj_12_4,
        np.linspace(40, 200, 10001),
        filename="lj-12-4.dat",
        label="LJ124",
        comment="LJ 12-4",
    )
    write_bond_table(
        shifted_morse_potential,
        np.linspace(1, 150, 14901),
        filename="shifted-morse.dat",
        label="SHIFTMORSE",
        comment="Morse with d_e=0.1657, r_eqm=50.0, alpha=0.15",
    )


if __name__ == "__main__":
    main()
