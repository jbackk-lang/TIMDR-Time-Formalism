"""
tests/test_chronoprocess.py

Testy timdr_time.Chronoprocess -- kazda liczba ponizej jest PRZEPISANA
z juz ustalonych, hand-traced faktow w repo-siostrach (nie nowa
derywacja): x_tempo/x_drift odtwarzaja dokladnie
tests/test_chronosignal.py::test_tempo_basic/test_drift_basic z
TIMDR-Math-Formalism; gamma_surface odtwarza rownowaznosc z
tests/test_chronocongruence.py z TIMDR-Geometry-Formalism;
phi_phase/phi_interference odtwarzaja tests/test_phase_sync.py z
TIMDR-Modal-Formalism. Cel tego pliku to nie ponowna derywacja
matematyki, tylko sprawdzenie, ze ORKIESTRACJA (import miedzy-repo,
delegacja) faktycznie dziala i nie psuje wynikow po drodze.

WYMAGA: TIMDR-Math-Formalism, TIMDR-Geometry-Formalism i
TIMDR-Modal-Formalism jako repo-siostry obok TIMDR-Time-Formalism
(patrz timdr_time/_bootstrap.py) -- bez nich import `timdr_time`
rzuci czytelny ImportError.

UWAGA: ten plik zostal odtad faktycznie uruchomiony przez uzytkownika
(`pytest tests/ -v`) -- ZWERYFIKOWANE, 8/8 testow przeszlo.
"""
import numpy as np
import pytest

from timdr_time import Chronoprocess

# Te importy dzialaja tylko dzieki temu, ze `from timdr_time import
# Chronoprocess` powyzej juz uruchomilo _bootstrap.py i wstrzyknelo
# sciezki repo-siostr do sys.path.
from timdr_geometry import make_cylinder_mesh
from timdr_modal import Modality


# ---------------------------------------------------------------------
# Konstrukcja / przypadki brzegowe
# ---------------------------------------------------------------------

def test_rejects_single_point_carrier():
    with pytest.raises(ValueError):
        Chronoprocess(T=[5.0])


def test_rejects_2d_carrier():
    with pytest.raises(ValueError):
        Chronoprocess(T=[[0.0, 1.0], [2.0, 3.0]])


def test_accepts_minimal_two_point_carrier():
    cp = Chronoprocess(T=[0.0, 1.0])
    np.testing.assert_array_equal(cp.T, np.array([0.0, 1.0]))


# ---------------------------------------------------------------------
# Rzut M/S -- x_tempo / x_drift (odtwarza TIMDR-Math-Formalism)
# ---------------------------------------------------------------------

def test_x_tempo_matches_chronosignal_directly():
    # Te same timestamps i oczekiwany wynik co
    # test_chronosignal.py::test_tempo_basic w TIMDR-Math-Formalism.
    cp = Chronoprocess(T=[0, 60, 120, 185])
    np.testing.assert_array_equal(cp.x_tempo(), np.array([60.0, 60.0, 65.0]))


def test_x_drift_matches_chronosignal_directly():
    # Te same timestamps/nominal i oczekiwany wynik co
    # test_chronosignal.py::test_drift_basic w TIMDR-Math-Formalism.
    cp = Chronoprocess(T=[0, 60, 125, 180])
    np.testing.assert_array_equal(cp.x_drift(nominal_interval=60.0), np.array([0.0, 5.0, -5.0]))


# ---------------------------------------------------------------------
# Rzut G -- gamma_surface (odtwarza rownowaznosc z
# TIMDR-Geometry-Formalism/tests/test_chronocongruence.py)
# ---------------------------------------------------------------------

def test_gamma_surface_matches_make_cylinder_mesh():
    from timdr_geometry.chronocongruence import cylindrical_congruence

    radius, n_theta, n_z, height = 1.5, 24, 12, 3.0
    zs = np.linspace(0.0, height, n_z)
    thetas = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)

    cp = Chronoprocess(T=zs)
    mesh = cp.gamma_surface(
        lambda t, s: cylindrical_congruence(t, s, radius=radius),
        s_values=thetas,
        s_periodic=True,
    )
    expected = make_cylinder_mesh(n_theta, n_z, radius=radius, height=height)

    np.testing.assert_allclose(mesh.vertices, expected.vertices, atol=1e-10)
    np.testing.assert_array_equal(mesh.faces, expected.faces)


# ---------------------------------------------------------------------
# Rzut K -- phi_phase / phi_interference (odtwarza TIMDR-Modal-Formalism)
# ---------------------------------------------------------------------

def test_phi_phase_matches_instantaneous_phase_directly():
    # f=1, phi=0, T=[0, 0.25] -> theta=[0, pi/2] (patrz
    # test_phase_sync.py::test_instantaneous_phase_array).
    cp = Chronoprocess(T=[0.0, 0.25])
    m = Modality(f=1.0, phi=0.0, A=1.0)
    result = cp.phi_phase(m)
    np.testing.assert_allclose(result, np.array([0.0, np.pi / 2]))


def test_phi_interference_matches_interference_directly():
    # m1: A=2,phi=0 ; m2: A=3,phi=pi/2 ; oba f=1.
    # t=0:    2*sin(0)      + 3*sin(pi/2)    = 0 + 3 = 3.0
    # t=0.25: 2*sin(pi/2)   + 3*sin(pi)      = 2 + ~0 = ~2.0
    cp = Chronoprocess(T=[0.0, 0.25])
    m1 = Modality(f=1.0, phi=0.0, A=2.0)
    m2 = Modality(f=1.0, phi=np.pi / 2, A=3.0)
    result = cp.phi_interference([m1, m2])
    np.testing.assert_allclose(result, np.array([3.0, 2.0]), atol=1e-9)
