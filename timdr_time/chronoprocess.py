"""
timdr_time/chronoprocess.py

Chronoproces Xi=(T,x,Gamma,phi): kontener TRZECH NIEZALEZNYCH rzutow
wspolnego nosnika T na obiekty natywne dla kazdej galezi TIMDR --
M/S (chronosignal.tempo/drift, TIMDR-Math-Formalism), G
(chronocongruence.make_congruence_mesh, TIMDR-Geometry-Formalism), K
(phase_sync.instantaneous_phase/interference, TIMDR-Modal-Formalism).

ZADNEJ IDENTYFIKACJI miedzy rzutami -- to jest warunek zgodnosci z
GIA-TIMDR/docs/theory/TIMDR_Branch_Specification.md (nieredukowalnosc
galezi): Chronoproces NIE twierdzi, ze "czas" jest tym samym obiektem
matematycznym w M/S, G i K. Kazda metoda ponizej wola WYLACZNIE kod
juz istniejacy w odpowiednim repo-siostrze -- ten modul NIE dodaje
zadnej nowej matematyki, tylko orkiestracje trzech juz zbudowanych
kawalkow.

T (nosnik) jest jedynym polem dzielonym miedzy trzema rzutami -- i
nawet ono jest uzywane INACZEJ w kazdej galezi:
  - M/S czyta T jako znaczniki czasu i ROZNICZKUJE je (tempo=diff(T)).
  - G czyta T jako os "t" siatki Gamma(t,s) -- potrzebuje DODATKOWO
    domeny I (s_values), ktora nie pochodzi z T w ogole.
  - K NIE uzywa T do zdefiniowania modalnosci -- modalnosc (f,phi,A)
    jest niezalezna od T; T sluzy tylko jako punkty, w ktorych
    OCENIAMY jej faze/interferencje.
Ta niejednorodnosc jest ZAMIERZONA, nie niedopatrzeniem -- to jest
dokladnie to, co "trzy niezalezne rzuty, zero identyfikacji" ma
znaczyc konkretnie w kodzie.

UWAGA O ZALEZNOSCIACH: ten modul wymaga TIMDR-Math-Formalism,
TIMDR-Geometry-Formalism i TIMDR-Modal-Formalism jako repo-siostry pod
tym samym katalogiem nadrzednym (patrz _bootstrap.py) -- nie jest
samodzielny i nie duplikuje ich kodu.

UWAGA O WYKONANIU: napisane bez dostepu do sandboxa bash w tej sesji.
Wartosci liczbowe w tests/test_chronoprocess.py sa przepisane z juz
ustalonych, hand-traced faktow w repo-siostrach (nie nowa derywacja) --
ale nic tutaj nie zostalo faktycznie uruchomione. Uruchom
`pytest tests/ -v`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from . import _bootstrap  # noqa: F401 -- musi biec PRZED importami ponizej

try:
    from timdr_formalism.chronosignal import tempo as _tempo, drift as _drift
except ImportError as e:
    raise ImportError(
        "Nie mozna zaimportowac timdr_formalism.chronosignal. Sprawdz: (1) "
        "czy TIMDR-Math-Formalism istnieje jako repo-siostra obok "
        "TIMDR-Time-Formalism (oczekiwana sciezka wzgledna "
        "../TIMDR-Math-Formalism); (2) czy jego zaleznosci "
        "(numpy, scipy) sa zainstalowane -- patrz oryginalny wyjatek ponizej."
    ) from e

try:
    from timdr_geometry.chronocongruence import make_congruence_mesh as _make_congruence_mesh
    from timdr_geometry.weingarten import Mesh as _Mesh
except ImportError as e:
    raise ImportError(
        "Nie mozna zaimportowac timdr_geometry.chronocongruence. Sprawdz: (1) "
        "czy TIMDR-Geometry-Formalism istnieje jako repo-siostra (oczekiwana "
        "sciezka wzgledna ../TIMDR-Geometry-Formalism); (2) czy numpy jest "
        "zainstalowane -- patrz oryginalny wyjatek ponizej."
    ) from e

try:
    from timdr_modal import (
        Modality as _Modality,
        instantaneous_phase as _instantaneous_phase,
        interference as _interference,
    )
except ImportError as e:
    raise ImportError(
        "Nie mozna zaimportowac timdr_modal. Sprawdz: (1) czy "
        "TIMDR-Modal-Formalism istnieje jako repo-siostra (oczekiwana "
        "sciezka wzgledna ../TIMDR-Modal-Formalism); (2) czy numpy jest "
        "zainstalowane -- patrz oryginalny wyjatek ponizej."
    ) from e


__all__ = ["Chronoprocess"]


@dataclass
class Chronoprocess:
    """Xi=(T,x,Gamma,phi) -- patrz naglowek modulu.

    T: wspolny nosnik (np. znaczniki czasu), >= 2 punkty.
    """

    T: np.ndarray

    def __post_init__(self) -> None:
        self.T = np.asarray(self.T, dtype=float)
        if self.T.ndim != 1:
            raise ValueError("T musi byc jednowymiarowa tablica")
        if self.T.size < 2:
            raise ValueError("Chronoproces wymaga >= 2 punktow nosnika T")

    # ------------------------------------------------------------
    # Rzut M/S -- tempo/drift (TIMDR-Math-Formalism.chronosignal)
    # ------------------------------------------------------------

    def x_tempo(self) -> np.ndarray:
        """tempo(T) -- bez zadnej zmiany, patrz
        timdr_formalism.chronosignal.tempo."""
        return _tempo(self.T)

    def x_drift(self, nominal_interval: float) -> np.ndarray:
        """drift(T, nominal_interval) -- bez zadnej zmiany, patrz
        timdr_formalism.chronosignal.drift."""
        return _drift(self.T, nominal_interval)

    # ------------------------------------------------------------
    # Rzut G -- kongruencja Gamma(t,s) (TIMDR-Geometry-Formalism.chronocongruence)
    # ------------------------------------------------------------

    def gamma_surface(
        self,
        gamma: Callable[[float, float], np.ndarray],
        s_values: Sequence[float],
        s_periodic: bool = False,
    ) -> _Mesh:
        """Buduje S=Gamma(TxI): self.T jest osia "t" (zawsze
        t_periodic=False -- czas sam w sobie sie nie zawija),
        s_values jest domena I. Bez zadnej zmiany, patrz
        timdr_geometry.chronocongruence.make_congruence_mesh."""
        return _make_congruence_mesh(
            gamma, t_values=self.T, s_values=s_values,
            t_periodic=False, s_periodic=s_periodic,
        )

    # ------------------------------------------------------------
    # Rzut K -- proces modalny (TIMDR-Modal-Formalism.phase_sync)
    # ------------------------------------------------------------

    def phi_phase(self, modality: _Modality) -> np.ndarray:
        """theta(T) dla podanej modalnosci, oceniona w punktach self.T.
        Modalnosc SAMA NIE pochodzi z T -- (f,phi,A) sa niezalezne od
        nosnika, to jest znany, udokumentowany ograniczony zakres
        (Axioms_K modeluje modalnosc jako stala -- patrz
        TIMDR-Modal-Formalism/README.md, sekcja 'Czego to NIE robi').
        Bez zadnej zmiany, patrz timdr_modal.instantaneous_phase."""
        return _instantaneous_phase(modality, self.T)

    def phi_interference(self, modalities: Sequence[_Modality]) -> np.ndarray:
        """I(T) -- interferencja podanych modalnosci ocenionych w
        punktach self.T. Bez zadnej zmiany, patrz
        timdr_modal.interference."""
        return _interference(modalities, self.T)
