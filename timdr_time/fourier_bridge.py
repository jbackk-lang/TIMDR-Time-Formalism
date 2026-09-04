"""
timdr_time/fourier_bridge.py

Most Fouriera M/S<->K: pierwszy w tym projekcie faktycznie sprawdzalny
MOST miedzy dwiema galeziami TIMDR (nie identyfikacja -- patrz uwaga
nizej). Zrodlo: dualizm falowo-czasteczkowy fotonu ma DOKLADNE zrodlo
matematyczne -- zasada nieoznaczonosci Heisenberga (Delta_x*Delta_p
>= hbar/2) ma dokladnie ta sama strukture co klasyczna zasada
nieoznaczonosci Gabora/Fouriera dla sygnalow (Delta_t*Delta_f >= const,
z rownoscia dla impulsu gaussowskiego) -- to jest ta sama matematyka
(sprzezone zmienne Fouriera), nie luzna analogia.

W TIMDR: x(t) z galezi M/S (`timdr_formalism.chronosignal`) jest
opisem "czastkowym" -- dokladna lokalizacja w czasie. Modalnosc
(f,phi,A) z galezi K (`timdr_modal`) jest opisem "falowym" -- dokladna
lokalizacja w czestotliwosci, zerowa w czasie. FFT jest DOKLADNA,
znana od 200 lat mapa miedzy tymi dwoma opisami. `fft_modalities()`
ponizej realizuje ta mape: x(t) -> lista obiektow Modality, gotowych
do uzycia w juz istniejacych funkcjach K (`interference`,
`is_resonant`).

## Dlaczego to NIE lamie nieredukowalnosci galezi

`GIA-TIMDR/docs/theory/TIMDR_Branch_Specification.md` wymaga, zeby M/S
i K byly formalnie odrebne. Ten modul tego nie neguje: nie twierdzi,
ze "x(t) TO JEST (f,phi,A)". Twierdzi, ze istnieje KONKRETNA, znana
TRANSFORMATA (FFT) miedzy jednym a drugim -- dokladnie tak, jak istnieje
transformata miedzy funkcja i jej pochodna, bez utozsamiania funkcji z
pochodna. `timdr_time/chronoprocess.py` (Xi=(T,x,Gamma,phi)) CELOWO
nie ma zadnej takiej funkcji -- ten modul jest jedynym, jawnie
wyodrebnionym wyjatkiem, uzasadnionym tym, ze FFT jest ustalona,
rygorystyczna matematyka, nie nowa hipoteza.

## Zasada nieoznaczonosci Gabora -- co dokladnie jest tu policzone

Dla energetycznie wazonych odchylen standardowych (Delta_t = odchylenie
std rozkladu |x(t)|^2 wzgledem srodka ciezkosci w czasie; Delta_f =
odchylenie std rozkladu A_k^2 WOKOL ZERA w czestotliwosci -- zero jest
poprawnym srodkiem dla KAZDEGO sygnalu rzeczywistego, bo jego pelne,
dwustronne widmo jest zawsze symetryczne wzgledem f=0 z symetrii
Hermite'a; rfft zwraca tylko polowe nieujemna, wiec liczenie wariancji
"wokol zera" zamiast "wokol sredniej jednostronnej" jest tym, co
odtwarza poprawna, fizyczna wariancje pelnego widma) -- dla impulsu
gaussowskiego `x(t)=exp(-(t-t0)^2/(2*sigma^2))` wyprowadzono TU, w tej
sesji, recznie (nie przepisane z pamieci):

    Delta_t = sigma/sqrt(2)
    Delta_f = 1/(2*sqrt(2)*pi*sigma)
    Delta_t * Delta_f = 1/(4*pi)  -- NIEZALEZNE od sigma

Wyprowadzenie: energia w czasie |x(t)|^2=exp(-(t-t0)^2/sigma^2) ma
wariancje sigma^2/2 (Gaussian o "efektywnym tau^2=sigma^2/2"). Ciagla
transformata Fouriera x(t) to (co do stalej) exp(-2*pi^2*sigma^2*f^2),
wiec jej energia |X(f)|^2 jest Gaussianem o wariancji 1/(8*pi^2*sigma^2)
wokol f=0. sqrt tych wariancji daje wzory powyzej; iloczyn upraszcza
sie do 1/(4*pi), niezaleznie od sigma -- dokladnie granica
Gabora/Heisenberga OSIAGANA (rownosc, nie tylko nierownosc) przez
impuls gaussowski, znany fakt z analizy Fouriera.

UWAGA O DYSKRETYZACJI: powyzsze jest wyprowadzone w granicy ciaglej.
Testy w tests/test_fourier_bridge.py licza to na DYSKRETNEJ,
SKONCZONEJ siatce (FFT o skonczonej rozdzielczosci) -- oczekiwane sa
odchylenia rzedu kilkunastu-kilkudziesieciu procent od 1/(4*pi), stad
szerokie tolerancje w tych testach (ten sam wzorzec co
tests/test_geometry_weingarten.py).

UWAGA O WYKONANIU: napisane bez dostepu do sandboxa bash w tej sesji.
Wyprowadzenie powyzej zostalo sprawdzone recznie dwa razy (raz wprost,
raz przez konsystencje wymiarowa -- niezaleznosc od sigma), ale kod
NIE zostal uruchomiony. Uruchom `pytest tests/test_fourier_bridge.py -v`.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from . import _bootstrap  # noqa: F401

try:
    from timdr_modal import Modality as _Modality
except ImportError as e:
    raise ImportError(
        "Nie mozna zaimportowac timdr_modal. Sprawdz, czy "
        "TIMDR-Modal-Formalism istnieje jako repo-siostra (oczekiwana "
        "sciezka wzgledna ../TIMDR-Modal-Formalism)."
    ) from e


__all__ = [
    "fft_modalities",
    "temporal_spread",
    "spectral_spread",
    "time_bandwidth_product",
]


def fft_modalities(
    x: Sequence[float], dt: float, include_dc: bool = False
) -> Tuple[List[_Modality], float]:
    """x(t) [rownomiernie probkowany, odstep dt] -> lista Modality(f,phi,A).

    Standardowy rozklad Fouriera sygnalu rzeczywistego (rfft), przepisany
    na obiekty K-branchowe. Konwencja fazy: K uzywa sin(2*pi*f*t+phi)
    (timdr_modal.instantaneous_phase); rekonstrukcja FFT naturalnie daje
    cosinusy (X_k=|X_k|*e^{i*theta_k} -> skladowa |X_k|*cos(2*pi*f*t+theta_k)).
    Zeby dwie konwencje sie zgadzaly, faza modalnosci jest przesunieta:
    phi_k := angle(X_k) + pi/2 (bo sin(a+pi/2)=cos(a)) -- to jest
    sprawdzone w tests/test_fourier_bridge.py przez porownanie
    `timdr_modal.interference()` na zwroconych modalnosciach z
    oryginalnym sygnalem (dokladny test odtworzenia).

    Zwraca (modalnosci, dc_offset) -- dc_offset to srednia sygnalu
    (skladowa f=0), zwrocona OSOBNO domyslnie (include_dc=False),
    zeby uniknac degenerowanej modalnosci o f=0 w typowym uzyciu
    (np. karmienie wyniku do is_resonant()). Ustaw include_dc=True,
    zeby DC bylo wlaczone jako zwykla Modality(f=0,...) w liscie --
    WYMAGANE przy liczeniu spectral_spread() dla sygnalow niebazowanych
    na zerze (np. impuls gaussowski dodatni), bo dla takich sygnalow
    DC jest SZCZYTEM widma, nie czyms do odrzucenia (patrz uzycie w
    time_bandwidth_product poniżej i w testach).

    Rzuca ValueError dla sygnalow < 2 probek.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("fft_modalities() wymaga >= 2 probek")

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=dt)
    dc_offset = float(X[0].real) / n

    is_even = (n % 2 == 0)
    nyquist_index = n // 2 if is_even else None

    modalities: List[_Modality] = []
    start_k = 0 if include_dc else 1
    for k in range(start_k, len(X)):
        special = (k == 0) or (is_even and k == nyquist_index)
        factor = (1.0 / n) if special else (2.0 / n)
        amplitude = factor * float(np.abs(X[k]))
        phi = float(np.angle(X[k])) + np.pi / 2.0
        modalities.append(_Modality(f=float(freqs[k]), phi=phi, A=amplitude))

    return modalities, dc_offset


def temporal_spread(x: Sequence[float], dt: float) -> Tuple[float, float]:
    """(t_srednie, Delta_t) -- srodek ciezkosci i odchylenie standardowe
    rozkladu ENERGII |x(t)|^2 w dziedzinie czasu (wazona wariancja
    wokol wlasnej sredniej -- poprawne dla dowolnego sygnalu, niezaleznie
    od tego, gdzie lezy jego "srodek")."""
    x = np.asarray(x, dtype=float)
    n = x.size
    t = np.arange(n, dtype=float) * dt
    weights = x ** 2
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("sygnal zerowy -- rozklad energii w czasie niezdefiniowany")
    t_mean = float(np.sum(t * weights) / total)
    t_var = float(np.sum((t - t_mean) ** 2 * weights) / total)
    return t_mean, float(np.sqrt(t_var))


def spectral_spread(modalities: Sequence[_Modality]) -> Tuple[float, float]:
    """(0.0, Delta_f) -- odchylenie standardowe rozkladu energii A_k^2
    WOKOL ZERA (nie wokol jednostronnej sredniej -- patrz uzasadnienie
    w naglowku modulu: zero jest fizycznie poprawnym srodkiem pelnego,
    dwustronnego widma kazdego sygnalu rzeczywistego). Pierwszy element
    krotki jest zawsze 0.0, zwrocony tylko dla symetrii API z
    temporal_spread()."""
    if len(modalities) == 0:
        raise ValueError("pusta lista modalnosci")
    f = np.array([m.f for m in modalities], dtype=float)
    weights = np.array([m.A ** 2 for m in modalities], dtype=float)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("wszystkie amplitudy zerowe -- rozklad energii w częstotliwości niezdefiniowany")
    f_var = float(np.sum((f ** 2) * weights) / total)
    return 0.0, float(np.sqrt(f_var))


def time_bandwidth_product(x: Sequence[float], dt: float, include_dc: bool = False) -> float:
    """Delta_t * Delta_f -- iloczyn czas-pasmo, ilosciowa tresc zasady
    nieoznaczonosci Gabora dla sygnalu x. Nie zaklada zadnej konkretnej
    dolnej granicy w kodzie (patrz naglowek modulu za wyprowadzenie
    1/(4*pi) dla impulsu gaussowskiego, testowane osobno w
    tests/test_fourier_bridge.py) -- ta funkcja tylko liczy sam iloczyn."""
    modalities, _ = fft_modalities(x, dt, include_dc=include_dc)
    _, delta_t = temporal_spread(x, dt)
    _, delta_f = spectral_spread(modalities)
    return delta_t * delta_f
