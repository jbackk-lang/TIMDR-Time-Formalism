"""
tests/test_fourier_bridge.py

Testy timdr_time.fourier_bridge -- most Fouriera M/S<->K.

Dwie kategorie:
  1. Dokladny test jednotonowy (bez zadnej tolerancji poza floatami) --
     sygnal dokladnie na jednym binie FFT, wszystko policzalne recznie.
  2. Impuls gaussowski i zasada nieoznaczonosci Gabora -- wyprowadzenie
     Delta_t*Delta_f=1/(4*pi) jest w naglowku fourier_bridge.py;
     testy tutaj uzywaja SZEROKICH tolerancji (dyskretyzacja/skonczone
     okno FFT odchyla wynik od granicy ciaglej), zgodnie z konwencja
     ustalona juz w tests/test_geometry_weingarten.py.

UWAGA: ten plik NIE zostal uruchomiony w sesji, w ktorej powstal
(sandbox bash niedostepny). Uruchom `pytest tests/ -v`.
"""
import numpy as np
import pytest

from timdr_time.fourier_bridge import (
    fft_modalities,
    temporal_spread,
    spectral_spread,
    time_bandwidth_product,
)
from timdr_modal import interference


# ---------------------------------------------------------------------
# Test dokladny: pojedynczy ton dokladnie na jednym binie FFT
# ---------------------------------------------------------------------

class TestSingleToneExact:
    N = 8
    DT = 1.0
    F0 = 0.25   # = bin k=2 dla N=8, dt=1 (freqs = [0,1,2,3,4]/8)
    A0 = 3.0
    PHI0 = 0.5

    def _signal(self):
        t = np.arange(self.N) * self.DT
        return self.A0 * np.cos(2 * np.pi * self.F0 * t + self.PHI0)

    def test_energy_concentrated_in_expected_bin(self):
        # Recznie wyprowadzone w naglowku modulu / komentarzach sesji:
        # dla tonu dokladnie na binie k, X[k] = (N/2)*A0*e^{i*phi0} ->
        # |X[k]|=N*A0/2=8*3/2=12 ; amplituda po skalowaniu (factor=2/N)
        # = 0.25*12 = 3.0 = A0 dokladnie. Wszystkie inne biny ~0
        # (ortogonalnosc DFT dla tonu dokladnie na binie, brak przecieku).
        x = self._signal()
        modalities, dc = fft_modalities(x, self.DT, include_dc=False)

        # bin k=2 -> indeks 1 na liscie (bo include_dc=False pomija k=0,
        # wiec lista ma indeksy odpowiadajace k=1,2,3,4 -> pozycje 0,1,2,3)
        target = modalities[1]
        assert target.f == pytest.approx(self.F0)
        assert target.A == pytest.approx(self.A0, abs=1e-9)
        # phi_k = angle(X_k)+pi/2 ; dla czystego tonu na binie, angle(X_k)=PHI0
        assert target.phi == pytest.approx(self.PHI0 + np.pi / 2, abs=1e-9)

        others = [m for i, m in enumerate(modalities) if i != 1]
        for m in others:
            assert m.A < 1e-9, f"oczekiwano ~0 amplitudy poza binem docelowym, dostano {m.A}"
        assert abs(dc) < 1e-9

    def test_interference_reconstructs_original_signal(self):
        # Rekonstrukcja: interference() na zwroconych modalnosciach
        # (juz istniejaca, przetestowana funkcja z timdr_modal) musi
        # odtworzyc oryginalny sygnal w punktach probkowania -- to jest
        # DOKLADNY test twierdzenia o syntezie Fouriera, nie przyblizenie.
        x = self._signal()
        modalities, dc = fft_modalities(x, self.DT, include_dc=False)
        t = np.arange(self.N) * self.DT
        reconstructed = dc + interference(modalities, t)
        np.testing.assert_allclose(reconstructed, x, atol=1e-9)

    def test_rejects_too_short_signal(self):
        with pytest.raises(ValueError):
            fft_modalities([1.0], dt=1.0)


# ---------------------------------------------------------------------
# Impuls gaussowski -- zasada nieoznaczonosci Gabora (Delta_t*Delta_f)
# ---------------------------------------------------------------------

def _gaussian_pulse(n: int, dt: float, sigma_samples: float, t0_samples: float) -> np.ndarray:
    n_arr = np.arange(n, dtype=float)
    return np.exp(-((n_arr - t0_samples) ** 2) / (2.0 * sigma_samples ** 2))


class TestGaborUncertainty:
    N = 512
    DT = 1.0
    T0 = 256.0
    SIGMA_NARROW = 8.0
    SIGMA_WIDE = 16.0
    EXPECTED_PRODUCT = 1.0 / (4.0 * np.pi)  # ~0.07958, wyprowadzone w naglowku modulu

    def test_narrow_pulse_time_bandwidth_product_near_gabor_limit(self):
        x = _gaussian_pulse(self.N, self.DT, self.SIGMA_NARROW, self.T0)
        product = time_bandwidth_product(x, self.DT, include_dc=True)
        # Tolerancja SZEROKA (30%) -- dyskretyzacja/skonczone okno FFT,
        # nie uruchomione w tej sesji, patrz naglowek pliku.
        assert product == pytest.approx(self.EXPECTED_PRODUCT, rel=0.3)

    def test_wide_pulse_time_bandwidth_product_near_gabor_limit(self):
        x = _gaussian_pulse(self.N, self.DT, self.SIGMA_WIDE, self.T0)
        product = time_bandwidth_product(x, self.DT, include_dc=True)
        assert product == pytest.approx(self.EXPECTED_PRODUCT, rel=0.3)

    def test_product_is_approximately_scale_invariant(self):
        # Rdzen zasady nieoznaczonosci: iloczyn NIE zalezy od sigma
        # (wyprowadzenie w naglowku modulu) -- sprawdzone tu jako
        # porownanie miedzy dwoma impulsami roznej szerokosci, z
        # szerokim tolerowanym stosunkiem (1.5x), nie precyzyjna liczba.
        x_narrow = _gaussian_pulse(self.N, self.DT, self.SIGMA_NARROW, self.T0)
        x_wide = _gaussian_pulse(self.N, self.DT, self.SIGMA_WIDE, self.T0)
        p_narrow = time_bandwidth_product(x_narrow, self.DT, include_dc=True)
        p_wide = time_bandwidth_product(x_wide, self.DT, include_dc=True)
        ratio = p_narrow / p_wide
        assert 1.0 / 1.5 < ratio < 1.5, (
            f"oczekiwano w przyblizeniu rownych iloczynow (niezaleznosc od "
            f"sigma), dostano p_narrow={p_narrow}, p_wide={p_wide}, "
            f"stosunek={ratio}"
        )

    def test_narrower_pulse_has_larger_bandwidth_and_smaller_time_spread(self):
        # Kierunek kompromisu (bezpieczniejszy test niz precyzyjna
        # wartosc): waskie w czasie -> szerokie w czestotliwosci, i
        # odwrotnie. Powinno byc prawdziwe niezaleznie od precyzji
        # dyskretyzacji.
        x_narrow = _gaussian_pulse(self.N, self.DT, self.SIGMA_NARROW, self.T0)
        x_wide = _gaussian_pulse(self.N, self.DT, self.SIGMA_WIDE, self.T0)

        _, dt_narrow = temporal_spread(x_narrow, self.DT)
        _, dt_wide = temporal_spread(x_wide, self.DT)
        assert dt_narrow < dt_wide

        mods_narrow, _ = fft_modalities(x_narrow, self.DT, include_dc=True)
        mods_wide, _ = fft_modalities(x_wide, self.DT, include_dc=True)
        _, df_narrow = spectral_spread(mods_narrow)
        _, df_wide = spectral_spread(mods_wide)
        assert df_narrow > df_wide

    def test_temporal_spread_matches_hand_derivation(self):
        # Wyprowadzenie w naglowku modulu: Delta_t = sigma/sqrt(2).
        x = _gaussian_pulse(self.N, self.DT, self.SIGMA_NARROW, self.T0)
        t_mean, delta_t = temporal_spread(x, self.DT)
        assert t_mean == pytest.approx(self.T0, abs=1.0)
        expected = self.SIGMA_NARROW / np.sqrt(2.0)
        assert delta_t == pytest.approx(expected, rel=0.05)

    def test_rejects_empty_modalities_list(self):
        with pytest.raises(ValueError):
            spectral_spread([])

    def test_rejects_zero_signal(self):
        with pytest.raises(ValueError):
            temporal_spread(np.zeros(10), dt=1.0)
