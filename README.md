# TIMDR-Time-Formalism

Pełny moduł Chronoprocesu `Ξ=(T,x,Γ,φ)` — orkiestracja trzech już
zbudowanych, niezależnych repo-sióstr na wspólnym nośniku `T`, **bez
żadnej identyfikacji między nimi**:

- **`x`** (gałąź M/S) — `tempo`/`drift`, [`TIMDR-Math-Formalism`](../TIMDR-Math-Formalism)
- **`Γ`** (gałąź G) — kongruencja `Γ(t,s)` + operator kształtu, [`TIMDR-Geometry-Formalism`](../TIMDR-Geometry-Formalism)
- **`φ`** (gałąź K) — proces modalny + mapa synchronizacji faz, [`TIMDR-Modal-Formalism`](../TIMDR-Modal-Formalism)

Ten repo **nie dodaje żadnej nowej matematyki**. Każda metoda
`Chronoprocess` woła wyłącznie kod już istniejący w odpowiednim
repo-siostrze — jedyna nowość to orkiestracja: wspólny nośnik `T`,
trzy oddzielne metody dostępowe, zero prób "spięcia" ich w jeden
obiekt.

**Dlaczego zero identyfikacji jest warunkiem, nie ograniczeniem:**
`GIA-TIMDR/docs/theory/TIMDR_Branch_Specification.md` ustala, że M/S,
G i K są formalnie odrębnymi, nieredukowalnymi obiektami. Chronoproces
NIE twierdzi, że "czas" jest tym samym obiektem matematycznym w
trzech gałęziach — to byłby dokładnie ten błąd, przed którym
`TIMDR_Branch_Specification.md`/`TIMDR_Twists.md` mają chronić. `T`
jest jedynym polem współdzielonym, i nawet ono jest czytane inaczej w
każdej gałęzi: M/S różniczkuje je (`tempo=diff(T)`), G używa go jako
jednej osi siatki `Γ(t,s)` (potrzebuje dodatkowo domeny `I`, która nie
pochodzi z `T`), K w ogóle nie definiuje modalności z `T` — `T` służy
tam tylko jako punkty, w których *oceniamy* fazę/interferencję już
zadanej modalności. Pełny opis całej konstrukcji, z genezą i statusem
wykonania wszystkich czterech repo naraz:
`GIA-TIMDR/docs/theory/TIMDR_Chronoprocess.md`.

## 🧩 Zależności między-repo

Ten moduł **wymaga** wszystkich trzech repo-sióstr jako katalogów
rodzeństwa pod tym samym katalogiem nadrzędnym — dokładnie tak, jak są
dziś rozłożone:

```
a/
├── TIMDR-Math-Formalism/
├── TIMDR-Geometry-Formalism/
├── TIMDR-Modal-Formalism/
└── TIMDR-Time-Formalism/      <- ten repo
```

`timdr_time/_bootstrap.py` wstrzykuje ścieżki tych trzech katalogów do
`sys.path` automatycznie przy imporcie `timdr_time` — nie trzeba nic
instalować ręcznie, wystarczy sklonować wszystkie cztery repo obok
siebie. Brak któregoś z nich daje czytelny `ImportError` wskazujący,
którego repo brakuje.

## 🔧 Instalacja

```
pip install -r requirements.txt
```

`numpy` używane bezpośrednio; `scipy` wymagane transytywnie (import
`timdr_formalism` z `TIMDR-Math-Formalism` uruchamia jego
`pipeline.py`, które używa `scipy.stats`) — patrz komentarz w
`requirements.txt`.

## 🚀 Szybki start

```python
from timdr_time import Chronoprocess
from timdr_geometry.chronocongruence import cylindrical_congruence
from timdr_modal import Modality
import numpy as np

cp = Chronoprocess(T=[0, 60, 120, 185])          # wspólny nośnik: znaczniki czasu

# Rzut M/S — bez zmian względem TIMDR-Math-Formalism
print(cp.x_tempo())                               # [60. 60. 65.]
print(cp.x_drift(nominal_interval=60.0))          # [0. 0. 5.]

# Rzut G — T jako oś "t" kongruencji, domena I podana osobno
thetas = np.linspace(0, 2*np.pi, 12, endpoint=False)
mesh = cp.gamma_surface(
    lambda t, s: cylindrical_congruence(t, s, radius=1.5),
    s_values=thetas, s_periodic=True,
)
print(mesh.n_vertices)                            # 4*12 = 48

# Rzut K — modalność niezależna od T, oceniana W punktach T
m = Modality(f=0.01, phi=0.0, A=1.0)
print(cp.phi_phase(m))                            # theta(T) dla tej modalności
```

## 🧪 Testy

```
pip install pytest
pytest tests/ -v
```

Każda liczba w `tests/test_chronoprocess.py` jest **przepisana** z już
ustalonych, ręcznie prześledzonych faktów w repo-siostrach (nie nowa
derywacja) — cel tych testów to sprawdzenie, że orkiestracja
(import między-repo, delegacja) faktycznie działa i nie psuje wyników
po drodze, nie ponowne dowodzenie matematyki, która jest już
przetestowana gdzie indziej.

**✅ Zweryfikowane.** `pytest tests/ -v` — 18/18 testów przeszło:
8/8 w `test_chronoprocess.py` (import między trzema repo-siostrami +
wszystkie trzy delegacje) i 10/10 w `test_fourier_bridge.py` (patrz
sekcja 🌈 niżej).

## 🌈 Most Fouriera M/S↔K — `timdr_time.fourier_bridge`

Jedyny w tym repo wyjątek od "zero identyfikacji między gałęziami" —
i jedyny uzasadniony, bo opiera się na ustalonej, 200-letniej
matematyce (FFT), nie na nowej hipotezie. Powód: dualizm
falowo-cząsteczkowy fotonu ma dokładne źródło matematyczne — zasada
nieoznaczoności Heisenberga ma dokładnie tę samą strukturę co
klasyczna zasada nieoznaczoności Gabora dla sygnałów. `x(t)` z M/S
(dokładna lokalizacja w czasie — "cząstka") i modalność `(f,φ,A)` z K
(dokładna lokalizacja w częstotliwości, zerowa w czasie — "fala") są
połączone konkretną, znaną transformatą (FFT), nie utożsamione.

```python
from timdr_time import fft_modalities, time_bandwidth_product
import numpy as np

t = np.arange(8)
x = 3.0 * np.cos(2*np.pi*0.25*t + 0.5)     # czysty ton na jednym binie FFT
modalities, dc_offset = fft_modalities(x, dt=1.0)
print(modalities[1])   # Modality(f=0.25, phi≈2.07, A≈3.0) -- odtwarza sygnal dokladnie

pulse = np.exp(-((t-4.0)**2)/(2*1.5**2))
print(time_bandwidth_product(pulse, dt=1.0, include_dc=True))  # ~1/(4*pi) dla impulsu gaussowskiego
```

**Wyprowadzenie `Δt·Δf=1/(4π)` dla impulsu gaussowskiego** (ręczne, w
tej sesji, nie przepisane z pamięci) jest w nagłówku
`fourier_bridge.py` — niezależne od σ, dokładnie granica
Gabora/Heisenberga OSIĄGANA (nie tylko spełniona) przez impuls
gaussowski.

**✅ Zweryfikowane.** `pytest tests/test_fourier_bridge.py -v` — 10/10
testów przeszło (część jednotonowa dokładna algebrą + część
gaussowska w szerokich tolerancjach ±30%, oba typy testów zielone).

## ⚠️ Czego to NIE robi

`chronoprocess.py` nie definiuje żadnego nowego obiektu matematycznego,
nie "dowodzi", że czas jest jednym spójnym bytem w M/S+G+K — to jest
orkiestracja trzech osobnych, przetestowanych (hand-traced) narzędzi
na wspólnym nośniku, nic więcej. Nie zawiera żadnej logiki spinającej
wyniki jednej gałęzi z drugą (np. nie ma tu żadnej funkcji, która
bierze wynik `x_tempo()` i przekazuje go do `gamma_surface()`).

`fourier_bridge.py` jest jedynym, jawnie oznaczonym wyjątkiem od tej
zasady (patrz sekcja 🌈 wyżej) — a nawet on NIE łączy G z niczym, tylko
M/S z K, i wyłącznie przez ustaloną matematykę (FFT), nie nową
hipotezę. Nie implementuje żadnego mostu G↔M/S ani G↔K — gałąź
geometryczna zostaje całkowicie osobna, bez naturalnego kandydata na
taką transformatę.
