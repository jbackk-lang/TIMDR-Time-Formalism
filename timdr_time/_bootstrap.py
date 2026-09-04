"""
timdr_time/_bootstrap.py

Wstrzykuje sciezki do trzech repo-siostr (TIMDR-Math-Formalism,
TIMDR-Geometry-Formalism, TIMDR-Modal-Formalism) do sys.path, PRZED
jakimkolwiek importem z nich w chronoprocess.py. Zaklada uklad
katalogow: wszystkie cztery repo jako rodzenstwo pod tym samym
katalogiem nadrzednym -- dokladnie tak, jak sa dzis rozlozone
(C:\\Users\\jback\\Downloads\\a\\TIMDR-*-Formalism).

Ten modul nigdy nie rzuca wyjatku sam z siebie -- jesli dany katalog
nie istnieje, jest po cichu pomijany. Brak sciezki na sys.path i tak
da czytelny, wlasny ImportError w chronoprocess.py (patrz tam), wiec
nie ma potrzeby duplikowac ostrzezenia tutaj.
"""
import os
import sys

_THIS_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT_DIR = os.path.dirname(_THIS_REPO_ROOT)
_SIBLING_REPO_NAMES = (
    "TIMDR-Math-Formalism",
    "TIMDR-Geometry-Formalism",
    "TIMDR-Modal-Formalism",
)

for _name in _SIBLING_REPO_NAMES:
    _path = os.path.join(_PARENT_DIR, _name)
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)
