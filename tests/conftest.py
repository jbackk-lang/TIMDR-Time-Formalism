import os
import sys

# Repo nie jest zainstalowanym pakietem — dorzuć katalog główny do
# sys.path, żeby `import timdr_time` działało niezależnie od tego,
# skąd odpalane jest pytest. Wstrzyknięcie ścieżek do repo-sióstr
# (TIMDR-Math/Geometry/Modal-Formalism) dzieje się osobno, w
# timdr_time/_bootstrap.py, uruchamianym automatycznie przy imporcie
# pakietu timdr_time — nie trzeba go duplikować tutaj.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
