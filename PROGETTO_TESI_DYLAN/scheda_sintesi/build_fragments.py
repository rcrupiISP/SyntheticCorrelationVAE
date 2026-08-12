# -*- coding: utf-8 -*-
"""Converte gli abstract in testo semplice (abstract_*.txt) in frammenti LaTeX
(abstract_*.tex) e stampa il conteggio dei caratteri.

Il testo *autoritativo* e' quello dei .txt: e' quello che va incollato nel
portale di consegna ed e' quello su cui si misura il limite di 3550 caratteri.
I .tex sono solo la resa tipografica dello stesso identico contenuto.
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LIMIT = 3550

# ordine importante: la barra rovesciata va sostituita per prima
ESCAPES = [
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
]

# caratteri tipografici unicode -> comandi LaTeX equivalenti
TYPOGRAPHY = [
    ("—", "---"),   # em dash
    ("–", "--"),    # en dash
    ("’", "'"),     # apostrofo tipografico
    ("“", "``"),
    ("”", "''"),
]


def to_latex(text):
    for src, dst in ESCAPES:
        text = text.replace(src, dst)
    for src, dst in TYPOGRAPHY:
        text = text.replace(src, dst)
    return text


def main():
    for lang in ("it", "en"):
        src = os.path.join(HERE, "abstract_%s.txt" % lang)
        dst = os.path.join(HERE, "abstract_%s.tex" % lang)
        raw = io.open(src, encoding="utf-8").read().strip()

        n = len(raw)
        status = "OK" if n <= LIMIT else "SFORATO"
        print("abstract_%s: %d caratteri (spazi inclusi) / %d  -> %s (margine %d)"
              % (lang, n, LIMIT, status, LIMIT - n))

        body = "\n\n".join(to_latex(p.strip()) for p in raw.split("\n\n") if p.strip())
        io.open(dst, "w", encoding="utf-8").write(body + "\n")

        # il conteggio va anche in un file, cosi' il .tex lo puo' stampare
        io.open(os.path.join(HERE, "count_%s.tex" % lang), "w",
                encoding="utf-8").write("%d" % n)


if __name__ == "__main__":
    main()
