"""Lo que comparten los balances: el veredicto del segundo principio y los textos.

Vademecum §10.2: S_gen = ΔS_universo ≥ 0. Con S_gen > 0 el proceso es irreversible, con
S_gen = 0 reversible y con S_gen < 0 imposible: el balance de energía puede cerrar igual
(el primer principio no lo prohíbe), pero los datos violan el segundo.
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    "G",
    "S_FLOOR",
    "VERDICTS",
    "Verdict",
    "deg_c",
    "heat_direction_message",
    "num",
    "snap_sgen",
    "upper_first",
    "verdict",
    "violation_message",
]

#: Aceleración de la gravedad normal (m/s²).
G: float = 9.80665

#: Piso de la escala de entropía (J/(kg·K), del orden de un c_p) para decidir si una S_gen
#: es ruido de redondeo: la referencia de s es arbitraria y cerca de ella |s| ≈ 0.
S_FLOOR: float = 1e3

Verdict = Literal["reversible", "irreversible", "imposible"]
VERDICTS: dict[Verdict, str] = {
    "reversible": "Reversible (S_gen = 0)",
    "irreversible": "Irreversible (S_gen > 0)",
    "imposible": "Imposible (S_gen < 0)",
}


def num(x: float, fmt: str = ".4g") -> str:
    """Un número con coma decimal y el signo menos tipográfico."""
    return f"{x:{fmt}}".replace(".", ",").replace("-", "−")


def deg_c(T: float) -> str:
    return f"{num(T - 273.15)} °C"


def upper_first(text: str) -> str:
    """La primera letra en mayúscula sin tocar el resto: «el R-134a» → «El R-134a»
    (``str.capitalize`` lo dejaba «El r-134a»)."""
    return text[:1].upper() + text[1:]


def snap_sgen(S_gen: float, scale: float) -> float:
    """S_gen = 0 si es ruido de redondeo frente a ``scale`` (los términos del balance)."""
    return 0.0 if abs(S_gen) <= 1e-9 * (abs(scale) + 1e-12) else S_gen


def verdict(S_gen: float) -> Verdict:
    """Vademecum §10.2: S_gen > 0 irreversible, = 0 reversible y < 0 imposible."""
    if S_gen == 0.0:
        return "reversible"
    return "irreversible" if S_gen > 0.0 else "imposible"


def violation_message(
    sgen_text: str,
    Q: float,
    dS: float,
    T_b: float,
    T_min: float,
    T_max: float,
    who: str = "el sistema",
) -> str:
    """Por qué una S_gen < 0 hace imposible el proceso, con la pista para corregirlo.

    ``sgen_text`` es la S_gen ya escrita con su unidad; ``Q`` y ``dS``, el calor que recibe
    y el cambio de entropía (por kg o totales, en la misma base); ``T_min`` y ``T_max``, las
    temperaturas extremas de ``who`` durante el proceso. Con S_gen = ΔS − Q/T_b ≥ 0, la
    temperatura límite de la fuente (o del medio) es T* = Q/ΔS.
    """
    base = (
        f"{sgen_text} < 0: el proceso es imposible. El balance de energía cierra, pero viola "
        "el segundo principio (vademecum §10.2)."
    )
    if Q == 0.0:
        return (
            f"{base} Sin calor, la entropía no puede bajar: un proceso adiabático solo genera "
            "entropía."
        )
    if Q > 0.0:
        if dS <= 0.0:
            return f"{base} Si {who} recibe calor, su entropía no puede bajar. Revisá los datos."
        hint = (
            f"Con estos datos la fuente tendría que estar a más de {deg_c(Q / dS)} "
            f"(T* = Q/ΔS), y está a {deg_c(T_b)}."
        )
        if T_b < T_max:
            hint += (
                f" Además es más fría que {who} (llega a {deg_c(T_max)}): el calor no pasa "
                "solo de frío a caliente."
            )
        return f"{base} {hint} Subí la temperatura de la fuente T_b."
    if dS >= 0.0:  # no pasa: con Q < 0 y ΔS ≥ 0, S_gen > 0
        return f"{base} Revisá los datos."
    hint = (
        f"Con estos datos el medio tendría que estar a menos de {deg_c(Q / dS)} "
        f"(T* = Q/ΔS), y está a {deg_c(T_b)}."
    )
    if T_b > T_min:
        hint += f" Además es más caliente que {who} (baja a {deg_c(T_min)})."
    return (
        f"{base} {hint} Bajá la temperatura del medio T_b, o revisá el estado final: con ese "
        "calor no se llega."
    )


def heat_direction_message(
    Q: float, T_b: float, T_min: float, T_max: float, who: str = "el sistema"
) -> str | None:
    """Clausius (vademecum §9.1): el calor no pasa solo de frío a caliente.

    Si ``who`` está siempre más caliente que la fuente de la que recibe calor (o siempre más
    frío que el medio al que lo cede), el proceso es imposible aunque la S_gen total salga
    positiva: la irreversibilidad interna (una mezcla, un estrangulamiento) no compensa el
    calor que va al revés.
    """
    if Q > 0.0 and T_b < T_min:
        return (
            f"El proceso es imposible (vademecum §9.1): el calor entra desde una fuente a "
            f"{deg_c(T_b)} y {who} está siempre más caliente (entre {deg_c(T_min)} y "
            f"{deg_c(T_max)}): el calor no pasa solo de frío a caliente. Subí la temperatura "
            "de la fuente T_b."
        )
    if Q < 0.0 and T_b > T_max:
        return (
            f"El proceso es imposible (vademecum §9.1): el calor sale hacia un medio a "
            f"{deg_c(T_b)} y {who} está siempre más frío (entre {deg_c(T_min)} y "
            f"{deg_c(T_max)}): el calor no pasa solo de frío a caliente. Bajá la temperatura "
            "del medio T_b."
        )
    return None
