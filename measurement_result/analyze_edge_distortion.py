#!/usr/bin/env python3
"""Compare paired coin images; distinguish a position effect from lens calibration.

Run: /usr/bin/python3 CV_lab/Pictures/measurement_results/analyze_edge_distortion.py
Dependencies: numpy, OpenCV, matplotlib. Original images/measurements are read only.
"""

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/coin-edge-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT = Path(__file__).resolve().parent
PICTURES = OUTPUT.parent
MARKER = "VERGLEICH DER MESSREIHEN _1 UND _2 / BILDRANDANALYSE"
TARGET = PICTURES / "Coins_measurements_measured.txt"
LEGACY = PICTURES / "Coins_masurements_measured.txt"
THRESHOLDS = list(range(140, 181, 5))


def number(value, places=3, signed=False):
    return format(value, ("+" if signed else "") + f".{places}f").replace(".", ",")


def edge_profile(row):
    image = cv2.imread(str(PICTURES / "Single Coins" / row["file"]))
    gray = cv2.GaussianBlur(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 0.8
    )
    cx, cy = row["seed_center_xy"]
    theta = np.arange(4096) * (2 * np.pi / 4096)
    radius = np.arange(row["ellipse_minor_px"] / 2 - 40,
                       row["ellipse_major_px"] / 2 + 40, 0.125)
    x = (cx + np.cos(theta)[:, None] * radius).astype(np.float32)
    y = (cy + np.sin(theta)[:, None] * radius).astype(np.float32)
    values = cv2.remap(gray, x, y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return theta, radius, values, cx, cy, image.shape[1], image.shape[0]


def diameter_at_threshold(profile, threshold):
    theta, radii, values, cx, cy, width, height = profile
    level = np.broadcast_to(np.asarray(threshold).reshape(-1, 1), (len(theta), 1))
    crossing = (values[:, :-1] <= level) & (values[:, 1:] > level)
    if not np.all(crossing.any(axis=1)):
        raise ValueError("Nicht alle Strahlen haben einen gültigen äußeren Übergang")
    index = crossing.shape[1] - 1 - np.argmax(crossing[:, ::-1], axis=1)
    ray = np.arange(len(theta))
    low, high = values[ray, index], values[ray, index + 1]
    radius = radii[index] + (level[:, 0] - low) / (high - low) * 0.125
    points = np.column_stack((cx + radius * np.cos(theta),
                              cy + radius * np.sin(theta))).astype(np.float32)
    if (points < 0).any() or (points[:, 0] >= width).any() or (
        points[:, 1] >= height
    ).any():
        raise ValueError("Kontrollkontur liegt außerhalb des Bildes")
    hull = cv2.convexHull(points).reshape(-1, 2).astype(np.float64)
    distances = ((hull[:, None] - hull[None, :]) ** 2).sum(axis=2)
    axes = cv2.fitEllipse(points.reshape(-1, 1, 2))[1]
    return {"feret_px": float(np.sqrt(distances.max())),
            "ellipse_mean_px": float(sum(axes) / 2)}, index


def controls(row):
    profile = edge_profile(row)
    values = profile[2]
    original, index = diameter_at_threshold(profile, row["otsu_threshold"])
    if abs(original["feret_px"] - row["diameter_px"]) > 0.05:
        raise ValueError(f'Kontrollmessung weicht unerwartet ab: {row["file"]}')
    ray = np.arange(len(index))[:, None]
    # Local contrast control: sample each plateau 6-12 pixels from the Otsu edge.
    inside = np.median(values[ray, index[:, None] + np.arange(-96, -47)], axis=1)
    outside = np.median(values[ray, index[:, None] + np.arange(48, 97)], axis=1)
    if np.min(outside - inside) <= 0:
        raise ValueError("Ungültiger lokaler Kontrast")
    local_level = (inside + outside) / 2
    local, _ = diameter_at_threshold(profile, local_level)
    return {
        "original_recheck": original,
        "shared_thresholds": {
            str(t): diameter_at_threshold(profile, t)[0] for t in THRESHOLDS
        },
        "local_contrast_midpoint": local,
        "local_threshold_mean": float(local_level.mean()),
    }


def paired_stats(rows, control):
    pairs = []
    coins = list(dict.fromkeys(row["coin"] for row in rows if row["coin"] != "Referenz"))
    for coin in coins:
        a, b = sorted((row for row in rows if row["coin"] == coin),
                      key=lambda row: row["file"])
        center = np.array([(a["image_width_px"] - 1) / 2,
                           (a["image_height_px"] - 1) / 2])
        d1, d2 = a["diameter_px"], b["diameter_px"]
        e1, e2 = [(r["ellipse_minor_px"] + r["ellipse_major_px"]) / 2 for r in (a, b)]
        local1 = control[a["file"]]["local_contrast_midpoint"]
        local2 = control[b["file"]]["local_contrast_midpoint"]
        pairs.append({
            "coin": coin, "file_1": a["file"], "file_2": b["file"],
            "center_1_xy": a["seed_center_xy"], "center_2_xy": b["seed_center_xy"],
            "radius_1_px": float(np.linalg.norm(a["seed_center_xy"] - center)),
            "radius_2_px": float(np.linalg.norm(b["seed_center_xy"] - center)),
            "diameter_1_px": d1, "diameter_2_px": d2,
            "delta_feret_px": d2 - d1, "delta_feret_percent": 100 * (d2 / d1 - 1),
            "delta_ellipse_px": e2 - e1, "delta_ellipse_percent": 100 * (e2 / e1 - 1),
            "local_delta_feret_percent": 100 * (local2["feret_px"] / local1["feret_px"] - 1),
            "local_delta_ellipse_percent": 100 * (
                local2["ellipse_mean_px"] / local1["ellipse_mean_px"] - 1
            ),
        })
    stats = {}
    for key in ("delta_feret_px", "delta_feret_percent", "delta_ellipse_px",
                "delta_ellipse_percent", "local_delta_feret_percent",
                "local_delta_ellipse_percent"):
        array = np.array([p[key] for p in pairs])
        stats[key] = {"mean": float(array.mean()), "median": float(np.median(array)),
                      "sample_sd": float(array.std(ddof=1)),
                      "negative_pairs": int((array < 0).sum()),
                      "minimum": float(array.min()), "maximum": float(array.max())}
    radius_squared = np.array([p["radius_2_px"] ** 2 for p in pairs])
    stats["pearson_r_radius_squared_ellipse_delta"] = float(np.corrcoef(
        radius_squared, [p["delta_ellipse_percent"] for p in pairs]
    )[0, 1])
    stats["shape_anisotropy_percent"] = {
        suffix: float(np.mean([
            100 * (r["ellipse_major_px"] / r["ellipse_minor_px"] - 1)
            for r in rows if r["file"].endswith(suffix) and r["coin"] != "Referenz"
        ])) for suffix in ("_1.bmp", "_2.bmp")
    }
    sweep = []
    for threshold in THRESHOLDS:
        item = {"threshold": threshold}
        for feature in ("feret_px", "ellipse_mean_px"):
            changes = [100 * (
                control[p["file_2"]]["shared_thresholds"][str(threshold)][feature] /
                control[p["file_1"]]["shared_thresholds"][str(threshold)][feature] - 1
            ) for p in pairs]
            item[feature + "_mean_delta_percent"] = float(np.mean(changes))
        sweep.append(item)
    return pairs, stats, sweep


def plot(pairs, stats, sweep):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.6), constrained_layout=True)
    blue, orange = "#2268a5", "#cf6b17"
    ax = axs[0, 0]
    label_positions = {
        "0,01€": (520, 190), "0,02€": (520, 285), "0,05€": (520, 390),
        "0,10€": (520, 710), "0,20€": (520, 1470), "0,50€": (520, 485),
        "1€": (520, 865), "2€": (520, 1035),
    }
    for p in pairs:
        a, b = p["center_1_xy"], p["center_2_xy"]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#cdd3db", lw=0.8)
        ax.scatter(*b, color=orange, s=28)
        ax.annotate(p["coin"], b, xytext=label_positions[p["coin"]], fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.6})
    a = np.array([p["center_1_xy"] for p in pairs])
    ax.scatter(a[:, 0], a[:, 1], color=blue, label="Reihe _1: zentral", s=24)
    ax.scatter([], [], color=orange, label="Reihe _2: links am Rand")
    ax.scatter([1295.5], [971.5], marker="+", s=90, color="black")
    ax.set(xlim=(0, 2591), ylim=(1943, 0), xlabel="x [Pixel]", ylabel="y [Pixel]",
           title="A  Aufnahmeorte im Bild")
    ax.legend(loc="upper right", fontsize=9)
    ax = axs[0, 1]
    x = np.arange(len(pairs))
    ax.bar(x - 0.18, [p["delta_feret_percent"] for p in pairs], width=0.35,
           label="Äußerer Feret-Durchmesser", color=blue)
    ax.bar(x + 0.18, [p["delta_ellipse_percent"] for p in pairs], width=0.35,
           label="Mittel der Ellipsenachsen", color=orange)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [p["coin"] for p in pairs])
    ax.set(ylabel="(_2 / _1 − 1) × 100 [%]", title="B  Unterschiede je Münzwert")
    ax.legend(fontsize=8, loc="lower right")
    ax = axs[1, 0]
    for p in pairs:
        ax.scatter(p["radius_2_px"], p["delta_ellipse_percent"], color=orange, s=35)
        offset = (6, -14) if p["coin"] == "0,02€" else (5, 5)
        ax.annotate(p["coin"], (p["radius_2_px"], p["delta_ellipse_percent"]),
                    xytext=offset, textcoords="offset points", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    r = stats["pearson_r_radius_squared_ellipse_delta"]
    ax.set(xlim=(880, 1340), ylim=(-0.40, 0.06), xlabel="Abstand der _2-Münzmitte zur Bildmitte [px]",
           ylabel="Änderung des Ellipsen-Mittels [%]",
           title=f"C  Kein klarer Verlauf mit dem Radius (r für Radius² = {r:.2f})")
    ax = axs[1, 1]
    for key, name, color in [("feret_px", "Feret: gemeinsamer Schwellenwert", blue),
                             ("ellipse_mean_px", "Ellipse: gemeinsamer Schwellenwert", orange)]:
        ax.plot([s["threshold"] for s in sweep],
                [s[key + "_mean_delta_percent"] for s in sweep],
                "o-", ms=4, label=name, color=color)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="Identische Grauwertschwelle T in allen Bildern",
           ylabel="Mittlere Änderung _2 gegenüber _1 [%]",
           title="D  Abhängigkeit von der Randbestimmung")
    ax.legend(fontsize=8)
    for ax in axs.flat:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Münzmessungen: kleine Größenabweichung am linken Bildrand", fontsize=16)
    fig.supxlabel("8 Bildpaare · ursprüngliche Pixelkoordinaten · keine Linsenkalibrierung", fontsize=10)
    fig.savefig(OUTPUT / "edge_distortion_analysis.png", dpi=180)
    plt.close(fig)


def report(pairs, stats, sweep):
    f = stats["delta_feret_percent"]
    lines = [
        MARKER, "=" * len(MARKER), "Auswertung: 28.10.2026", "",
        "Kurzbefund", "----------",
        "Ja, zwischen _1 und _2 gibt es einen kleinen gemessenen Größenunterschied.",
        f'Der äußere Durchmesser ist in {f["negative_pairs"]} von 8 Paaren am linken Rand kleiner.',
        f'Mittelwert: {number(stats["delta_feret_px"]["mean"], 3, True)} px bzw. '
        f'{number(f["mean"], 3, True)} %.',
        "Dies ist ein Hinweis auf eine positionsabhängige Größen-/Messabweichung.",
        "Eine ausschließlich durch die Linse verursachte Verzeichnung ist mit diesen",
        "Aufnahmen nicht eindeutig nachgewiesen. Die Wahl des Bildrandschwellenwerts",
        "hat einen Einfluss derselben Größenordnung. Eine optische Mitursache bleibt möglich.",
        "", "1. Paarvergleich der ursprünglichen Messwerte", "-" * 46,
        "Differenz = D(_2) - D(_1); relativ = 100 * (D(_2) / D(_1) - 1).",
        "Negative Werte bedeuten: kleinere Abbildung in _2. Berechnung aus den",
        "ungerundeten JSON-Werten; zusätzliche Stellen dienen dem Zahlenvergleich.", "",
        f'{"Münze":<10} {"D_1 [px]":>11} {"D_2 [px]":>11} {"Delta [px]":>12} {"Delta [%]":>12}',
        "-" * 61,
    ]
    for p in pairs:
        lines.append(f'{p["coin"]:<10} {number(p["diameter_1_px"], 2):>11} '
                     f'{number(p["diameter_2_px"], 2):>11} '
                     f'{number(p["delta_feret_px"], 2, True):>12} '
                     f'{number(p["delta_feret_percent"], 3, True):>12}')
    lines += [
        "", f'Median der relativen Änderungen: {number(f["median"], 3, True)} %.',
        f'Streuung über die acht Paare (Stichproben-SD): {number(f["sample_sd"], 3)} Prozentpunkte.',
        "Dies ist eine deskriptive Streuung, keine Wiederholgenauigkeit oder Messunsicherheit.",
        "Größte Abnahme: 20 Cent mit -1,93 px bzw. -0,308 %.",
        "Ausnahme: 2 Cent mit +0,72 px bzw. +0,136 % im äußeren Maximaldurchmesser.",
        "Die Referenz hat keine _2-Aufnahme und wird nicht als weiteres Paar gewertet.",
        "Der Mittelwert ist das ungewichtete Mittel der acht relativen Paaränderungen.",
        "", "2. Zusammenhang mit der Bildposition", "-" * 39,
        "Als geometrische Bildmitte wird (1295,5; 971,5) px verwendet.",
        "Sie ist eine Näherung und kein kalibrierter optischer Hauptpunkt.",
        "r bezeichnet den Abstand der Münzmitte von dieser Bildmitte.", "",
        f'{"Münze":<10} {"x_2 [px]":>11} {"y_2 [px]":>11} {"r_1 [px]":>11} {"r_2 [px]":>11}',
        "-" * 60,
    ]
    for p in pairs:
        lines.append(f'{p["coin"]:<10} {number(p["center_2_xy"][0], 1):>11} '
                     f'{number(p["center_2_xy"][1], 1):>11} '
                     f'{number(p["radius_1_px"], 1):>11} {number(p["radius_2_px"], 1):>11}')
    lines += [
        "", "_1 liegt zentral (r etwa 4 bis 43 px); _2 liegt links außen (r etwa 920 bis 1284 px).",
        "Es wurden somit Zentrum und linker Rand verglichen. Für den rechten Rand",
        "und ein flächendeckendes Verzeichnungsfeld fehlen entsprechende Einzelbilder.",
        "Innerhalb der Randaufnahmen wird die Abnahme nicht eindeutig größer, wenn r wächst.",
        f'Korrelation zwischen r_2² und relativer Änderung des Ellipsen-Mittels: '
        f'{number(stats["pearson_r_radius_squared_ellipse_delta"], 3, True)} (nur 8 Paare).',
        "Daraus lässt sich keine belastbare radiale Verzeichnungskurve bestimmen.",
        "", "3. Kontrollen der Messmethode", "-" * 30,
        "a) Ellipsen-Mittel = (kleine Achse + große Achse) / 2.",
        "   Dieses Vergleichsmaß nutzt den ganzen Rand und reagiert weniger auf",
        "   einzelne Spitzen als der maximale Feret-Durchmesser. Es ist bei der",
        "   gekerbten 20-Cent-Münze kein nomineller physischer Außendurchmesser.",
        f'   Mittlere Änderung: {number(stats["delta_ellipse_px"]["mean"], 3, True)} px / '
        f'{number(stats["delta_ellipse_percent"]["mean"], 3, True)} %.',
        "   Alle acht Vorzeichen sind negativ. Bei 2 Cent beträgt die Differenz",
        "   jedoch nur -0,057 px und ist allein kein belastbarer Einzelbefund.",
        "", "b) Lokale Kontrastkontrolle:",
        "   Je radialem Profil werden Innen- und Hintergrundhelligkeit 6-12 px",
        "   vom ursprünglichen Rand entfernt geschätzt. Ihr Mittelwert ersetzt",
        "   die globale Schwelle. Danach werden Feret und Ellipsen-Mittel neu bestimmt.",
        f'   Mittlere Änderung Feret: {number(stats["local_delta_feret_percent"]["mean"], 3, True)} %;',
        f'   mittlere Änderung Ellipsen-Mittel: {number(stats["local_delta_ellipse_percent"]["mean"], 3, True)} %.',
        "   Das Ellipsen-Mittel bleibt in allen Paaren kleiner; die 2-Cent-Differenz",
        "   liegt dabei praktisch bei null. Beim Maximaldurchmesser wechseln",
        "   einzelne Vorzeichen. Ein einzelner Randpunkt kann das Maximum beeinflussen.",
        "", "c) Gemeinsame absolute Grauwertschwelle in allen Bildern:",
        "   Es werden dieselben Schwellen T=140,145,...,180 auf alle Bilder angewendet.",
        "   Die nachfolgende Spanne ist ein Methodenvergleich, kein Konfidenzintervall.",
        "   Der gleiche absolute Schwellwert garantiert bei unterschiedlicher lokaler",
        "   Helligkeit noch keine identische physische Randlage.", "",
        f'{"T":>6} {"Mittel Delta Feret [%]":>25} {"Mittel Delta Ellipse [%]":>27}',
        "-" * 62,
    ]
    for s in sweep:
        lines.append(f'{s["threshold"]:>6} '
                     f'{number(s["feret_px_mean_delta_percent"], 3, True):>25} '
                     f'{number(s["ellipse_mean_px_mean_delta_percent"], 3, True):>27}')
    anisotropy = stats["shape_anisotropy_percent"]
    lines += [
        "", "   Der mittlere Feret-Effekt kann mit der Schwelle sogar sein Vorzeichen wechseln.",
        "   Die negative Änderung des Ellipsen-Mittels bleibt als Tendenz erhalten,",
        "   ihre Größe hängt aber ebenfalls von der Randdefinition ab.",
        "", "d) Formabweichung:",
        "   Mittleres Achsenverhältnis, ausgedrückt als 100*(groß/klein-1):",
        f'   _1: {number(anisotropy["_1.bmp"], 3)} %; '
        f'_2: {number(anisotropy["_2.bmp"], 3)} %.',
        "   Die Münzen bleiben annähernd kreisförmig. Es zeigt sich keine starke,",
        "   über alle Münzen konsistente zusätzliche Elliptizität am Rand.",
        "   Das schließt eine kleine geometrische Verzeichnung nicht aus.",
        "", "4. Einordnung und Grenzen", "-" * 25,
        "Der Datensatz zeigt eine kleine positionsassoziierte Größen-/Messabweichung",
        "von ungefähr 0,1 bis 0,2 % im Mittel, abhängig von der Messdefinition.",
        "Dies ist mit einer schwachen Randverzeichnung vereinbar, identifiziert aber",
        "deren Ursache und Typ nicht eindeutig. Insbesondere sind Tonnen- oder",
        "Kissenverzeichnung und Verzeichnungskoeffizienten hieraus nicht sicher bestimmbar.",
        "", "Alternative beziehungsweise zusätzliche Einflüsse sind Beleuchtung und",
        "Farbsäume am Objektrand, Schwellenwahl, Münzseite/-drehung, perspektivischer",
        "Maßstab bei einer nicht exakt parallelen Messebene, Objekt-/Kamerabewegung",
        "und mögliche Einstellungs- oder Höhenänderungen. Für jede Münze liegt pro",
        "Position nur ein Bild vor; diese Einflüsse sind nicht durch Wiederholungen getrennt.",
        "", "Zur Größenordnung: Mit der zentralen 25-mm-Referenz entsprechen rund 28,16 px",
        "einem Millimeter. Die mittlere Feret-Abnahme von etwa 1,01 px entspricht",
        "rechnerisch etwa 0,036 mm, die größte beobachtete Abnahme etwa 0,069 mm.",
        "Das dient nur zur Orientierung und ist keine korrigierte metrische Kalibrierung.",
        "", "5. Nächster Schritt für einen belastbaren Verzeichnungsnachweis", "-" * 60,
        "- Dieselbe flache Referenz bei unverändertem Fokus, Abstand, Belichtung und",
        "  Orientierung auf einem Raster über die gesamte Messebene aufnehmen.",
        "- Mitte, alle vier Ränder und Ecken abdecken; pro Position mindestens drei",
        "  unabhängige Wiederholungen erstellen und die Referenzhöhe konstant halten.",
        "- Zusätzlich ein Muster mit bekannten Maßen in mehreren Lagen aufnehmen",
        "  und Kameramatrix sowie Verzeichnungsparameter kalibrieren. Die metrische",
        "  Ebenenabbildung separat prüfen, damit Perspektive nicht als Linsenfehler gilt.",
        "- Positionsabhängige Maßabweichung vor und nach der Korrektur auf unabhängigen",
        "  Kontrollaufnahmen vergleichen. Erst danach eine Ortskorrektur übernehmen.",
        "", "Die bestehenden Pixeldurchmesser und Mittelwerte wurden nicht nachträglich",
        "korrigiert. Ein globaler Korrekturfaktor aus diesen acht Paaren wäre nicht belegt.",
        "", "Abbildung: measurement_results/edge_distortion_analysis.png",
        "Ungerundete Paare und Kontrollmessungen: measurement_results/edge_distortion_analysis.json",
        "Reproduzieren: /usr/bin/python3 CV_lab/Pictures/measurement_results/analyze_edge_distortion.py",
        "Kalibrierungsgrundlage (OpenCV, radiale/tangentiale Modelle und Messmuster):",
        "https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html",
        "https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html",
    ]
    return "\n".join(lines) + "\n"


def main():
    data = json.loads((OUTPUT / "measurements.json").read_text())
    rows = data["single_images"]
    if hashlib.sha256((PICTURES / "Coins_masurements.txt").read_bytes()).hexdigest() != (
        data["original_measurement_file_sha256"]
    ):
        raise ValueError("Ursprüngliche Messliste stimmt nicht mit der Messgrundlage überein")
    for row in rows:
        if hashlib.sha256((PICTURES / "Single Coins" / row["file"]).read_bytes()).hexdigest() != (
            row["source_sha256"]
        ):
            raise ValueError(f'Bild geändert, Messung bitte erneuern: {row["file"]}')
    control = {r["file"]: controls(r) for r in rows if r["coin"] != "Referenz"}
    pairs, stats, sweep = paired_stats(rows, control)
    plot(pairs, stats, sweep)
    text = report(pairs, stats, sweep)
    existing = TARGET if TARGET.exists() else LEGACY
    base = existing.read_text().split(MARKER, 1)[0].rstrip()
    combined = base + "\n\n\n" + text
    for path in (TARGET, LEGACY):
        path.write_text(combined, encoding="utf-8")
    payload = {"pairs": pairs, "summary": stats, "common_threshold_sweep": sweep,
               "image_controls": control,
               "interpretation": "small_position_associated_bias_not_unique_lens_distortion_proof",
               "position_origin": "geometric_image_center_not_calibrated_principal_point"}
    (OUTPUT / "edge_distortion_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Verglichen: {len(pairs)} Bildpaare")
    print(f"Dokumentation: {TARGET}")
    print("Bisherige Dateinamensvariante ebenfalls aktualisiert.")


if __name__ == "__main__":
    main()
