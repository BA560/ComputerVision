#!/usr/bin/env python3
"""Measure native-image coin silhouettes and create an auditable copy of the list.

Run with /usr/bin/python3; requires numpy, OpenCV and Pillow.
The source BMPs and Coins_masurements.txt are read only.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PICTURES = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent
ANGLE_COUNT = 4096
RADIAL_STEP = 0.125
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def decimal(value, digits=1):
    return f"{value:.{digits}f}".replace(".", ",")


def segment(gray, minimum_area):
    threshold, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return threshold, [c for c in contours if cv2.contourArea(c) > minimum_area]


def measure(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Bild nicht lesbar: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0.8)
    threshold, contours = segment(np.rint(smoothed).astype(np.uint8), 500)
    if len(contours) != 1:
        raise ValueError(f"{path.name}: {len(contours)} statt eines Einzelobjekts")
    contour = contours[0]
    x, y, width, height = cv2.boundingRect(contour)
    margin = min(x, y, image.shape[1] - x - width, image.shape[0] - y - height)
    if margin <= 0:
        raise ValueError(f"{path.name}: Objekt berührt den Bildrand")

    # The ellipse supplies only a seed center and search band, not the diameter.
    (cx, cy), axes, _ = cv2.fitEllipse(contour)
    theta = np.arange(ANGLE_COUNT) * (2 * np.pi / ANGLE_COUNT)
    radii = np.arange(min(axes) / 2 - 40, max(axes) / 2 + 40, RADIAL_STEP)
    map_x = (cx + np.cos(theta)[:, None] * radii).astype(np.float32)
    map_y = (cy + np.sin(theta)[:, None] * radii).astype(np.float32)
    profiles = cv2.remap(
        smoothed, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    measurements = []
    for level in (threshold - 15, threshold, threshold + 15):
        transitions = (profiles[:, :-1] <= level) & (profiles[:, 1:] > level)
        if not np.all(np.any(transitions, axis=1)):
            raise ValueError(f"{path.name}: Außenkante nicht für alle Winkel gefunden")
        # Choose the outer silhouette, not an engraving or inner ring.
        index = transitions.shape[1] - 1 - np.argmax(transitions[:, ::-1], axis=1)
        ray = np.arange(ANGLE_COUNT)
        low = profiles[ray, index]
        high = profiles[ray, index + 1]
        radius = radii[index] + (level - low) / (high - low) * RADIAL_STEP
        points = np.column_stack(
            (cx + radius * np.cos(theta), cy + radius * np.sin(theta))
        ).astype(np.float32)
        if (points < 0).any() or (points[:, 0] >= image.shape[1]).any() or (
            points[:, 1] >= image.shape[0]
        ).any():
            raise ValueError(f"{path.name}: interpolierter Umriss außerhalb des Bildes")
        hull = cv2.convexHull(points).reshape(-1, 2).astype(np.float64)
        squared_distance = np.sum((hull[:, None] - hull[None, :]) ** 2, axis=2)
        first, second = np.unravel_index(
            np.argmax(squared_distance), squared_distance.shape
        )
        fitted_axes = sorted(cv2.fitEllipse(points.reshape(-1, 1, 2))[1])
        measurements.append({
            "diameter_px": float(np.sqrt(squared_distance[first, second])),
            "ellipse_minor_px": float(fitted_axes[0]),
            "ellipse_major_px": float(fitted_axes[1]),
            "width_px": float(np.ptp(points[:, 0])),
            "height_px": float(np.ptp(points[:, 1])),
            "endpoints_xy": hull[[first, second]].tolist(),
            "boundary": points,
        })
    main = measurements[1]
    key = path.stem.rsplit("_", 1)[0].replace("_", ",")
    row = {
        "file": path.name,
        "coin": key,
        "image_width_px": image.shape[1],
        "image_height_px": image.shape[0],
        "otsu_threshold": float(threshold),
        "bbox_xywh": [x, y, width, height],
        "minimum_image_border_distance_px": margin,
        "seed_center_xy": [cx, cy],
        **{k: v for k, v in main.items() if k != "boundary"},
        "diameter_threshold_minus_15_px": measurements[0]["diameter_px"],
        "diameter_threshold_plus_15_px": measurements[2]["diameter_px"],
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return row, main["boundary"]


def single_overviews(rows, boundaries):
    font = ImageFont.truetype(FONT_PATH, 17)
    tile_width, tile_height = 580, 650
    for page, start in enumerate(range(0, len(rows), 9), 1):
        sheet = Image.new("RGB", (3 * tile_width, 3 * tile_height), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(rows[start:start + 9]):
            image = cv2.imread(str(PICTURES / "Single Coins" / row["file"]))
            outline = np.rint(boundaries[row["file"]]).astype(np.int32)
            cv2.polylines(image, [outline.reshape(-1, 1, 2)], True, (0, 220, 0), 2,
                          cv2.LINE_AA)
            first, second = np.rint(row["endpoints_xy"]).astype(int)
            cv2.line(image, tuple(first), tuple(second), (255, 60, 255), 2, cv2.LINE_AA)
            for point in (first, second):
                cv2.circle(image, tuple(point), 5, (0, 0, 255), -1, cv2.LINE_AA)
            x, y, width, height = row["bbox_xywh"]
            crop = image[max(0, y - 22):y + height + 22,
                         max(0, x - 22):x + width + 22]
            crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crop.thumbnail((tile_width - 16, tile_height - 85))
            ox = index % 3 * tile_width
            oy = index // 3 * tile_height
            sheet.paste(crop, (ox + (tile_width - crop.width) // 2, oy + 75))
            lines = [row["file"],
                     f'D max = {decimal(row["diameter_px"])} px',
                     "grün: Rand; magenta: äußerer Durchmesser"]
            for line_number, line in enumerate(lines):
                draw.text((ox + 8, oy + 8 + line_number * 22), line,
                          fill="black", font=font)
        sheet.save(OUTPUT / f"measurement_overview_{page}.jpg", quality=95)


def analyze_multiple():
    rows = []
    font = ImageFont.truetype(FONT_PATH, 17)
    for group, expected_count in [(1, 3), (2, 5), (3, 4), (4, 4)]:
        sheet = Image.new("RGB", (4 * 490, 4 * 470), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        paths = sorted((PICTURES / "Multiple Coins" / str(group)).glob("*.bmp"),
                       key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        for index, path in enumerate(paths):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Bild nicht lesbar: {path}")
            gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0.8)
            threshold, contours = segment(gray, 10000)
            if not contours:
                raise ValueError(f"Kein Vordergrund in {path}")
            boxes = [cv2.boundingRect(c) for c in contours]
            touches = any(x == 0 or y == 0 or x + w == image.shape[1]
                          or y + h == image.shape[0] for x, y, w, h in boxes)
            rows.append({
                "file": str(path.relative_to(PICTURES)), "group": group,
                "documented_coin_count": expected_count,
                "foreground_components": len(contours),
                "touches_image_border": touches, "threshold": float(threshold),
                "image_width_px": image.shape[1], "image_height_px": image.shape[0],
            })
            x, y, width, height = cv2.boundingRect(np.concatenate(contours))
            crop = image[max(0, y - 10):y + height + 10,
                         max(0, x - 10):x + width + 10]
            crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crop.thumbnail((474, 415))
            ox, oy = index % 4 * 490, index // 4 * 470
            sheet.paste(crop, (ox + (490 - crop.width) // 2, oy + 48))
            draw.text((ox + 8, oy + 6), path.name, fill="black", font=font)
            caption = f'Komponenten: {len(contours)} | Rand: {"ja" if touches else "nein"}'
            draw.text((ox + 8, oy + 26), caption, fill="black", font=font)
        sheet.save(OUTPUT / f"multiple_group_{group}.jpg", quality=92)
    return rows


def write_measurement_copy(source_text, rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["coin"], []).append(row)
    output_lines = []
    found = set()
    for line in source_text.splitlines():
        match = re.fullmatch(r"(Referenz|[0-9,]+€)\s+([0-9]+,[0-9]+)mm\s*", line)
        if match:
            coin, size_text = match.groups()
            size_mm = float(size_text.replace(",", "."))
            group = grouped[coin]
            mean = float(np.mean([row["diameter_px"] for row in group]))
            for row in group:
                row["documented_diameter_mm"] = size_mm
            output_lines.append(
                f'{coin:<16} {size_text + "mm":<22} {decimal(mean) + " px":>20}'
            )
            found.add(coin)
        elif line.startswith("Coin:"):
            output_lines.append(
                f'{"Coin:":<16} {"Measurement [mm]:":<22} {"Measurement [px]:":>20}'
            )
        else:
            output_lines.append(line)
    if found != set(grouped):
        raise ValueError("Bildklassen und ursprüngliche Messliste stimmen nicht überein")
    output_lines += [
        "", "Kopie mit Bildmessungen - 27.10.2026", "=" * 60,
        "Die mm-Werte stammen unverändert aus Coins_masurements.txt.",
        "Die px-Spalte oben enthält den Mittelwert der beiden Aufnahmen je Münzwert.",
        "Für die Referenz liegt genau eine Aufnahme vor.",
        "", "Einzelmessungen in der nativen Bildauflösung (2592 x 1944 px):", "",
        f'{"Bilddatei":<24} {"D außen [px]":>13} {"Ellipse klein [px]":>19} '
        f'{"Ellipse groß [px]":>18}',
        "-" * 78,
    ]
    for row in rows:
        output_lines.append(
            f'{row["file"]:<24} {decimal(row["diameter_px"]):>13} '
            f'{decimal(row["ellipse_minor_px"]):>19} '
            f'{decimal(row["ellipse_major_px"]):>18}'
        )
    output_lines += [
        "", "Messdefinition und Verfahren", "-" * 29,
        "D außen ist der maximale Feret-Durchmesser: der größte euklidische Abstand",
        "zweier Punkte des äußeren Bildumrisses. Innenringe und Prägung zählen nicht.",
        "Auch die 20-Cent-Münze wird am äußeren Umriss statt über ihre Fläche gemessen.",
        "Die Ellipsenachsen sind zusätzliche Vergleichsmaße, nicht die Hauptmessung.",
        "", "Verarbeitung: Graustufen, Gaußfilter 5 x 5 (sigma = 0,8 px), Otsu-Schwelle,",
        "größtes äußeres Objekt. Entlang 4096 radialer Profile wird der äußere",
        "Dunkel-Hell-Übergang mit 0,125-px-Abtastung und linearer Interpolation",
        "lokalisiert. Der maximale Abstand wird auf der konvexen Hülle berechnet.",
        "Messung direkt in den Originalbildern, ohne Skalierung oder geometrische",
        "Entzerrung. Die Kontrollbilder werden nur zur Darstellung verkleinert.",
        "", "Die Nachkommastelle beschreibt die numerische Auswertung, keine garantierte",
        "Messgenauigkeit von 0,1 px. Als Empfindlichkeitsprüfung wurde die",
        "Otsu-Schwelle um jeweils 15 Graustufen verändert. Die resultierenden",
        "Durchmesser-Spannen stehen unten; sie sind kein Konfidenzintervall und",
        "enthalten weder Objektivverzeichnung noch Perspektivfehler.",
        "", f'{"Bilddatei":<24} {"D bei T-15 [px]":>17} {"D bei T+15 [px]":>17}',
        "-" * 60,
    ]
    for row in rows:
        output_lines.append(
            f'{row["file"]:<24} '
            f'{decimal(row["diameter_threshold_minus_15_px"]):>17} '
            f'{decimal(row["diameter_threshold_plus_15_px"]):>17}'
        )
    output_lines += [
        "", "Zuordnung: _1 und _2 bezeichnen die zugehörigen Bilddateien; der Mittelwert",
        "ist eine Zusammenfassung zweier Aufnahmen an verschiedenen Bildpositionen.",
        "Er ist keine zusätzliche Einzelmessung und kein Nachweis der Wiederholgenauigkeit.",
        "Alle 17 Einzelobjekte wurden vollständig im Bild erfasst. Besonders nah am",
        "Rand liegen 1€_2.bmp (3 px) und 0_01€_2.bmp (8 px), bezogen auf die Binärkontur.",
        "", "Kontrollbilder und ausführliche Ordneranalyse: measurement_results/",
        "Reproduzieren: /usr/bin/python3 CV_lab/Pictures/measurement_results/measure_diameters.py",
        "Abhängigkeiten: numpy, cv2 (OpenCV), PIL (Pillow); Randvergleich zusätzlich matplotlib.",
        "Verfahrensreferenz: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html",
    ]
    destination = PICTURES / "Coins_measurements_measured.txt"
    content = "\n".join(output_lines) + "\n"
    for path in (destination, PICTURES / "Coins_masurements_measured.txt"):
        path.write_text(content, encoding="utf-8")
    return destination


def write_analysis(rows, multiple_rows):
    all_files = sorted(PICTURES.rglob("*.bmp"))
    byte_count = sum(path.stat().st_size for path in all_files)
    report = [
        "ANALYSE DES ORDNERINHALTS - 27.10.2026", "=" * 48, "",
        f"{len(all_files)} lesbare BMP-Bilder: 2592 x 1944 px, 24-Bit-Farbe.",
        f"Gesamte Bilddaten: {byte_count} Bytes ({byte_count / 1024**3:.2f} GiB).",
        f"Single Coins: {len(rows)} Bilder; acht Münzwerte mit jeweils zwei Aufnahmen",
        "und ein separates Bild der laut Messliste 25,00 mm großen Referenz.",
        "Multiple Coins: vier Serien mit je 16 Bildern.", "",
        "Dokumentierte Soll-Zusammensetzung aus Datasets.txt:",
        "1: 3 Münzen, 3,50 Euro (2 Euro + 1 Euro + 50 Cent).",
        "2: 5 Münzen, 0,38 Euro (20 + 10 + 5 + 2 + 1 Cent).",
        "3: 4 Münzen, 2,62 Euro (2 Euro + 50 + 10 + 2 Cent).",
        "4: 4 Münzen, 1,26 Euro (1 Euro + 20 + 5 + 1 Cent).", "",
        "Alle vier in der Liste angegebenen Summen sind rechnerisch korrekt.",
        "Die Liste beschreibt die Serien; pro Bild fehlen Objektpositionen und",
        "Sichtbarkeitsangaben. Ein verdecktes Objekt ist dadurch nicht separat annotiert.",
        "Datasets.txt nennt Coins_measurements.txt; tatsächlich vorhanden ist",
        "Coins_masurements.txt. Verwendet wurde die tatsächlich vorhandene Datei.", "",
        "Befund der Bildsichtung und einfachen Vordergrundsegmentierung:",
        "Heller Hintergrund und dunkle Silhouetten ermöglichen eine klare Abgrenzung.",
        "Die Münzflächen sind teilweise sehr dunkel; ihre Farbe ist weniger stabil",
        "erkennbar als der äußere Rand. Die Einzelbilder zeigen zentrale und randnahe Lagen.",
        "Kein Einzelobjekt ist angeschnitten; zwei Aufnahmen haben weniger als 10 px Randabstand.",
        "Die Mehrmünzserien enthalten getrennte, berührende und deutlich überdeckte Münzen.",
        "Mehrere Szenen zeigen nicht jede laut Serienliste vorhandene Münze eigenständig.",
        "Daher vollständige Verdeckung beziehungsweise abweichenden Bildinhalt bildweise",
        "prüfen, bevor Serienbeträge als visuell nachweisbare Referenz verwendet werden.", "",
        "Automatisch gefundene zusammenhängende Vordergrundbereiche je Serie:",
        "Diese Bereiche sind NICHT mit der tatsächlichen Münzanzahl gleichzusetzen.",
    ]
    for group in range(1, 5):
        subset = [row for row in multiple_rows if row["group"] == group]
        counts = Counter(row["foreground_components"] for row in subset)
        report.append(f"Serie {group}: " + ", ".join(
            f'{count} {"Bereich" if count == 1 else "Bereiche"} in '
            f'{images} {"Bild" if images == 1 else "Bildern"}'
            for count, images in sorted(counts.items())
        ))
    merged = sum(row["foreground_components"] < row["documented_coin_count"]
                 for row in multiple_rows)
    report += ["", f"In {merged} von {len(multiple_rows)} Mehrmünzbildern werden weniger Bereiche",
               "als dokumentierte Münzen gefunden. Eine reine Komponenten-Zählung reicht",
               "für diese Bilder somit nicht aus.", "", "Bilder mit Randanschnitt:"]
    report.extend("- " + row["file"] for row in multiple_rows if row["touches_image_border"])
    differences = [abs(group[0]["diameter_px"] - group[1]["diameter_px"])
                   for coin in {r["coin"] for r in rows} if coin != "Referenz"
                   for group in [[r for r in rows if r["coin"] == coin]]]
    report += [
        "", "Einzelmünzmessungen:",
        "Alle 16 Münzaufnahmen und die Referenz wurden unabhängig am Bildumriss gemessen.",
        "Hauptmaß: maximaler Feret-Durchmesser. Einzelwerte, Mittelwerte, zusätzliche",
        "Ellipsenachsen und Schwellenempfindlichkeit stehen in der ausgefüllten Kopie.",
        f"Größter Unterschied zwischen _1 und _2: {decimal(max(differences), 2)} px.",
        "Diese Differenz kann Position, Münzseite, Aufnahmesituation und Randbestimmung",
        "enthalten; sie wird nicht pauschal einer bestimmten Ursache zugeschrieben.",
        "", "Erzeugte Ergebnisse:",
        "- ../Coins_masurements_measured.txt: ausgefüllte Kopie und Einzelwerte.",
        "- measurements.json: ungerundete Werte, Randpunkte des Durchmessers, Bild-Hashes.",
        "- measurement_overview_1.jpg und _2.jpg: Kontrolle aller 17 Messumrisse.",
        "- multiple_group_1.jpg bis _4.jpg: Übersicht sämtlicher 64 Mehrmünzbilder.",
        "- multiple_analysis.json: Komponentenstatistik und Randanschnitte je Bild.",
        "- measure_diameters.py: reproduzierbare Auswertung.",
        "", "Grenzen: Bilddurchmesser sind Messwerte im ursprünglichen Pixelkoordinatensystem.",
        "Es wurde keine Kamerakalibrierung oder Verzeichnungskorrektur vorausgesetzt.",
        "Originalbilder, Datasets.txt und die ursprüngliche Messliste werden nicht verändert.",
    ]
    (OUTPUT / "dataset_analysis.txt").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    source = PICTURES / "Coins_masurements.txt"
    source_bytes = source.read_bytes()
    source_text = source_bytes.decode("utf-8")
    paths = sorted((PICTURES / "Single Coins").glob("*.bmp"))
    if not paths:
        raise ValueError("Keine Einzelbilder vorhanden")
    rows, boundaries = [], {}
    for path in paths:
        row, outline = measure(path)
        rows.append(row)
        boundaries[path.name] = outline
    destination = write_measurement_copy(source_text, rows)
    single_overviews(rows, boundaries)
    multiple_rows = analyze_multiple()
    write_analysis(rows, multiple_rows)
    payload = {
        "method": "maximum_feret_subpixel_otsu",
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "angular_samples": ANGLE_COUNT, "radial_step_px": RADIAL_STEP,
        "original_measurement_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "single_images": rows,
    }
    (OUTPUT / "measurements.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "multiple_analysis.json").write_text(
        json.dumps(multiple_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if source.read_bytes() != source_bytes:
        raise RuntimeError("Die ursprüngliche Messliste wurde unerwartet verändert")
    # Keep the paired comparison and both filename variants current after a rerun.
    if (OUTPUT / "analyze_edge_distortion.py").exists():
        import analyze_edge_distortion
        analyze_edge_distortion.main()
    print(f"Vermessen: {len(rows)} Einzelbilder; analysiert: {len(multiple_rows)} Mehrmünzbilder")
    print(f"Ausgefüllte Kopie: {destination}")


if __name__ == "__main__":
    main()
