import cv2
import matplotlib.pyplot as plt
import os
import csv
import numpy as np

imagenes = [
    "2_5psi.jpg",
    "5_5psi.jpg",
    "14_5psi.jpg",
    "16_5psi.jpg"
]

os.makedirs("resultados", exist_ok=True)

resultados_csv = []

for nombre in imagenes:
    print(f"Procesando {nombre}...")

    ruta = f"imagenes/{nombre}"
    img = cv2.imread(ruta)

    if img is None:
        print(f"No se pudo cargar la imagen: {ruta}")
        continue

    alto, ancho = img.shape[:2]

    # ROI amplio
    y1 = int(alto * 0.25)
    y2 = int(alto * 0.75)
    x1 = int(ancho * 0.20)
    x2 = int(ancho * 0.80)

    roi = img[y1:y2, x1:x2]

    # Imagen original con ROI marcado
    img_roi_marcado = img.copy()
    cv2.rectangle(img_roi_marcado, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Procesamiento visual general
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, 120, 255, cv2.THRESH_BINARY)
    bordes = cv2.Canny(gris, 50, 150)
    heatmap = cv2.applyColorMap(gris, cv2.COLORMAP_JET)

    # Procesamiento sobre ROI
    gris_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Máscara del material: detecta zonas claras
    _, mascara_material = cv2.threshold(
        gris_roi,
        180,
        255,
        cv2.THRESH_BINARY
    )

    # Limpieza de ruido
    kernel = np.ones((3, 3), np.uint8)
    mascara_material = cv2.morphologyEx(
        mascara_material,
        cv2.MORPH_OPEN,
        kernel
    )

    # Métrica 1: porcentaje de píxeles activos
    pixeles_activos = cv2.countNonZero(mascara_material)
    pixeles_totales = mascara_material.size
    porcentaje_pixeles_activos = (pixeles_activos / pixeles_totales) * 100

    # Métrica 2: anchura media
    distancia = cv2.distanceTransform(mascara_material, cv2.DIST_L2, 5)
    anchuras = distancia[mascara_material > 0] * 2
    anchura_media = np.mean(anchuras) if len(anchuras) > 0 else 0

    # Métrica 3: desviación estándar de intensidad
    desviacion_intensidad = (
        np.std(gris_roi[mascara_material > 0])
        if pixeles_activos > 0
        else 0
    )

    nombre_sin_ext = nombre.replace(".jpg", "")
    carpeta = f"resultados/{nombre_sin_ext}"
    os.makedirs(carpeta, exist_ok=True)

    # Guardar imágenes generales
    cv2.imwrite(f"{carpeta}/original_roi_marcado.jpg", img_roi_marcado)
    cv2.imwrite(f"{carpeta}/gris.jpg", gris)
    cv2.imwrite(f"{carpeta}/binaria.jpg", binaria)
    cv2.imwrite(f"{carpeta}/bordes.jpg", bordes)
    cv2.imwrite(f"{carpeta}/heatmap.jpg", heatmap)

    # Guardar imágenes usadas para métricas
    cv2.imwrite(f"{carpeta}/roi.jpg", roi)
    cv2.imwrite(f"{carpeta}/mascara_material.jpg", mascara_material)

    resultados_csv.append([
        nombre_sin_ext,
        round(porcentaje_pixeles_activos, 2),
        round(anchura_media, 2),
        round(desviacion_intensidad, 2)
    ])

    # Comparación visual
    fig, axs = plt.subplots(1, 7, figsize=(28, 4))

    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")

    axs[1].imshow(cv2.cvtColor(img_roi_marcado, cv2.COLOR_BGR2RGB))
    axs[1].set_title("ROI marcado")

    axs[2].imshow(gris, cmap="gray")
    axs[2].set_title("Grises")

    axs[3].imshow(binaria, cmap="gray")
    axs[3].set_title("Binaria")

    axs[4].imshow(bordes, cmap="gray")
    axs[4].set_title("Bordes")

    axs[5].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    axs[5].set_title("Heatmap")

    axs[6].imshow(mascara_material, cmap="gray")
    axs[6].set_title("Máscara material")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{carpeta}/comparacion.png")
    plt.close()

# Guardar tabla CSV
with open("resultados/metricas.csv", "w", newline="", encoding="utf-8") as archivo:
    escritor = csv.writer(archivo)
    escritor.writerow([
        "Muestra",
        "% píxeles activos",
        "Anchura media de línea (px)",
        "Desviación estándar de intensidad"
    ])
    escritor.writerows(resultados_csv)

print("Procesamiento completado. Métricas guardadas en resultados/metricas.csv")