import cv2
import matplotlib.pyplot as plt
import os

# Lista de imágenes
imagenes = [
    "2_5psi.jpg",
    "5_5psi.jpg",
    "14_5psi.jpg",
    "16_5psi.jpg"
]

# Crear carpeta resultados
os.makedirs("resultados", exist_ok=True)


    # Procesar cada imagen
    for nombre in imagenes:

        ruta = f"imagenes/{nombre}"

        img = cv2.imread(ruta)

        # Escala de grises
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binarización
        _, binaria = cv2.threshold(gris, 120, 255, cv2.THRESH_BINARY)

        # Bordes
        bordes = cv2.Canny(gris, 50, 150)

        # Heatmap
        heatmap = cv2.applyColorMap(gris, cv2.COLORMAP_JET)

        # -----------------------------
        # Crear carpeta individual
        # -----------------------------

        nombre_sin_ext = nombre.replace(".jpg", "")

        carpeta = f"resultados/{nombre_sin_ext}"

        os.makedirs(carpeta, exist_ok=True)

        # Guardar imágenes
        cv2.imwrite(f"{carpeta}/gris.jpg", gris)
        cv2.imwrite(f"{carpeta}/binaria.jpg", binaria)
        cv2.imwrite(f"{carpeta}/bordes.jpg", bordes)
        cv2.imwrite(f"{carpeta}/heatmap.jpg", heatmap)

        # -----------------------------
        # Comparación visual
        # -----------------------------

        fig, axs = plt.subplots(1,5, figsize=(20,4))

        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original")

        axs[1].imshow(gris, cmap='gray')
        axs[1].set_title("Grises")

        axs[2].imshow(binaria, cmap='gray')
        axs[2].set_title("Binaria")

        axs[3].imshow(bordes, cmap='gray')
        axs[3].set_title("Bordes")

        axs[4].imshow(
            cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        )
        axs[4].set_title("Heatmap")

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()

        plt.savefig(f"{carpeta}/comparacion.png")

        plt.close()

print("Procesamiento completado.")