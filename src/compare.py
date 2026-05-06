import cv2
import numpy as np
import os

from src.visualization import show_matches_pdi


def procesar_firma(ruta_imagen):
    """Binariza (Otsu), limpia (Morfología), recorta el Bounding Box y normaliza el tamaño."""
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    _, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    limpia = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

    coords = cv2.findNonZero(limpia)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        recortada = limpia[y:y + h, x:x + w]
    else:
        recortada = limpia

    final = cv2.resize(recortada, (400, 200))
    return final


def comparar_firmas(firma_base, firma_test):
    """Compara toda la imagen píxel a píxel y devuelve el porcentaje total."""
    diferencia = cv2.absdiff(firma_base, firma_test)
    error_pixeles = np.sum(diferencia > 0)
    total_pixeles = firma_base.size
    similitud = 100 - (error_pixeles / total_pixeles * 100)
    return similitud

if __name__ == "__main__":
    import sys

    img_original = sys.argv[1] if len(sys.argv) > 1 else "original.png"
    img_prueba = sys.argv[2] if len(sys.argv) > 2 else "prueba.png"

    print("Procesando firmas...")

    try:
        firma1 = procesar_firma(img_original)
        firma2 = procesar_firma(img_prueba)

        porcentaje = comparar_firmas(firma1, firma2)

        UMBRAL_DECISION = 75.0
        decision = "GENUINA" if porcentaje >= UMBRAL_DECISION else "POSIBLE FALSIFICACIÓN"

        print(f"Score: {porcentaje:.2f}%  ->  {decision}")

        show_matches_pdi(firma1, firma2, porcentaje)

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Asegurate de pasar las rutas correctas a las imagenes.")