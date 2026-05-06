import cv2
import numpy as np
import os

from src.visualization import show_matches_pdi


def procesar_firma(ruta_imagen):
    """Binariza (Otsu), limpia (Morfología), recorta el Bounding Box y normaliza el tamaño."""
    
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

    # Lee la imagen en blanco y negro (escala de grises) ignorando colores
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    
    # Binarización de Otsu: convierte a blanco puro (tinta, 255) y negro puro (fondo, 0)
    _, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Crea un "kernel", una matriz de 3x3 píxeles llena de unos
    kernel = np.ones((3, 3), np.uint8)
    
    # Apertura Morfológica: limpia pequeños puntos blancos (ruido) del fondo negro
    limpia = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

    # Busca las coordenadas de todos los píxeles blancos (la firma)
    coords = cv2.findNonZero(limpia)
    
    if coords is not None:
        # Calcula el Bounding Box (el rectángulo más pequeño que encierra la firma)
        x, y, w, h = cv2.boundingRect(coords)
        
        # Recorta la imagen para eliminar el fondo negro sobrante alrededor
        recortada = limpia[y:y + h, x:x + w]
    else:
        # Si la imagen era totalmente negra, se queda con la original sin recortar
        recortada = limpia

    # Redimensiona la firma recortada a un tamaño fijo de 400x200 píxeles
    # Es fundamental para poder comparar dos firmas píxel por píxel después
    final = cv2.resize(recortada, (400, 200))
    
    # Devuelve la imagen lista y procesada
    return final


def comparar_firmas(firma_base, firma_test):
    """Compara toda la imagen píxel a píxel y devuelve el porcentaje total."""
    
    # Resta una imagen de la otra. Diferencias se marcan con valor > 0
    diferencia = cv2.absdiff(firma_base, firma_test)
    
    # Cuenta cuántos píxeles en total tienen diferencias (es decir, el error)
    error_pixeles = np.sum(diferencia > 0)
    
    # Obtiene la cantidad total de píxeles (400x200 = 80,000 píxeles)
    total_pixeles = firma_base.size
    
    # Calcula el porcentaje de similitud (100% si hay 0 píxeles de error)
    similitud = 100 - (error_pixeles / total_pixeles * 100)
    
    # Devuelve el porcentaje calculado
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