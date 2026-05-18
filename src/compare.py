import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import skeletonize

# Importa la función de visualización avanzada
from src.visualization import show_advanced_matches

def procesar_firma(ruta_imagen):
    """
    Procesa la firma preservando la intensidad.
    Retorna: (imagen_binaria, imagen_grayscale)
    """
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

    # 1. Cargar en escala de grises
    img_gray = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    
    # 2. Binarizar para encontrar el Bounding Box
    _, binaria = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Limpieza morfológica suave
    kernel = np.ones((3, 3), np.uint8)
    limpia = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

    # 4. Encontrar Bounding Box y recortar ambas versiones
    coords = cv2.findNonZero(limpia)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        rec_bin = limpia[y:y + h, x:x + w]
        rec_gray = img_gray[y:y + h, x:x + w]
    else:
        rec_bin = limpia
        rec_gray = img_gray

    # 5. Normalizar tamaño
    final_bin = cv2.resize(rec_bin, (400, 200))
    final_gray = cv2.resize(rec_gray, (400, 200))
    
    return final_bin, final_gray

def comparar_firmas(firma_base, firma_test):
    """Compara usando SSIM sobre esqueletos y Matching de Puntos Clave ORB."""
    
    # --- 1. Análisis de Forma Global (Esqueletización + SSIM) ---
    # Convertir a arreglo booleano (True donde hay trazo blanco)
    bool_base = firma_base > 127
    bool_test = firma_test > 127
    
    # Esqueletizar: reduce los trazos gruesos a líneas de 1 píxel de ancho.
    # Esto elimina la variabilidad causada por usar diferentes bolígrafos o presiones.
    skel_base = skeletonize(bool_base)
    skel_test = skeletonize(bool_test)
    
    # SSIM (Structural Similarity) mide cómo cambian las estructuras y patrones.
    # data_range=1.0 porque los arrays son 0 o 1.
    score_ssim_raw, _ = ssim(skel_base.astype(float), skel_test.astype(float), data_range=1.0, full=True)
    
    # SSIM va de -1 a 1, lo pasamos a escala de porcentaje 0-100%
    porcentaje_ssim = max(0, (score_ssim_raw + 1) / 2 * 100)
    
    # --- 2. Análisis de Características Locales (ORB) ---
    # ORB busca "esquinas" o giros bruscos en los trazos, útiles para ver si fluyen igual
    orb = cv2.ORB_create(nfeatures=1000)
    
    kp1, des1 = orb.detectAndCompute(firma_base, None)
    kp2, des2 = orb.detectAndCompute(firma_test, None)
    
    porcentaje_orb = 0.0
    good_matches = []
    
    if des1 is not None and des2 is not None:
        # Brute Force Matcher usando distancia de Hamming (óptima para descriptores binarios como ORB)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Encontramos los 2 vecinos más cercanos (k=2) para cada punto clave
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Aplicamos el "Ratio Test" de Lowe
        # Solo nos quedamos con el match si es MUCHO mejor que el segundo candidato
        for match_info in matches:
            if len(match_info) == 2:
                m, n = match_info
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            elif len(match_info) == 1:
                good_matches.append(match_info[0])
        
        # Calculamos un score basado en cuántos "Good Matches" logramos encontrar.
        # Empíricamente, encontrar más de 40 buenas conexiones en una firma de 400x200 es excelente.
        max_matches_esperados = 40.0 
        porcentaje_orb = min(100.0, (len(good_matches) / max_matches_esperados) * 100)
            
    # --- 3. Ponderación Final ---
    # Damos 50% de importancia a la estructura global y 50% a los detalles locales.
    score_final = (porcentaje_ssim * 0.5) + (porcentaje_orb * 0.5)
    
    return {
        'score_final': score_final,
        'ssim': porcentaje_ssim,
        'orb': porcentaje_orb,
        'skel_base': skel_base,
        'skel_test': skel_test,
        'kp1': kp1,
        'kp2': kp2,
        'good_matches': good_matches
    }

if __name__ == "__main__":
    import sys

    img_original = sys.argv[1] if len(sys.argv) > 1 else "original.png"
    img_prueba = sys.argv[2] if len(sys.argv) > 2 else "prueba.png"

    print("Procesando firmas y extrayendo características avanzadas...")

    try:
        bin1, gray1 = procesar_firma(img_original)
        bin2, gray2 = procesar_firma(img_prueba)

        resultados = comparar_firmas(bin1, bin2)
        score = resultados['score_final']

        # El umbral ahora puede ser un poco más relajado porque penaliza distinto
        UMBRAL_DECISION = 75.0
        
        decision = "GENUINA" if score >= UMBRAL_DECISION else "POSIBLE FALSIFICACIÓN"

        print(f"--- RESULTADOS ---")
        print(f"Puntaje SSIM (Estructura Global): {resultados['ssim']:.2f}%")
        print(f"Puntaje ORB (Detalles Locales)  : {resultados['orb']:.2f}% (Matches: {len(resultados['good_matches'])})")
        print(f"Score Ponderado: {score:.2f}%  ->  {decision}")

        show_advanced_matches(
            gray1, gray2, 
            resultados['skel_base'], resultados['skel_test'], 
            resultados['ssim'], resultados['orb'], score,
            resultados['kp1'], resultados['kp2'], resultados['good_matches']
        )

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Asegurate de pasar las rutas correctas a las imagenes.")