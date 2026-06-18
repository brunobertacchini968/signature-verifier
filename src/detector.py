import cv2
import numpy as np
import os
from ultralytics import YOLO

class SignatureDetectionError(Exception):
    """Excepción lanzada cuando no se puede detectar ninguna firma en el documento."""
    pass

def detectar_y_recortar_firma(ruta_documento, conf_threshold=0.3, padding=15):
    """
    Usa YOLOv8 para localizar y recortar la firma de un documento escaneado.
    
    Parámetros:
        ruta_documento (str): Ruta de la imagen del documento completo.
        conf_threshold (float): Umbral de confianza mínimo para la detección.
        padding (int): Píxeles de margen extra a agregar alrededor de la firma recortada.
        
    Retorna:
        numpy.ndarray: La firma recortada en escala de grises.
        tuple: Coordenadas de la caja (x_min, y_min, x_max, y_max).
    """
    if not os.path.exists(ruta_documento):
        raise FileNotFoundError(f"No se encontró el documento en: {ruta_documento}")
        
    # Cargar la imagen
    img = cv2.imread(ruta_documento)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen en: {ruta_documento}")
        
    h_orig, w_orig = img.shape[:2]
    
    # Cargar el modelo preentrenado de detección de firmas
    # Ultralytics descargará los pesos de Hugging Face automáticamente la primera vez
    model = YOLO("tech4humans/yolov8s-signature-detector")
    
    # Ejecutar la predicción
    results = model.predict(source=img, conf=conf_threshold, verbose=False)
    
    # Buscar la detección con mayor confianza
    best_box = None
    max_conf = -1.0
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Obtener confianza y clase
            conf = float(box.conf[0])
            # La mayoría de estos modelos solo tienen la clase 0 ("signature")
            if conf > max_conf:
                max_conf = conf
                # Convertir coordenadas a enteros en formato [x_min, y_min, x_max, y_max]
                best_box = box.xyxy[0].cpu().numpy().astype(int)
                
    if best_box is None:
        raise SignatureDetectionError(
            f"No se detectó ninguna firma con un nivel de confianza superior a {conf_threshold:.2f}"
        )
        
    x1, y1, x2, y2 = best_box
    
    # Aplicar un margen extra (padding) para no recortar bordes de trazos
    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(w_orig, x2 + padding)
    y2_pad = min(h_orig, y2 + padding)
    
    # Recortar la firma
    firma_bgr = img[y1_pad:y2_pad, x1_pad:x2_pad]
    
    # Convertir a escala de grises
    firma_gray = cv2.cvtColor(firma_bgr, cv2.COLOR_BGR2GRAY)
    
    print(f"[YOLO] Firma detectada con confianza del {max_conf * 100:.2f}% en la caja: {[x1_pad, y1_pad, x2_pad, y2_pad]}")
    
    return firma_gray, (x1_pad, y1_pad, x2_pad, y2_pad)

if __name__ == "__main__":
    # Test rápido de ejecución
    import sys
    if len(sys.argv) > 1:
        try:
            crop, bbox = detectar_y_recortar_firma(sys.argv[1])
            cv2.imwrite("test_yolo_crop.png", crop)
            print("Guardado 'test_yolo_crop.png' exitosamente.")
        except Exception as e:
            print("Error en prueba de YOLO:", e)
