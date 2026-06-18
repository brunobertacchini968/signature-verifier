import argparse
import sys
import os
import cv2
import warnings

# Suprimir advertencias de urllib3 sobre OpenSSL/LibreSSL en MacOS
warnings.filterwarnings("ignore", module="urllib3")

# Importamos nuestros módulos en src/
from src.detector import detectar_y_recortar_firma, SignatureDetectionError
from src.compare import procesar_firma, comparar_firmas
from src.verifier_signet import SigNetVerifier
from src.visualization import show_advanced_matches

def banner():
    print("=" * 70)
    print("      VERIFICADOR HÍBRIDO DE FIRMAS V3 — YOLOv8 + SigNet IA + ORB/SSIM      ")
    print("=" * 70)

def main():
    banner()
    
    parser = argparse.ArgumentParser(
        description="Pipeline Híbrido Avanzado de Detección y Verificación de Firmas."
    )
    parser.add_argument(
        "documento",
        type=str,
        help="Ruta de la imagen del documento escaneado (o firma de prueba)."
    )
    parser.add_argument(
        "referencia",
        type=str,
        help="Ruta de la firma genuina de referencia."
    )
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Evita usar YOLOv8 para detectar la firma (asume que la imagen ya está recortada)."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/signet.pth",
        help="Ruta personalizada para los pesos preentrenados de SigNet."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Umbral de decisión para clasificar como genuina (0-100)."
    )
    
    args = parser.parse_args()
    
    # 1. Validar existencia de archivos de entrada
    if not os.path.exists(args.documento):
        print(f"ERROR: No se encontró el documento en: {args.documento}")
        sys.exit(1)
    if not os.path.exists(args.referencia):
        print(f"ERROR: No se encontró la firma de referencia en: {args.referencia}")
        sys.exit(1)
        
    # 2. Obtener firma de prueba (con detección de YOLO o directa)
    print("\n[Fase 1] Extracción de Firma de Prueba...")
    if not args.no_detect:
        try:
            print("[YOLO] Buscando firma en el documento...")
            firma_test_raw, bbox = detectar_y_recortar_firma(args.documento)
            # Guardamos un archivo temporal para inspección
            cv2.imwrite("firma_recortada_yolo.png", firma_test_raw)
            print("[YOLO] Firma guardada temporalmente como 'firma_recortada_yolo.png' para auditoría.")
        except SignatureDetectionError as e:
            print(f"[YOLO] Advertencia: {e}")
            print("[Fallback] Cargando la imagen del documento completo directamente...")
            firma_test_raw = cv2.imread(args.documento, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            print(f"[YOLO] ERROR en la inicialización o ejecución de YOLO: {e}")
            print("[Fallback] Cargando la imagen directamente sin detección...")
            firma_test_raw = cv2.imread(args.documento, cv2.IMREAD_GRAYSCALE)
    else:
        print("[Manual] Saltando detección YOLO. Leyendo firma de prueba directamente...")
        firma_test_raw = cv2.imread(args.documento, cv2.IMREAD_GRAYSCALE)
        
    if firma_test_raw is None:
        print(f"ERROR: No se pudo leer la firma de prueba en {args.documento}")
        sys.exit(1)
        
    # Cargar firma de referencia en escala de grises
    firma_ref_raw = cv2.imread(args.referencia, cv2.IMREAD_GRAYSCALE)
    if firma_ref_raw is None:
        print(f"ERROR: No se pudo leer la firma de referencia en {args.referencia}")
        sys.exit(1)
        
    # 3. Procesamiento y Limpieza (Fase 2)
    print("\n[Fase 2] Preprocesamiento y Limpieza...")
    # Guardamos temporalmente las imágenes limpias para el visualizador
    # (El visualizador requiere una imagen en escala de grises de 400x200 de cada una)
    # procesar_firma devuelve (imagen_binaria, imagen_grises)
    bin_ref, gray_ref = procesar_firma(args.referencia)
    
    # Para la firma de prueba, la procesamos a partir de lo que YOLO recortó (o cargó)
    # procesar_firma acepta una ruta, pero como ya la tenemos en memoria por YOLO,
    # la guardamos en un archivo temporal corto y la leemos (para mantener compatibilidad absoluta con src.compare)
    temp_test_path = "temp_yolo_for_processing.png"
    cv2.imwrite(temp_test_path, firma_test_raw)
    try:
        bin_test, gray_test = procesar_firma(temp_test_path)
    finally:
        if os.path.exists(temp_test_path):
            os.remove(temp_test_path)
            
    # 4. Análisis Geométrico y Estructural Clásico (ORB + SSIM)
    print("\n[Fase 3] Ejecutando Pipeline Geométrico Clásico (ORB + SSIM)...")
    res_clasico = comparar_firmas(bin_ref, bin_test)
    score_ssim = res_clasico['ssim']
    score_orb = res_clasico['orb']
    score_clasico_total = res_clasico['score_final']
    
    # 5. Análisis con Deep Learning (SigNet)
    print("\n[Fase 4] Ejecutando Pipeline de Deep Learning (SigNet)...")
    try:
        signet_verifier = SigNetVerifier(model_path=args.weights)
        # Comparar usando las firmas limpias normalizadas
        score_ia, distancia = signet_verifier.comparar_firmas(gray_ref, gray_test)
    except Exception as e:
        print(f"[SigNet] ERROR de ejecución en IA: {e}")
        print("[Fallback] Continuando únicamente con métricas clásicas.")
        score_ia = score_clasico_total  # Fallback
        distancia = -1
        
    # 6. Fusión Híbrida y Decisión Final
    print("\n[Fase 5] Fusión de Métricas y Decisión...")
    # Ponderación: 40% Similitud Clásica (ORB/SSIM) + 60% Inteligencia Artificial (SigNet)
    score_final = (score_clasico_total * 0.4) + (score_ia * 0.6)
    
    decision = "GENUINA" if score_final >= args.threshold else "POSIBLE FALSIFICACIÓN"
    
    # 7. Imprimir Reporte en Consola
    print("\n" + "=" * 50)
    print("                 REPORTE DE VERIFICACIÓN                 ")
    print("=" * 50)
    print(f"Métrica SSIM (Estructura Global) : {score_ssim:.2f}%")
    print(f"Métrica ORB (Alineación y Giros) : {score_orb:.2f}% (Matches: {len(res_clasico['good_matches'])})")
    print(f"Métrica SigNet IA (Estilística)  : {score_ia:.2f}% (Distancia: {distancia:.4f})")
    print("-" * 50)
    print(f"Puntaje Clásico Combinado       : {score_clasico_total:.2f}%")
    print(f"Puntaje IA (SigNet)             : {score_ia:.2f}%")
    print(f"Puntaje HÍBRIDO FINAL           : {score_final:.2f}%")
    print("-" * 50)
    color_decision = "\033[92m" if decision == "GENUINA" else "\033[91m"
    print(f"Decisión (Umbral: {args.threshold}%): {color_decision}{decision}\033[0m")
    print("=" * 50)
    
    # 8. Lanzar Visualizador Interactivo Completo
    print("\nLanzando visualizador gráfico interactivo...")
    show_advanced_matches(
        gray_ref, gray_test,
        res_clasico['skel_base'], res_clasico['skel_test'],
        score_ssim, score_orb, score_ia, score_final,
        res_clasico['kp1'], res_clasico['kp2'], res_clasico['good_matches']
    )

if __name__ == "__main__":
    main()
