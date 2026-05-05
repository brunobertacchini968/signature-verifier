import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_matches_pdi(img_a, img_b, score_porcentaje_total):
    """
    Recrea la visualización con un layout más limpio: títulos, separación
    y líneas de conexión claras para no amontonar la vista.
    """
    h_a, w_a = img_a.shape
    h_b, w_b = img_b.shape

    # 1. Configuramos el lienzo con espacio extra para que "respiren"
    header_h = 60  # Espacio arriba para los títulos
    gap_w = 120  # Espacio negro en el medio para separar las firmas

    total_h = header_h + max(h_a, h_b)
    total_w = w_a + gap_w + w_b

    # Creamos el lienzo negro
    out_img = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    # 2. Pegamos la Firma 1 (Izquierda) desplazada hacia abajo por el header
    out_img[header_h:header_h + h_a, :w_a, :] = cv2.cvtColor(img_a, cv2.COLOR_GRAY2BGR)

    # 3. Pegamos la Firma 2 (Derecha) desplazada por el header y el gap
    inicio_x_b = w_a + gap_w
    out_img[header_h:header_h + h_b, inicio_x_b:inicio_x_b + w_b, :] = cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)

    # 4. Dibujamos la interfaz (Títulos y línea divisoria)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Títulos
    cv2.putText(out_img, "FIRMA 1 (BASE)", (w_a // 2 - 80, 40), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out_img, "FIRMA 2 (PRUEBA)", (inicio_x_b + w_b // 2 - 90, 40), font, 0.7, (255, 255, 255), 2,
                cv2.LINE_AA)

    # Línea divisoria en el medio del gap (gris oscura)
    medio_x = w_a + (gap_w // 2)
    cv2.line(out_img, (medio_x, 0), (medio_x, total_h), (70, 70, 70), 2)

    # 5. Puntos de anclaje (Análisis por subregiones / Block Matching)
    puntos_anclaje = [
        (80, 60), (200, 60), (320, 60),
        (80, 140), (200, 140), (320, 140)
    ]

    box_size = 40
    font_scale = 0.4
    thickness = 1
    text_color = (0, 255, 0)

    for pt in puntos_anclaje:
        cx, cy = pt

        # Extraemos el parche para la matemática
        y1, y2 = cy - box_size // 2, cy + box_size // 2
        x1, x2 = cx - box_size // 2, cx + box_size // 2

        parche_a = img_a[y1:y2, x1:x2]
        parche_b = img_b[y1:y2, x1:x2]

        # Calculamos similitud LOCAL
        if parche_a.size > 0:
            dif_local = cv2.absdiff(parche_a, parche_b)
            error_local = np.sum(dif_local > 0)
            similitud_local = max(0, 100 - (error_local / parche_a.size * 100))
        else:
            similitud_local = 0.0

        # Coordenadas en el lienzo global (sumando el header y el gap)
        pt_a = (cx, cy + header_h)
        pt_b_shifted = (cx + inicio_x_b, cy + header_h)

        y1_canvas = y1 + header_h
        y2_canvas = y2 + header_h

        # Dibujamos Cuadrados
        cv2.rectangle(out_img, (x1, y1_canvas), (x2, y2_canvas), (0, 255, 0), thickness=2)
        cv2.rectangle(out_img, (x1 + inicio_x_b, y1_canvas), (x2 + inicio_x_b, y2_canvas), (0, 255, 0), thickness=2)

        # Dibujamos Línea de unión (ahora va a cruzar limpiamente por el gap gris)
        cv2.line(out_img, pt_a, pt_b_shifted, (0, 150, 0), thickness=1)

        # Textos de porcentaje local
        text = f"{similitud_local:.0f}%"
        cv2.putText(out_img, text, (x1, y1_canvas - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)
        cv2.putText(out_img, text, (x1 + inicio_x_b, y1_canvas - 5), font, font_scale, text_color, thickness,
                    cv2.LINE_AA)

    # 6. Visualizar con Matplotlib
    out_img_rgb = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(14, 6))  # Un poco más apaisado para acomodar el gap
    plt.imshow(out_img_rgb)

    color_titulo = 'green' if score_porcentaje_total >= 75.0 else 'red'
    plt.title(f"Coincidencia Total: {score_porcentaje_total:.2f}%", fontsize=18, fontweight='bold', color=color_titulo)
    plt.axis('off')
    plt.tight_layout()
    plt.show()