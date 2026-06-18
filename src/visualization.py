import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

def show_advanced_matches(img_a, img_b, skel_a, skel_b, ssim_score, orb_score, signet_score, total_score, kp1, kp2, good_matches):
    """
    Visualización avanzada interactiva: 
    - Arriba: Esqueletos para análisis de similitud estructural (SSIM) y Score de IA (SigNet)
    - Abajo: Conexiones de Puntos Clave usando ORB (Interactivas con Hover)
    """
    fig = plt.figure(figsize=(14, 10))
    color_titulo = 'green' if total_score >= 70.0 else 'red'
    fig.suptitle(f"Validación Híbrida de Firma - Score Final: {total_score:.2f}%", 
                 fontsize=20, fontweight='bold', color=color_titulo)

    # --- Subplots de Intensidad (Falso Color - Presión de bolígrafo) ---
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title("Firma Base (Intensidad de Tinta / Presión)")
    # Invertimos para que el trazo tenga valores altos y se vea mejor con el colormap magma
    img_a_inv = 255 - img_a
    ax1.imshow(img_a_inv, cmap='magma')
    ax1.axis('off')

    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title(f"Firma Prueba | SSIM: {ssim_score:.1f}% | ORB: {orb_score:.1f}% | SigNet IA: {signet_score:.1f}%")
    img_b_inv = 255 - img_b
    ax2.imshow(img_b_inv, cmap='magma')
    ax2.axis('off')

    # --- Subplot ORB Interactivo ---
    ax3 = plt.subplot(2, 1, 2)
    ax3.set_title(f"Alineación Geométrica Clásica ({len(good_matches)} matches) | PASÁ EL MOUSE SOBRE LAS LÍNEAS PARA DETALLES")
    
    # Crear canvas combinado horizontalmente
    hA, wA = img_a.shape
    hB, wB = img_b.shape
    vis_h = max(hA, hB)
    vis_w = wA + wB
    vis = np.zeros((vis_h, vis_w), dtype=np.uint8)
    vis[0:hA, 0:wA] = img_a
    vis[0:hB, wA:wA + wB] = img_b

    # Mostrar la imagen de fondo en escala de grises para los matches
    ax3.imshow(vis, cmap='gray')
    ax3.axis('off')

    # Almacenaremos los objetos de línea de Matplotlib
    lines = []
    
    # Usaremos arrays de Numpy para calcular matemáticas de distancia de forma instantánea
    X1, Y1, X2, Y2 = [], [], [], []

    for m in good_matches:
        ptA = kp1[m.queryIdx].pt
        ptB = kp2[m.trainIdx].pt
        
        x1, y1 = ptA
        x2, y2 = ptB[0] + wA, ptB[1]  # Desplazar la coordenada X porque está en la imagen derecha
        
        X1.append(x1)
        Y1.append(y1)
        X2.append(x2)
        Y2.append(y2)
        
        # Crear la línea de Matplotlib muy transparente al inicio
        line = mlines.Line2D([x1, x2], [y1, y2], color='#00FF00', alpha=0.08, linewidth=1.0)
        ax3.add_line(line)
        lines.append(line)

    # Si no hay matches, salir temprano para evitar errores matemáticos
    if len(good_matches) > 0:
        X1 = np.array(X1)
        Y1 = np.array(Y1)
        X2 = np.array(X2)
        Y2 = np.array(Y2)

        # Precalcular longitud al cuadrado de cada segmento para optimización
        L2 = (X2 - X1)**2 + (Y2 - Y1)**2
        L2[L2 == 0] = 1e-6  # Evitar divisiones por cero

        # Variable de estado (usamos una lista mutable para poder modificarla dentro de la función)
        highlighted_idx = [-1]

        # Función Callback que se dispara cada vez que mueves el ratón
        def on_mouse_move(event):
            # Ignorar si el cursor no está sobre el gráfico de abajo
            if event.inaxes != ax3:
                return
                
            x0, y0 = event.xdata, event.ydata
            if x0 is None or y0 is None:
                return
                
            # Matemática Vectorizada: Calcular distancia de (x0,y0) a todos los segmentos a la vez
            # 1. Proyección escalar (t)
            t = ((x0 - X1) * (X2 - X1) + (y0 - Y1) * (Y2 - Y1)) / L2
            t = np.clip(t, 0.0, 1.0) # Forzar proyección dentro del segmento
            
            # 2. Coordenadas (x,y) del punto proyectado en la línea
            proj_x = X1 + t * (X2 - X1)
            proj_y = Y1 + t * (Y2 - Y1)
            
            # 3. Distancia al cuadrado (más rápido que sacar la raíz a todos)
            dist2 = (x0 - proj_x)**2 + (y0 - proj_y)**2
            
            # Encontrar el índice de la línea más cercana al cursor
            min_idx = np.argmin(dist2)
            min_dist = np.sqrt(dist2[min_idx])
            
            # Umbral de sensibilidad en píxeles
            threshold = 20.0
            
            # Caso A: El cursor se alejó de todas las líneas
            if min_dist > threshold:
                if highlighted_idx[0] != -1: # Si había una resaltada, resetear todo
                    for l in lines:
                        l.set_color('#00FF00')
                        l.set_alpha(0.08)
                        l.set_linewidth(1.0)
                        l.set_zorder(1)
                    highlighted_idx[0] = -1
                    fig.canvas.draw_idle()
                return
                
            # Caso B: El cursor está sobre una línea nueva
            if min_idx != highlighted_idx[0]:
                # "Apagar" todas las líneas
                for l in lines:
                    l.set_color('#00FF00')
                    l.set_alpha(0.08)
                    l.set_linewidth(1.0)
                    l.set_zorder(1)
                    
                # "Encender" la línea seleccionada
                lines[min_idx].set_color('red')
                lines[min_idx].set_alpha(1.0)
                lines[min_idx].set_linewidth(3.0) # Más gruesa
                lines[min_idx].set_zorder(10)     # Traer al frente
                
                highlighted_idx[0] = min_idx
                fig.canvas.draw_idle()

        # Conectar el evento a Matplotlib
        fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()