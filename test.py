from typing import List, Dict


def contar_palabras_clave(comentarios: List[str], palabras_clave: List[str]) -> Dict[str, int]:
    # 1. Inicializamos el diccionario con todas las palabras clave en 0
    # Guardamos las llaves en minúsculas para estandarizar
    diccio = {}
    for palabra in palabras_clave:
        diccio[palabra.lower()] = 0

    # 2. Recorremos cada comentario
    for comentario in comentarios:
        # Convertimos el comentario a minúsculas y lo separamos en una lista de palabras
        palabras_del_comentario = comentario.lower().split()

        # 3. Recorremos cada palabra de ese comentario
        for palabra in palabras_del_comentario:
            # Si la palabra es una de las que estamos buscando, sumamos 1
            if palabra in diccio:
                diccio[palabra] += 1

    return diccio


# --- CASOS DE PRUEBA PARA VALIDAR TU CÓDIGO ---
if __name__ == "__main__":
    comentarios_clientes = [
        "El servicio fue excelente y el envío muy rápido",
        "Un servicio excelente pero el envío fue un poco lento",
        "No lo recomiendo es muy lento",
        "Excelente producto lo recomiendo totalmente"
    ]

    mis_palabras = ["excelente", "lento", "recomiendo", "malo"]

    resultado = contar_palabras_clave(comentarios_clientes, mis_palabras)
    print("Resultado del conteo:", resultado)
    # Esperado: {'excelente': 3, 'lento': 2, 'recomiendo': 2, 'malo': 0}