import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import requests
from tqdm import tqdm
from collections import OrderedDict
# Definimos la arquitectura de SigNet en PyTorch (basada en luizgh/sigver)
# para que el script sea 100% independiente y no requiera instalar librerías externas rotas.

def conv_bn_relu(in_channels, out_channels, kernel_size,  stride=1, pad=0):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.ReLU()),
    ]))

def linear_bn_relu(in_features, out_features):
    return nn.Sequential(OrderedDict([
        ('fc', nn.Linear(in_features, out_features, bias=False)),  # Bias is added after BN
        ('bn', nn.BatchNorm1d(out_features)),
        ('relu', nn.ReLU()),
    ]))

# Definimos la arquitectura de SigNet en PyTorch desde luizgh/sigver
class SigNet(nn.Module):
    def __init__(self):
        super(SigNet, self).__init__()
        self.feature_space_size = 2048

        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', conv_bn_relu(1, 96, 11, stride=4)),
            ('maxpool1', nn.MaxPool2d(3, 2)),
            ('conv2', conv_bn_relu(96, 256, 5, pad=2)),
            ('maxpool2', nn.MaxPool2d(3, 2)),
            ('conv3', conv_bn_relu(256, 384, 3, pad=1)),
            ('conv4', conv_bn_relu(384, 384, 3, pad=1)),
            ('conv5', conv_bn_relu(384, 256, 3, pad=1)),
            ('maxpool3', nn.MaxPool2d(3, 2)),
        ]))

        self.fc_layers = nn.Sequential(OrderedDict([
            ('fc1', linear_bn_relu(256 * 3 * 5, 2048)),
            ('fc2', linear_bn_relu(self.feature_space_size, self.feature_space_size)),
        ]))

    def forward(self, inputs):
        x = self.conv_layers(inputs)
        x = x.view(x.shape[0], 256 * 3 * 5)
        x = self.fc_layers(x)
        return x

def descargar_pesos_gdrive(file_id, destino):
    """Descarga de Google Drive de manera robusta usando gdown y descomprime si es necesario."""
    import zipfile
    import shutil
    import gdown
    
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    zip_path = destino.replace(".pth", ".zip")
    
    print(f"[SigNet] Descargando pesos de Google Drive usando gdown en: {zip_path}...")
    try:
        gdown.download(id=file_id, output=zip_path, quiet=False)
    except Exception as e:
        print(f"[SigNet] ERROR al descargar con gdown: {e}")
        raise e
                
    print("[SigNet] Pesos descargados. Descomprimiendo archivo ZIP...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Listar archivos
            archivos = zip_ref.namelist()
            print(f"[SigNet] Contenido del ZIP: {archivos}")
            
            # Buscar el archivo .pth o .pkl o similar
            pth_file = None
            for arch in archivos:
                if arch.endswith(".pth") or arch.endswith(".pkl") or arch.endswith(".pt"):
                    pth_file = arch
                    break
                    
            if pth_file:
                # Extraer el archivo .pth a una carpeta temporal
                temp_dir = os.path.join(os.path.dirname(destino), "temp_extraction")
                zip_ref.extract(pth_file, temp_dir)
                
                # Mover el archivo extraído al destino definitivo
                extracted_path = os.path.join(temp_dir, pth_file)
                if os.path.exists(destino):
                    os.remove(destino)
                shutil.move(extracted_path, destino)
                
                # Eliminar carpeta temporal
                shutil.rmtree(temp_dir)
                print(f"[SigNet] Archivo {pth_file} extraído correctamente a {destino}.")
            else:
                # Si no hay .pth, extraer todo directamente
                zip_ref.extractall(os.path.dirname(destino))
                print("[SigNet] Archivo ZIP extraído por completo en el directorio models.")
    except Exception as e:
        print(f"[SigNet] ERROR al descomprimir: {e}")
        raise e
    finally:
        # Eliminar archivo ZIP temporal
        if os.path.exists(zip_path):
            os.remove(zip_path)

def normalizar_imagen_signet(img_gray):
    """
    Normaliza la firma según los requerimientos de SigNet (sigver):
    - Recorta la firma (Bounding Box)
    - INVIERTE los colores (Fondo negro 0.0, trazo blanco >0)
    - Redimensionado a 150 x 220
    """
    # 1. Binarizar y encontrar Bounding Box
    _, bin_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(bin_img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img_cropped = img_gray[y:y+h, x:x+w]
    else:
        img_cropped = img_gray
        
    # 2. Invertir colores (SigNet fue entrenado con fondo negro)
    img_inverted = 255 - img_cropped
    
    # 3. Cambiar tamaño a 150x220
    img_resized = cv2.resize(img_inverted, (220, 150))
    
    # 4. Convertir a flotante y escalar a [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # 5. Agregar dimensiones para PyTorch: (batch_size, channels, H, W) -> (1, 1, 150, 220)
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)
    return img_tensor

class SigNetVerifier:
    def __init__(self, model_path="models/signet.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        print(f"[SigNet] Ejecutando en dispositivo: {self.device}")
        
        self.model_path = model_path
        
        # Si no existen los pesos, los descarga automáticamente
        if not os.path.exists(model_path):
            file_id = "1GbvMf1V7_r6CajO4UQMh4myQ3qTG6xpt" # ID de google drive oficial
            descargar_pesos_gdrive(file_id, model_path)
            
        # Instanciar modelo
        self.model = SigNet()
        
        # Cargar pesos
        print(f"[SigNet] Cargando pesos desde: {model_path}...")
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            # A veces el state_dict viene empaquetado en una tupla (como en sigver) o en un diccionario
            if isinstance(state_dict, tuple) or isinstance(state_dict, list):
                state_dict = state_dict[0]
            elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Remover prefijos de las llaves si fueron guardados bajo un contenedor nn.DataParallel o similar
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[name] = v
                
            self.model.load_state_dict(new_state_dict)
            print("[SigNet] Pesos cargados con éxito en la red.")
        except Exception as e:
            print(f"[SigNet] ERROR al cargar pesos: {e}")
            raise e
            
        self.model.to(self.device)
        self.model.eval()

    def extraer_embeddings(self, img_gray):
        """Extrae el vector de 2048 dimensiones de la firma."""
        tensor = normalizar_imagen_signet(img_gray).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model(tensor)
            
        # Retorna el vector como un arreglo de NumPy plano
        return embeddings.cpu().numpy().flatten()

    def comparar_firmas(self, img_ref_gray, img_test_gray):
        """
        Compara dos firmas extrayendo sus vectores y calculando su distancia y score.
        """
        emb_ref = self.extraer_embeddings(img_ref_gray)
        emb_test = self.extraer_embeddings(img_test_gray)
        
        # 1. Distancia euclidiana
        distancia = np.linalg.norm(emb_ref - emb_test)
        
        # 2. Score de coincidencia (mapeado empírico para que la salida sea un porcentaje de 0 a 100%)
        # Un umbral típico de aceptación para firmas genuinas vs falsas en SigNet es ~15.0
        # A menor distancia, mayor similitud.
        umbral_max_distancia = 30.0 # Más allá de 30 la firma es extremadamente diferente
        similitud = max(0.0, 1.0 - (distancia / umbral_max_distancia))
        porcentaje_similitud = similitud * 100.0
        
        print(f"[SigNet] Distancia Euclidiana: {distancia:.4f} | Similitud de IA: {porcentaje_similitud:.2f}%")
        
        return porcentaje_similitud, distancia

if __name__ == "__main__":
    # Test corto
    try:
        verifier = SigNetVerifier()
        print("Modelo inicializado correctamente.")
    except Exception as e:
        print("Fallo en inicialización:", e)
