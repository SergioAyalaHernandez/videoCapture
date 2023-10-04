import cv2
import os

# Lista de índices de cámaras
camera_indices = [0, 2, 1]  # Cambiado a 0, 1 y 2 para las tres cámaras

# Directorio donde se encuentran las imágenes
dataPath = 'D:/OneDrive - AXEDE/capturas'

# Obtener la lista de rutas de imágenes en el directorio dataPath
imagePaths = []
for root, dirs, files in os.walk(dataPath):
    for file in files:
        if file.endswith('.jpg'):
            imagePaths.append(os.path.join(root, file))

# Inicializar el clasificador de detección de caras
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar el modelo entrenado
face_recognizer = cv2.face_LBPHFaceRecognizer.create()
face_recognizer.read('D:/OneDrive - AXEDE/Documentos/entrenamiento/ModeloFaceFrontalData2023.xml')

# Función para obtener el nombre de la persona
def get_person_name(image_path):
    return os.path.basename(os.path.dirname(image_path))

while True:
    for i, camera_index in enumerate(camera_indices):
        # Capturar un cuadro de la cámara
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            continue  # Si no se pudo leer el cuadro, pasar a la siguiente iteración

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 1)

        for (x, y, w, h) in faces:
            rostro = auxFrame[y:y + h, x:x + w]
            # Redimensionar el rostro al tamaño esperado
            rostro_redimensionado = cv2.resize(rostro, (360, 360), interpolation=cv2.INTER_LINEAR)
            # Realizar la predicción utilizando el rostro redimensionado
            result = face_recognizer.predict(rostro_redimensionado)

            # Mostrar el nombre de la persona reconocida
            nombre_reconocido = get_person_name(imagePaths[result[0]])
            cv2.putText(frame, nombre_reconocido, (x, y - 25), 1, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('frame_{}'.format(i), frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()

