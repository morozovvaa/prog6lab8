import cv2
import os
import sys

# загружаем веса для распознавания лиц
faceProto = "opencv_face_detector.pbtxt"
# и конфигурацию самой нейросети — слои и связи нейронов
faceModel = "opencv_face_detector_uint8.pb"

class FaceDetector:
    def __init__(self, face_model_path, face_proto_path):
        """Инициализация детектора лиц"""
        if not os.path.exists(face_model_path) or not os.path.exists(face_proto_path):
            raise FileNotFoundError("Файлы модели не найдены. Убедитесь, что файлы модели находятся в директории проекта.")
        
        # запускаем нейросеть по распознаванию лиц
        self.faceNet = cv2.dnn.readNet(face_model_path, face_proto_path)
    
    def detect_faces(self, frame, conf_threshold=0.7):
        """Функция определения лиц на изображении"""
        # делаем копию текущего кадра
        frameOpencvDnn = frame.copy()
        # высота и ширина кадра
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        
        # преобразуем картинку в двоичный пиксельный объект
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
        # устанавливаем этот объект как входной параметр для нейросети
        self.faceNet.setInput(blob)
        # выполняем прямой проход для распознавания лиц
        detections = self.faceNet.forward()
        
        # переменная для рамок вокруг лица
        faceBoxes = []

        # перебираем все блоки после распознавания
        for i in range(detections.shape[2]):
            # получаем результат вычислений для очередного элемента
            confidence = detections[0, 0, i, 2]
            # если результат превышает порог срабатывания — это лицо
            if confidence > conf_threshold:
                # формируем координаты рамки
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                # добавляем их в общую переменную
                faceBoxes.append([x1, y1, x2, y2])
                # рисуем рамку на кадре
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 
                            int(round(frameHeight/150)), 8)
        
        # возвращаем кадр с рамками
        return frameOpencvDnn, faceBoxes
    
    def process_image(self, image_path, output_path=None, conf_threshold=0.7):
        """Обработка отдельного изображения"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")
        
        # загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удается загрузить изображение: {image_path}")
        
        # распознаём лица в изображении
        result_img, face_boxes = self.detect_faces(image, conf_threshold)
        
        # выводим информацию о найденных лицах
        if face_boxes:
            print(f"Найдено лиц: {len(face_boxes)}")
            for i, box in enumerate(face_boxes, 1):
                print(f"Лицо {i}: координаты {box}")
        else:
            print("Лица не распознаны")
        
        # сохраняем результат, если указан путь
        if output_path:
            cv2.imwrite(output_path, result_img)
            print(f"Результат сохранен: {output_path}")
        
        return result_img, face_boxes
    
    def process_video_stream(self, conf_threshold=0.7):
        """Обработка видеопотока с камеры"""
        # получаем видео с камеры
        video = cv2.VideoCapture(0)
        
        if not video.isOpened():
            raise RuntimeError("Не удается открыть камеру")
        
        print("Нажмите любую клавишу для выхода...")
        
        # пока не нажата любая клавиша — выполняем цикл
        while cv2.waitKey(1) < 0:
            # получаем очередной кадр с камеры
            hasFrame, frame = video.read()
            # если кадра нет
            if not hasFrame:
                # останавливаемся и выходим из цикла
                cv2.waitKey()
                break
            
            # распознаём лица в кадре
            resultImg, faceBoxes = self.detect_faces(frame, conf_threshold)
            
            # если лиц нет
            if not faceBoxes:
                # выводим в консоли, что лицо не найдено
                print("Лица не распознаны")
            else:
                print(f"Найдено лиц: {len(faceBoxes)}")
            
            # выводим картинку с камеры
            cv2.imshow("Face detection", resultImg)
        
        # освобождаем ресурсы
        video.release()
        cv2.destroyAllWindows()


def main():
    """Главная функция с примерами использования"""
    try:
        # создаем детектор лиц
        detector = FaceDetector(faceModel, faceProto)
        
        # Пример 1: Обработка изображения
        print("=== Обработка изображения ===")
        image_filename = "test_image.jpg"  # замените на имя вашего файла
        
        if os.path.exists(image_filename):
            # обрабатываем изображение
            result_img, face_boxes = detector.process_image(
                image_filename, 
                output_path="result_" + image_filename
            )
            
            # показываем результат
            cv2.imshow("Face Detection Result", result_img)
            print("Нажмите любую клавишу для продолжения...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Файл {image_filename} не найден. Пропускаем обработку изображения.")
        
        # Пример 2: Работа с видеокамерой
        print("\n=== Работа с видеокамерой ===")
        user_input = input("Запустить видеокамеру? (y/n): ")
        if user_input.lower() == 'y':
            detector.process_video_stream()
    
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)


# Функция для обработки отдельного изображения (упрощенный интерфейс)
def detect_faces_in_image(image_filename, output_filename=None, conf_threshold=0.7):
    """
    Упрощенная функция для обработки одного изображения
    
    Args:
        image_filename (str): Имя файла изображения
        output_filename (str, optional): Имя выходного файла. Если None, то автоматически
        conf_threshold (float): Порог уверенности для детекции лиц
    
    Returns:
        tuple: (обработанное_изображение, список_координат_лиц)
    """
    detector = FaceDetector(faceModel, faceProto)
    
    if output_filename is None:
        name, ext = os.path.splitext(image_filename)
        output_filename = f"{name}_faces_detected{ext}"
    
    return detector.process_image(image_filename, output_filename, conf_threshold)


if __name__ == "__main__":
    main()