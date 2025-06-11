import cv2
import os
import sys
import numpy as np
import argparse

# загружаем веса для распознавания лиц
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

# загружаем веса для определения пола и возраста
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

# настраиваем свет
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# итоговые результаты работы нейросетей для пола и возраста
genderList = ['Male', 'Female']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

class AgeGenderDetector:
    def __init__(self, face_model_path, face_proto_path, 
                 gender_model_path, gender_proto_path,
                 age_model_path, age_proto_path):
        """Инициализация детектора лиц, пола и возраста"""
        
        # Проверяем существование всех файлов моделей
        model_files = {
            'Face model': face_model_path,
            'Face proto': face_proto_path,
            'Gender model': gender_model_path,
            'Gender proto': gender_proto_path,
            'Age model': age_model_path,
            'Age proto': age_proto_path
        }
        
        missing_files = []
        for name, path in model_files.items():
            if not os.path.exists(path):
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            raise FileNotFoundError(f"Файлы моделей не найдены:\n" + "\n".join(missing_files))
        
        # Загружаем нейросети
        self.faceNet = cv2.dnn.readNet(face_model_path, face_proto_path)
        self.genderNet = cv2.dnn.readNet(gender_model_path, gender_proto_path)
        self.ageNet = cv2.dnn.readNet(age_model_path, age_proto_path)
    
    def detect_faces(self, frame, conf_threshold=0.7):
        """Функция определения лиц на изображении"""
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        
        faceBoxes = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                faceBoxes.append([x1, y1, x2, y2])
        
        return frameOpencvDnn, faceBoxes
    
    def predict_age_gender(self, face_crop):
        """Определение возраста и пола для обрезанного изображения лица"""
        # Получаем бинарный пиксельный объект
        blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Определяем пол
        self.genderNet.setInput(blob)
        genderPreds = self.genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        gender_confidence = genderPreds[0].max()
        
        # Определяем возраст
        self.ageNet.setInput(blob)
        agePreds = self.ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        age_confidence = agePreds[0].max()
        
        return gender, age, gender_confidence, age_confidence
    
    def analyze_faces(self, frame, conf_threshold=0.7, draw_boxes=True):
        """Полный анализ лиц: детекция + определение возраста и пола"""
        result_img = frame.copy()
        
        # Находим лица
        _, faceBoxes = self.detect_faces(frame, conf_threshold)
        
        face_data = []
        
        # Анализируем каждое найденное лицо
        for i, faceBox in enumerate(faceBoxes):
            x1, y1, x2, y2 = faceBox
            
            # Получаем изображение лица на основе рамки
            face_crop = frame[max(0, y1):min(y2, frame.shape[0] - 1),
                             max(0, x1):min(x2, frame.shape[1] - 1)]
            
            if face_crop.size == 0:  # Проверяем, что обрезка не пустая
                continue
            
            # Определяем пол и возраст
            gender, age, gender_conf, age_conf = self.predict_age_gender(face_crop)
            
            # Сохраняем данные о лице
            face_info = {
                'box': faceBox,
                'gender': gender,
                'age': age,
                'gender_confidence': gender_conf,
                'age_confidence': age_conf
            }
            face_data.append(face_info)
            
            if draw_boxes:
                # Рисуем рамку вокруг лица
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 
                            int(round(frame.shape[0]/150)), 8)
                
                # Добавляем текст с полом и возрастом
                label = f'{gender}, {age}'
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                # Позиционируем текст над рамкой
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + label_size[1] + 10
                
                # Добавляем фон для текста для лучшей читаемости
                cv2.rectangle(result_img, (text_x, text_y - label_size[1] - 3), 
                            (text_x + label_size[0], text_y + 3), (0, 0, 0), -1)
                
                # Добавляем текст
                cv2.putText(result_img, label, (text_x, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        return result_img, face_data
    
    def process_image(self, image_path, output_path=None, conf_threshold=0.7):
        """Обработка отдельного изображения с определением возраста и пола"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")
        
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удается загрузить изображение: {image_path}")
        
        # Анализируем лица
        result_img, face_data = self.analyze_faces(image, conf_threshold)
        
        # Выводим информацию о найденных лицах
        if face_data:
            print(f"Найдено лиц: {len(face_data)}")
            for i, face_info in enumerate(face_data, 1):
                print(f"Лицо {i}:")
                print(f"  Координаты: {face_info['box']}")
                print(f"  Пол: {face_info['gender']} (уверенность: {face_info['gender_confidence']:.2f})")
                print(f"  Возраст: {face_info['age']} лет (уверенность: {face_info['age_confidence']:.2f})")
                print()
        else:
            print("Лица не распознаны")
        
        # Сохраняем результат, если указан путь
        if output_path:
            cv2.imwrite(output_path, result_img)
            print(f"Результат сохранен: {output_path}")
        
        return result_img, face_data
    
    def process_video_stream(self, source=0, conf_threshold=0.7):
        """Обработка видеопотока с камеры или изображения с определением возраста и пола"""
        video = cv2.VideoCapture(source)
        
        if not video.isOpened():
            if isinstance(source, str):
                raise RuntimeError(f"Не удается открыть файл: {source}")
            else:
                raise RuntimeError("Не удается открыть камеру")
        
        # Проверяем, работаем ли мы с изображением или видеопотоком
        is_image = isinstance(source, str) and source != "0"
        
        if is_image:
            print(f"Обрабатываем изображение: {source}")
        else:
            print("Нажмите любую клавишу для выхода...")
        
        frame_count = 0
        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                if is_image:
                    print("Изображение обработано")
                    break
                else:
                    cv2.waitKey()
                    break
            
            frame_count += 1
            
            # Анализируем лица в кадре
            result_img, face_data = self.analyze_faces(frame, conf_threshold)
            
            # Выводим информацию в консоль
            if face_data:
                if is_image:
                    print(f"Найдено лиц: {len(face_data)}")
                    for i, face_info in enumerate(face_data, 1):
                        print(f"Лицо {i}:")
                        print(f"  Пол: {face_info['gender']} (уверенность: {face_info['gender_confidence']:.4f})")
                        print(f"  Возраст: {face_info['age']} лет (уверенность: {face_info['age_confidence']:.4f})")
                        print(f"  Координаты: {face_info['box']}")
                        print()
                else:
                    print(f"Кадр {frame_count}: найдено лиц - {len(face_data)}")
                    for i, face_info in enumerate(face_data, 1):
                        print(f"  Лицо {i}: {face_info['gender']}, {face_info['age']}")
            else:
                print("Лица не распознаны")
            
            # Показываем результат
            window_title = f"Detecting age and gender - {source}" if is_image else "Detecting age and gender"
            cv2.imshow(window_title, result_img)
            
            # Для изображений ждем нажатия клавиши, для видео - проверяем на выход
            if is_image:
                print("Нажмите любую клавишу для закрытия...")
                cv2.waitKey(0)
                break
            elif cv2.waitKey(1) >= 0:
                break
        
        # Освобождаем ресурсы
        video.release()
        cv2.destroyAllWindows()


def main():
    """Главная функция с поддержкой аргументов командной строки"""
    # Подключаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description='Детекция лиц с определением возраста и пола')
    # Добавляем аргумент для работы с изображениями
    parser.add_argument('--image', help='Путь к изображению для анализа')
    # Добавляем аргумент для порога уверенности
    parser.add_argument('--confidence', type=float, default=0.7, 
                       help='Порог уверенности для детекции лиц (по умолчанию: 0.7)')
    # Сохраняем аргументы в отдельную переменную
    args = parser.parse_args()
    
    try:
        # Создаем детектор возраста и пола
        detector = AgeGenderDetector(
            faceModel, faceProto,
            genderModel, genderProto,
            ageModel, ageProto
        )
        
        # Определяем источник: изображение или камера
        source = args.image if args.image else 0
        
        if args.image:
            print(f"=== Анализ изображения: {args.image} ===")
            if not os.path.exists(args.image):
                print(f"Ошибка: файл {args.image} не найден")
                sys.exit(1)
        else:
            print("=== Анализ видеопотока с камеры ===")
        
        # Запускаем обработку
        detector.process_video_stream(source, args.confidence)
    
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)


# Упрощенная функция для обработки отдельного изображения
def detect_age_gender_in_image(image_filename, output_filename=None, conf_threshold=0.7):
    """
    Упрощенная функция для анализа возраста и пола на изображении
    
    Args:
        image_filename (str): Имя файла изображения
        output_filename (str, optional): Имя выходного файла
        conf_threshold (float): Порог уверенности для детекции лиц
    
    Returns:
        tuple: (обработанное_изображение, список_данных_о_лицах)
    """
    detector = AgeGenderDetector(
        faceModel, faceProto,
        genderModel, genderProto,
        ageModel, ageProto
    )
    
    if output_filename is None:
        name, ext = os.path.splitext(image_filename)
        output_filename = f"{name}_age_gender{ext}"
    
    return detector.process_image(image_filename, output_filename, conf_threshold)


if __name__ == "__main__":
    main()