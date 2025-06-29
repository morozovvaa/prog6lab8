# Лабораторная работа 8
## Часть 1
### Необходимо отрефакторить код таким образом, чтобы работа велась не только с изображением, полученным с видеокамеры, но и программа могла бы обрабатывать изображение, выданное на вход функции (строка с именем файла, лежащего в каталоге с проектом) и выдавала исходное изображение с обозначенными на нем рамками лица (или всех лиц).

Основные методы:

- detect_faces() - базовый метод детекции лиц (рефакторированная версия highlightFace)
- process_image() - обработка отдельного изображения с возможностью сохранения
- process_video_stream() - работа с видеокамерой (как в оригинальном коде)

![test_image](https://github.com/user-attachments/assets/505cdfc7-ec42-487a-9d2f-6301cb83b209)  

![result_test_image](https://github.com/user-attachments/assets/e96f6fce-18e8-40f9-b7e1-343d3551a59f)  

## Часть 2

- Класс AgeGenderDetector - расширен для работы с моделями определения пола и возраста
- Метод predict_age_gender() - определяет пол и возраст для обрезанного изображения лица
- Метод analyze_faces() - комплексный анализ всех лиц с выводом возраста и пола

![image](https://github.com/user-attachments/assets/136c2072-99d6-480a-b003-3e74dff30059)  

![result_age_gender_test_image](https://github.com/user-attachments/assets/90b8dc35-97fc-483a-8ff9-e7614f631544)  

## Часть 3

Функциональность:

- Парсер аргументов - добавлена поддержка argparse
- Параметр --image - для указания пути к изображению
- Параметр --confidence - для настройки порога уверенности детекции
- Универсальный метод process_video_stream() - теперь работает как с камерой, так и с изображениями

Способы использования:

1. Обработка изображения:
```
python main.py --image test_image.jpg
```
2. Обработка изображения с настройкой порога уверенности:
```
python main.py --image test_image.jpg --confidence 0.5
```
3. Работа с камерой (по умолчанию):
```
python main.py
```

![image](https://github.com/user-attachments/assets/9dd73a1d-1995-41fb-9be1-f70a1e4b2411)


## tkinter

![image](https://github.com/user-attachments/assets/d8cad972-ab9a-421c-adda-1eeea701d0a2)

![image](https://github.com/user-attachments/assets/98b95911-bcbc-4f0b-a2c2-3d78566d4476)
