#region Imports
import playwright
from playwright.sync_api import sync_playwright
import time
import cv2
import numpy as np
import os
import threading
import shutil
import random
from pynput import keyboard
import sys
import platform
from datetime import datetime
#endregion

#region Путь к изображениям
images_path = './images/'
loot_images = [f'{images_path}loot1.png', f'{images_path}loot2.png']
materials_images = []
world_image = f'{images_path}world.png'
advert_image = f'{images_path}advert.png'
advert_close1_image = f'{images_path}advert_close1.png'
advert_close2_image = f'{images_path}advert_close2.png'
station_coin_image = f'{images_path}station_coin.png'
close1_image = f'{images_path}close1.png'
restart_image = f'{images_path}restart.png'
collect_all_image = f'{images_path}collect_all.png'
dispatch_all_image = f'{images_path}dispatch_all.png'
basket_image = f'{images_path}basket.png'
basket_free_image = f'{images_path}basket_free.png'
basket_advert_image = f'{images_path}basket_advert.png'
start_free_gear_image = f'{images_path}start_free_gear.png'
start_advert_gear_image = f'{images_path}start_advert_gear.png'
collect_gear_image = f'{images_path}collect_gear.png'
station_image = f'{images_path}station.png'
build_shop_image = f'{images_path}build_shop.png'
failed_ads_image = f'{images_path}failed_ads.png'
close_failed_ads_image = f'{images_path}close_failed_ads.png'
store_image = f'{images_path}store.png'
road_image = f'{images_path}road.png'
kicked_image = f'{images_path}kicked.png'
locomotive_image = f'{images_path}locomotive.png'
#endregion

#region Обявление переменных
screenshot_counter = 0
debug_screenshot_counter = 0
error_counter = 0
iteration_count = 0
break_mark = False
screenshot_directory = "./screenshots" 

# Название окна браузера, в котором открыта игра
target_window_title = "Play Trainstation 2 on PC"

# Контекст текущего браузера
context = None

# Максимальное количество сохраняемых скриншотов
MAX_SCREENSHOTS = 10

# Максимальное количество неверных итерация до перезагрузки
max_errors = 10

# Счётчик нажатий на вкусняшки
loot_clicks_count = 0
advert_clicks_count = 0

# Время в секундах для интервалов
loot_interval = 5  # 5 секунд
advert_interval = 5  # 30 секунд
basket_interval = 1800 # 30 минут
station_coin_interval = 1800  # 30 минут
restart_interval = 30 # 30 секунд
station_collect_coin_interval = -1

# Время последнего выполнения
last_loot_time = time.time()
last_advert_time = time.time()
last_basket_time = time.time() - 1800 # Запуск первой проверки через 15 минут после запуска скрипта
last_station_coin_time = time.time() - 1800 # Запуск первой проверки сразу после запуска скрипта
last_station_collect_coin_time = time.time()
last_restart = time.time()
station_pos = [0,0]
get_gear_pos = [0,0]
#endregion

#region Функция для очистки папки со скриншотами
def clear_screenshot_directory(directory=screenshot_directory):
    """Очищает папку со скриншотами при запуске скрипта."""
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Удаление папки со всеми файлами
    os.makedirs(directory)  # Создание пустой папки
#endregion

#region Стартовые приготовления
# Создание папки для скриншотов, если её нет
if not os.path.exists(screenshot_directory):
    os.makedirs(screenshot_directory)

# Очистка папки со скриншотами при запуске
clear_screenshot_directory(screenshot_directory)
#endregion

#region Функция для сохранения скриншота с меткой клика
def save_screenshot_with_marker(screenshot_array, x, y, directory=screenshot_directory, prefix="screenshot", region_size=400):
    """
    Сохраняет часть скриншота с меткой клика в центре. Сохраняется только предполагаемая область вокруг клика.

    :param screenshot_array: Исходный скриншот в формате numpy array.
    :param x: Координата X предполагаемого клика.
    :param y: Координата Y предполагаемого клика.
    :param directory: Путь к директории для сохранения.
    :param prefix: Префикс для имени файла.
    :param region_size: Размер области вокруг клика (в пикселях), которая будет сохранена.
    """
    global screenshot_counter

    # Приведение координат к целым числам
    x, y = int(x), int(y)

    # Имя файла с использованием счетчика (от 0 до MAX_SCREENSHOTS-1)
    screenshot_filename = f"{directory}/{prefix}_{screenshot_counter % MAX_SCREENSHOTS}.png"

    # Определяем границы области вокруг клика
    h, w = screenshot_array.shape[:2]
    half_size = region_size // 2

    # Координаты области, которую мы вырезаем
    top = max(0, y - half_size)
    bottom = min(h, y + half_size)
    left = max(0, x - half_size)
    right = min(w, x + half_size)

    # Вырезаем область вокруг клика
    cropped_screenshot = screenshot_array[top:bottom, left:right]

    # Рисуем метку в центре области (относительно вырезанной области)
    cv2.circle(cropped_screenshot, (min(half_size, x - left), min(half_size, y - top)), 5, (0, 0, 255), -1)

    # Сохраняем вырезанный скриншот с меткой
    cv2.imwrite(screenshot_filename, cropped_screenshot)

    # Увеличиваем счетчик
    screenshot_counter += 1
#endregion

#region Функция для получения скриншота основного фрейма в виде массива (без мерцания)
def get_screenshot(page):
    try:
        # Получаем скриншот в виде бинарных данных
        screenshot_binary = page.screenshot()
        screenshot_array = np.frombuffer(screenshot_binary, dtype=np.uint8)
        screenshot_image = cv2.imdecode(screenshot_array, cv2.IMREAD_COLOR)
        
        if screenshot_image is None:
            raise ValueError("Ошибка при декодировании скриншота в изображение.")

        return screenshot_image
    except Exception as e:
        return None
#endregion

#region Функция для нахождения наилучшего масштаба изображения на экране с отметкой углов и центра (без мерцания)
def find_best_scale(page, template_image, lower_scale=0.5, upper_scale=2.0, step=0.1, mark_center=False):
    """
    Функция для нахождения лучшего масштаба при поиске шаблона на странице.
    
    :param page: Страница, с которой мы снимаем скриншот
    :param template_image: Путь к изображению шаблона
    :param lower_scale: Нижняя граница масштабирования
    :param upper_scale: Верхняя граница масштабирования
    :param step: Шаг масштабирования
    :param mark_center: Маркировать центр найденного шаблона кружком
    :return: Лучший масштаб, координаты центра найденного шаблона, скриншот с отметками
    """
    best_scale = None
    best_val = -1
    best_location = None
    best_screenshot = None

    # Получение скриншота страницы
    screenshot_array = get_screenshot(page)
    
    if screenshot_array is None:
        return None, None, None

    # Проход по диапазону масштабов
    for scale in np.arange(lower_scale, upper_scale + step, step):
        found, location, screenshot_with_marker = find_template_in_image(screenshot_array, template_image, scale, threshold=0.75, mark_center=mark_center)
        
        if found:
            # Проверка, если найденное совпадение лучше, чем текущее лучшее
            result = cv2.matchTemplate(screenshot_array, cv2.resize(cv2.imread(template_image), (0, 0), fx=scale, fy=scale), cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_scale = round(scale, 2)
                best_val = max_val
                best_location = location
                best_screenshot = screenshot_with_marker

    if best_scale is not None:
        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Лучший масштаб найден: {best_scale:.2f}, уверенность: {best_val:.2f}.')
        return best_scale, best_location, best_screenshot
    else:
        return None, None, None
#endregion
    
#region Функция для переключения на вкладку с определённым заголовком
def switch_to_tab(context, tab_title):
    for page in context.pages:
        try:
            # Пробуем получить заголовок страницы
            if page.title() == tab_title:
                page.bring_to_front()  # Делаем вкладку активной
                return page
        except playwright._impl._errors.Error as e:
            print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Ошибка при получении заголовка страницы: {e}')
            # Если контекст выполнения разрушен, продолжаем искать дальше
            continue
    return None
#endregion

#region Функция для эмуляции нажатия клавиши Esc в фрейме
def press_escape(page):
    # Эмулируем нажатие клавиши Esc
    page.keyboard.press("Escape")
#endregion

#region Функция для прокрутки колесом мыши внутри фрейма
def scroll_wheel(page, delta_y, steps=10):
    try:
        # Получаем размеры окна браузера (это может быть полезно для расчета центра)
        window_size = page.evaluate("({ width: window.innerWidth, height: window.innerHeight })")
        center_x = window_size['width'] / 2
        center_y = window_size['height'] / 2

        # Перемещаем виртуальный курсор в центр игрового поля
        page.mouse.move(center_x, center_y)

        # Разбиваем прокрутку на несколько шагов
        step_size = delta_y // steps  # Количество пикселей для прокрутки за один шаг
        for _ in range(steps):
            page.mouse.wheel(0, step_size)
            time.sleep(0.05)  # Небольшая задержка между прокрутками

        direction = "вниз" if delta_y > 0 else "вверх"
        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Колесо прокручено {direction} в main_frame на {delta_y} пикселей в {steps} шагов.')
    except Exception as e:
        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Ошибка при прокрутке колесом в main_frame: {e}')
#endregion

#region Функция поиска статического объекта
def find_template(page, template_image, best_scale, threshold=0.8, mark_center=False):
    """Функция поиска статического объекта."""
    
    screenshot_array = get_screenshot(page)
    if screenshot_array is None:
        return False, None, None

    found, location, marked_screenshot = find_template_in_image(screenshot_array, template_image, scale=best_scale, threshold=threshold, mark_center=mark_center)

    if found:
        x, y = location
        return True, location, marked_screenshot
    else:
        return False, None, None
#endregion

#region Функция поиска статического объекта с учётом альфаканала
def find_template_in_image_with_alpha(page, template_image, best_scale, threshold=0.8, mark_center=False):
    """
    Функция для поиска шаблона в скриншоте с использованием альфа-канала.
    
    :param page: Страница с игрой.
    :param template_image: Путь к изображению шаблона с альфа-каналом.
    :param scale: Масштаб для поиска.
    :param threshold: Пороговое значение совпадения.
    :param mark_center: Маркировать центр найденного шаблона кружком.
    :return: Найден ли шаблон (True/False), координаты центра шаблона, модифицированный скриншот.
    """

    # Получение скриншота страницы
    screenshot_array = get_screenshot(page)
    if screenshot_array is None:
        return False, None, None

    # Загрузка шаблона с альфа-каналом
    template_rgba = cv2.imread(template_image, cv2.IMREAD_UNCHANGED)
    
    if template_rgba is None:
        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Ошибка: не удалось загрузить изображение {template_image}.')
        return False, None, None

    # Проверяем, содержит ли изображение альфа-канал
    if template_rgba.shape[2] != 4:
        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Ошибка: шаблон {template_image} не содержит альфа-канала.')
        return False, None, None

    # Отделяем альфа-канал
    b, g, r, alpha = cv2.split(template_rgba)

    # Собираем обратно RGB изображение
    template_rgb = cv2.merge([b, g, r])

    # Масштабирование шаблона и маски
    template_resized = cv2.resize(template_rgb, (0, 0), fx=best_scale, fy=best_scale)
    mask_resized = cv2.resize(alpha, (template_resized.shape[1], template_resized.shape[0]))

    # Преобразуем маску к одно-канальному изображению (чёрно-белому)
    mask_resized_3channel = cv2.merge([mask_resized, mask_resized, mask_resized])

    # Проверка размеров маски и изображения
    if screenshot_array.shape[:2] != mask_resized_3channel.shape[:2]:
        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Ошибка: размеры маски ({mask_resized_3channel.shape[:2]}) и изображения ({screenshot_array.shape[:2]}) не совпадают.')
        return False, None, None

    # Применяем маску к скриншоту
    screenshot_masked = cv2.bitwise_and(screenshot_array, screenshot_array, mask=mask_resized)

    # Применяем маску к шаблону
    template_masked = cv2.bitwise_and(template_resized, template_resized, mask=mask_resized)

    # Поиск шаблона в изображении
    result = cv2.matchTemplate(screenshot_masked, template_masked, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        # Координаты центра найденного шаблона
        center_x = max_loc[0] + template_resized.shape[1] // 2
        center_y = max_loc[1] + template_resized.shape[0] // 2

        if mark_center:
            # Рисуем кружок в центре найденного шаблона
            cv2.circle(screenshot_array, (center_x, center_y), 10, (0, 0, 255), -1)

        # Наложение самого шаблона на скриншот (для отладки)
        top_left = max_loc
        bottom_right = (top_left[0] + template_resized.shape[1], top_left[1] + template_resized.shape[0])
        cv2.rectangle(screenshot_array, top_left, bottom_right, (255, 0, 0), 2)

        return True, (center_x, center_y), screenshot_array
    else:
        return False, None, screenshot_array
#endregion

#region Функция для клика по позиции
def click_by_pos(page, x, y):
    """Функция клика по статическому объекту по координатам."""

    page.mouse.move(x, y)
    page.mouse.click(x, y)
#endregion

#region Функция для клика по статическому объекту с использованием найденного масштаба
def click_static_template(page, template_image, best_scale, threshold=0.8, delay_time=0, offset_x=0, offset_y=0, save_screenshot=True):
    """Функция клика по статическому объекту с сохранением скриншота с меткой."""

    time.sleep(delay_time)
    found, location, marked_screenshot = find_template(page=page, template_image=template_image, best_scale=best_scale, threshold=threshold)

    if found:
        x, y = location
        x += offset_x
        y += offset_y
        page.mouse.move(x, y)
        page.mouse.click(x, y)

        if save_screenshot:
            # Сохраняем скриншот с меткой
            save_screenshot_with_marker(marked_screenshot, x, y, screenshot_directory)

        return True, x, y
    else:
        return False, None, None
#endregion

#region Функция для поиска шаблона в скриншоте с использованием заданного масштаба.
def find_template_in_image(screenshot_array, template_image, scale, threshold=0.8, mark_center=False):
    """
    Функция для поиска шаблона в скриншоте с использованием заданного масштаба.
    
    :param screenshot_array: Массив скриншота (numpy array)
    :param template_image: Путь к изображению шаблона
    :param scale: Масштаб для поиска
    :param threshold: Пороговое значение совпадения
    :param mark_center: Маркировать центр найденного шаблона кружком
    :return: Найден ли шаблон (True/False), координаты центра шаблона, модифицированный скриншот
    """
    # Загрузка шаблона изображения
    template = cv2.imread(template_image, cv2.IMREAD_COLOR)
    
    if template is None:
        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Ошибка: не удалось загрузить изображение {template_image}.')
        return False, None, None

    # Масштабирование шаблона
    template_resized = cv2.resize(template, (0, 0), fx=scale, fy=scale)

    # Поиск шаблона в изображении
    result = cv2.matchTemplate(screenshot_array, template_resized, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        # Координаты центра найденного шаблона
        center_x = max_loc[0] + template_resized.shape[1] // 2
        center_y = max_loc[1] + template_resized.shape[0] // 2

        if mark_center:
            # Рисуем кружок в центре найденного шаблона
            cv2.circle(screenshot_array, (center_x, center_y), 10, (0, 0, 255), -1)

        # Наложение самого шаблона на скриншот (для отладки)
        top_left = max_loc
        bottom_right = (top_left[0] + template_resized.shape[1], top_left[1] + template_resized.shape[0])
        cv2.rectangle(screenshot_array, top_left, bottom_right, (255, 0, 0), 2)

        return True, (center_x, center_y), screenshot_array
    else:
        return False, None, screenshot_array
#endregion
 
#region Функция для клика по движущемуся объекту с учетом найденного масштаба
def click_moving_template(page, template_image, best_scale, threshold=0.8, search_radius=100, save_screenshot=True, click_radius=5):
    """Функция клика по движущемуся объекту с использованием заданного масштаба и ограничением области поиска на втором скриншоте."""

    # Первый скриншот
    screenshot_array = get_screenshot(page)
    if screenshot_array is None:
        return False

    found, location1, marked_screenshot1 = find_template_in_image(screenshot_array, template_image, scale=best_scale, threshold=threshold, mark_center=False)

    if not found:
        return False

    # Определяем область интереса вокруг первого совпадения
    x1, y1 = location1
    height, width = screenshot_array.shape[:2]

    # Ограничиваем область на втором скриншоте
    roi_top = max(0, y1 - search_radius)
    roi_bottom = min(height, y1 + search_radius)
    roi_left = max(0, x1 - search_radius)
    roi_right = min(width, x1 + search_radius)

    # Второй скриншот
    time.sleep(0.1)  # Короткая пауза для фиксации движения
    screenshot_array2 = get_screenshot(page)
    if screenshot_array2 is None:
        return False

    # Ограничиваем изображение второй области интереса
    roi_screenshot = screenshot_array2[roi_top:roi_bottom, roi_left:roi_right]

    # Поиск шаблона в области второго скриншота
    found, location2, marked_screenshot2 = find_template_in_image(roi_screenshot, template_image, scale=best_scale, threshold=threshold, mark_center=False)

    if not found:
        return False

    # Приводим координаты второго шаблона к глобальной системе координат
    location2_global = (location2[0] + roi_left, location2[1] + roi_top)

    # Определяем движение объекта
    delta_x = location2_global[0] - location1[0]
    delta_y = location2_global[1] - location1[1]

    # Предсказываем будущее местоположение объекта
    predicted_x = location2_global[0] + delta_x #* (5 if delta_x > 0 else -1)
    predicted_y = location2_global[1] + delta_y #* (5 if delta_y > 0 else -1)

    # Генерируем случайные координаты в радиусе вокруг предполагаемой точки
    random_offset_x = random.randint(-click_radius, click_radius)
    random_offset_y = random.randint(-click_radius, click_radius)

    adjusted_x = predicted_x + random_offset_x
    adjusted_y = predicted_y + random_offset_y

    # Выполняем клик
    page.mouse.move(adjusted_x, adjusted_y)
    page.mouse.click(adjusted_x, adjusted_y)

    # Сохранение скриншота с маркерами при необходимости
    if save_screenshot:
        # Используем функцию save_screenshot_with_marker для сохранения скриншота с отметками
        screenshot_counter = save_screenshot_with_marker(screenshot_array2, adjusted_x, adjusted_y, region_size=search_radius)

    press_escape(page)
    return True
#endregion

#region Функция нажатия на координаты с зажатием
def click_and_hold(page, x, y, hold_time=0.2):
    """
    Функция для клика с зажатием на малое время.
    
    :param page: Страница, на которой выполняется клик
    :param x: Координата X для клика
    :param y: Координата Y для клика
    :param hold_time: Время удержания клика в секундах
    """
    # Перемещаем мышь на координаты объекта
    page.mouse.move(x, y)
    
    # Нажимаем и удерживаем кнопку
    page.mouse.down()
    time.sleep(hold_time)  # Удерживаем нажатие в течение заданного времени

    # Отпускаем кнопку
    page.mouse.up()
#endregion

#region Функция для отслеживания объекта с помощью Optical Flow
def track_object_with_optical_flow(page, template_path, best_scale, threshold=0.6, tracking_duration=5):
    start_time = time.time()
    result = False
    try:
        # Первоначальный поиск объекта
        found, first_position, first_screenshot = find_template(page, template_path, best_scale, threshold, mark_center=False)

        if not found:
            return False

        # Преобразуем изображение в черно-белое для использования Optical Flow
        prev_gray = cv2.cvtColor(first_screenshot, cv2.COLOR_BGR2GRAY)
        prev_points = np.array([[first_position]], dtype=np.float32)

        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        while time.time() - start_time < tracking_duration:
            # Получаем текущий кадр (снимок экрана)
            screenshot = get_screenshot(page)
            current_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

            # Используем Optical Flow для отслеживания положения объекта
            next_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, prev_points, None, **lk_params)

            if status[0][0] == 1:
                new_position = next_points[0][0]
                x, y = float(new_position[0]), float(new_position[1])

                # Перемещаем мышь и кликаем по новому положению
                click_and_hold(page, x, y, hold_time=0)
                result = True

                # Закрытие рекламы (или других окон) после клика
                close_advert(page, best_scale, 0.2)

                # Обновляем предыдущий кадр и точки для следующей итерации
                prev_gray = current_gray.copy()
                prev_points = next_points

            time.sleep(0.2)
        if result:
            # Сохранение скриншота с меткой нажатия
            save_screenshot_with_marker(screenshot, x, y, screenshot_directory)
    except Exception as e:
        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Ошибка при отслеживании объекта: {e}')
    
    return result
#endregion

#region Функция закрытия рекламы
def close_advert(page, best_scale, delay=0):
    no_ads = click_static_template(page, failed_ads_image, best_scale=best_scale, save_screenshot=False)
    if no_ads[0]:
        time.sleep(20)
        click_static_template(page, close_failed_ads_image, best_scale=best_scale, save_screenshot=False)
        time.sleep(5)
        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Посмотрели рекламку')
    press_escape(page)
    time.sleep(delay)
#endregion

#region Перезагрузка страницы
def reload_page(page):
    page.reload()
    time.sleep(90)
    perform_mouse_scroll(page, distance_percentage_x=-0.12, distance_percentage_y=0.1)
    scroll_wheel(page, delta_y=300, steps=3)
#endregion

#region Сбор монеток на станции
def collect_coins(page, x, y, best_scale):
    click_by_pos(page,x,y)
    time.sleep(0.5)
    click_static_template(page, collect_all_image, best_scale, threshold=0.85)
    press_escape(page)
#endregion

#region Функция для анализа изменений в области с помощью шаблона
def analyze_area_change(page, center_coords, area_size, original_image, screenshot_array, threshold=0.8, save_debug=False, screenshot_directory="./screenshots", MAX_SCREENSHOTS=10):
    """
    Анализирует изменения в области вокруг центра найденного шаблона.

    :param page: Объект страницы браузера
    :param center_coords: Координаты центра найденного шаблона (x, y)
    :param area_size: Размер области для анализа (ширина, высота)
    :param original_image: Оригинальное изображение для сравнения
    :param threshold: Порог для определения изменений
    :param save_debug: Сохранять ли анализируемую область для дебага
    :param screenshot_directory: Папка для сохранения скриншотов
    :param MAX_SCREENSHOTS: Максимальное количество скриншотов для сохранения
    :return: True, если обнаружены изменения, иначе False
    """
    global debug_screenshot_counter

    # Извлекаем область вокруг центра
    center_x, center_y = center_coords
    width, height = area_size

    # Рассчитываем координаты углов области на основе центра
    left = int(center_x - width // 2)
    top = int(center_y - height // 2)
    right = int(center_x + width // 2)
    bottom = int(center_y + height // 2)

    # Убедимся, что координаты остаются в пределах изображения
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, screenshot_array.shape[1])
    bottom = min(bottom, screenshot_array.shape[0])

    # Извлекаем область для анализа
    current_area = screenshot_array[top:bottom, left:right]

    # Сравниваем текущую область с оригинальным изображением
    res = cv2.matchTemplate(current_area, original_image, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)

    # Если сохранение для дебага включено, сохраняем область
    if save_debug:
        if not os.path.exists(screenshot_directory):
            os.makedirs(screenshot_directory)

        # Ограничиваем количество сохранённых скриншотов
        screenshot_filename = f"{screenshot_directory}/collect_checks_{debug_screenshot_counter % MAX_SCREENSHOTS}.png"
        save_screenshot_with_marker(screenshot_array, center_x, center_y, screenshot_directory, prefix="collect_checks")
        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Анализируемая область сохранена: {screenshot_filename}')

        debug_screenshot_counter += 1

    if max_val < threshold:
        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Изменение обнаружено в указанной области.')
        return True
    else:
        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Изменений не обнаружено.')
        return False
#endregion

#region Подбор местоположения игрового поля
def adjust_projection_and_find_template_with_alpha(page, template_image, best_scale, known_corner=None, threshold=0.8, step_percentage=0.05, max_attempts=10):
    """
    Корректирует проекцию игрового поля и ищет шаблон на странице с учётом известного угла.
    
    :param page: Страница, с которой мы снимаем скриншот
    :param template_image: Путь к изображению шаблона
    :param best_scale: Масштаб для поиска
    :param known_corner: Координаты известного угла шаблона (если есть)
    :param threshold: Порог для поиска совпадений
    :param step_percentage: Шаг перемещения поля в процентах
    :param max_attempts: Максимальное количество попыток для корректировки проекции
    :return: Найден ли шаблон (True/False), координаты шаблона, скорректированный скриншот
    """
    attempts = 0
    directions = ['вверх', 'вниз', 'вправо', 'влево']
    found = False

    while attempts < max_attempts and not found:
        # Получаем текущий скриншот
        screenshot_array = get_screenshot(page)

        # Если известен угол, ограничиваем область поиска
        if known_corner:
            x_min, y_min = known_corner
            x_max, y_max = x_min + int(screenshot_array.shape[1] * step_percentage), y_min + int(screenshot_array.shape[0] * step_percentage)
            search_area = screenshot_array[y_min:y_max, x_min:x_max]
        else:
            search_area = screenshot_array

        # Ищем шаблон с альфаканалом
        found, location, marked_screenshot = find_template_in_image_with_alpha(search_area, template_image, scale=best_scale, threshold=threshold, mark_center=True)

        if found:
            print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Шаблон найден на попытке {attempts + 1} в направлении {directions[attempts % 4]}.')
            return True, location, marked_screenshot

        # Если не найден, двигаем игровое поле
        if attempts % 4 == 0:
            perform_mouse_scroll(page, distance_percentage_y=-step_percentage)  # Вверх
        elif attempts % 4 == 1:
            perform_mouse_scroll(page, distance_percentage_y=step_percentage)  # Вниз
        elif attempts % 4 == 2:
            perform_mouse_scroll(page, distance_percentage_x=step_percentage)  # Вправо
        elif attempts % 4 == 3:
            perform_mouse_scroll(page, distance_percentage_x=-step_percentage)  # Влево

        attempts += 1

    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Не удалось найти шаблон в пределах максимального количества попыток.')
    return False, None, None
#endregion

#region Функция для выполнения скролла в пределах фрейма (эмуляция движения мыши)
def perform_mouse_scroll(frame, distance_percentage_x=0, distance_percentage_y=0):
    """
    Эмулирует перемещение мыши и зажатие для скролла игрового поля.
    :param frame: Фрейм, в котором происходит перемещение
    :param distance_percentage_x: Процент перемещения по оси X (вправо-влево), положительное значение — вправо
    :param distance_percentage_y: Процент перемещения по оси Y (вверх-вниз), положительное значение — вниз
    """
    # Определяем размер видимой области фрейма
    viewport_size = frame.viewport_size
    if not viewport_size:
        viewport_size = frame.evaluate("() => ({ width: window.innerWidth, height: window.innerHeight })")
    
    center_x = viewport_size['width'] // 2
    center_y = viewport_size['height'] // 2

    scroll_distance_x = int(viewport_size['width'] * distance_percentage_x)
    scroll_distance_y = int(viewport_size['height'] * distance_percentage_y)

    try:
        # Эмулируем зажатие левой кнопки мыши и перемещение на заданные проценты по X и Y
        frame.mouse.move(center_x, center_y)  # Перемещаем курсор в центр фрейма
        frame.mouse.down()  # Зажимаем левую кнопку мыши

        # Перемещение по оси X и Y
        frame.mouse.move(center_x + scroll_distance_x, center_y + scroll_distance_y, steps=10)  
        frame.mouse.up()  # Отпускаем левую кнопку мыши

        direction_x = "вправо" if scroll_distance_x > 0 else "влево" if scroll_distance_x < 0 else "по оси X не перемещено"
        direction_y = "вниз" if scroll_distance_y > 0 else "вверх" if scroll_distance_y < 0 else "по оси Y не перемещено"

        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Скроллинг в фрейме выполнен на {scroll_distance_x} пикселей по X ({direction_x}) и {scroll_distance_y} пикселей по Y ({direction_y}).')
    except Exception as e:
        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Ошибка при выполнении скролла в фрейме: {e}')
#endregion

#region Запуска отслеживания комбинаций клавиш
def handle_key_combinations():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
#endregion

#region Обработки нажатий клавиш
def on_press(key):
    global break_mark
    try:
        if key == (keyboard.Key.ctrl_l and keyboard.Key.alt_l and keyboard.KeyCode(char='q')):
            print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Нажата комбинация Ctrl+Alt+Q. Устанавливаем индикатор для завершения скрипта.')
            break_mark = True

    except AttributeError:
        pass
#endregion

#region Запись состояния
def save_game_state(page, start):

    name_prefix = f'{"0_start" if start else "1_stop"}'

    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Сохраняем состояние {"старт" if start else "стоп"}')
    # Скрин главного экрана
    screenshot = get_screenshot(page)
    cv2.imwrite(f"{screenshot_directory}/{name_prefix}_main.png", screenshot)

    # Скрин депо
    click_static_template(page, locomotive_image, best_scale, save_screenshot=False)
    time.sleep(0.5)
    screenshot = get_screenshot(page)
    cv2.imwrite(f"{screenshot_directory}/{name_prefix}_depo.png", screenshot)
    press_escape(page)
    time.sleep(0.5)

    # Скрин склада
    click_static_template(page, store_image, best_scale, save_screenshot=False)
    time.sleep(0.5)
    screenshot = get_screenshot(page)
    cv2.imwrite(f"{screenshot_directory}/{name_prefix}_store.png", screenshot)
    press_escape(page)
    time.sleep(0.5)
#endregion

#region Логика скрипта

# Запуск функции отслеживания комбинаций клавиш в отдельном потоке
#key_thread = threading.Thread(target=handle_key_combinations)
#key_thread.start()

with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp("http://localhost:8888")
    context = browser.contexts[0]  # Берем первый контекст (окно)

    # Пытаемся переключиться на вкладку с нужным заголовком
    page = switch_to_tab(context, target_window_title)
    
    if not page: # Если вкладка не найдена, открываем новую
        page = context.new_page()
        page.goto("https://portal.pixelfederation.com/en/trainstation2")
    
    # Прокрутка колесом вниз (например, для уменьшения масштаба) внутри фрейма
    scroll_wheel(page, delta_y=300, steps=3)

    # Запуск функции поиска наилучшего масштаба
    best_scale, best_location, screenshot_with_marker = find_best_scale(page, build_shop_image, lower_scale=0.5, upper_scale=2.0, step=0.1, mark_center=True)

    # Запись начального состояния
    save_game_state(page, start=True)

    #'''
    # Бесконечный цикл
    while True:

        if break_mark:
            save_game_state(page, start=False)
            sys.exit()

        if error_counter >= max_errors:
            print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Произошло {max_errors} ошибок, подвигаем мышкой.')
            perform_mouse_scroll(page, distance_percentage_y=-2)  # Вверх
            perform_mouse_scroll(page, distance_percentage_y=2)  # Вниз
            error_counter = 0
            
        result = find_template(page, kicked_image, best_scale=1)
        if(result[0]):
            print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Похоже нас кикнули снова, перегружаем страницу.')
            reload_page(page)
            continue

        close_advert(page,best_scale)
        
        click_static_template(page, station_image, best_scale, save_screenshot=False)
        found, store_location, screenshot = find_template(page, store_image, best_scale, threshold=0.7)
        if not found:
            error_counter += 1
            print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Область с шаблоном store.png не найдена, пропускаем итерацию - {error_counter} из {max_errors}')
            continue
        else:
            iteration_count += 1
            print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Новая итерация #{iteration_count}.')

        current_time = time.time()

        # Проверка и нажатие на loot изображения каждые 5 секунд
        if current_time - last_loot_time >= loot_interval:
            for loot_image in loot_images:
                result = track_object_with_optical_flow(page, loot_image, best_scale, threshold=0.6)
                if result:
                    loot_clicks_count += 1
                    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Что-то из лута было найдено {loot_clicks_count} раз, но хз, собрано ли...')
            last_loot_time = current_time

        # Проверка и нажатие на advert изображение каждые 5 секунд
        if current_time - last_advert_time >= advert_interval:
            result = track_object_with_optical_flow(page, advert_image, best_scale, threshold=0.6)
            if result:
                advert_clicks_count += 1
                print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Похоже мы нашли гемку {advert_clicks_count} раз, но хз, собрано ли...')
            last_advert_time = current_time

        # Проверка на запуск второй копии игры каждые 30 секунд
        if current_time - last_restart >= restart_interval:
            # Проверка и нажатие на restart изображение если оно есть, значит есть второй вход - ждём 5 минут
            result = find_template(page, restart_image, best_scale)
            if result:
                if result[0]:
                    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Есть второй вход - ждём 5 минут.')
                    time.sleep(299)
                    reload_page(page)

            last_restart_time = current_time

        #'''
        # Проверка и нажатие на station_coin изображение каждые 30 минут
        if current_time - last_station_coin_time >= station_coin_interval:
            print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Пришло время проверить станцию')
            found, station_x, station_y = click_static_template(page, station_coin_image, best_scale, threshold=0.85)

            if found:
                station_pos = station_x, station_y
                time.sleep(0.5)
                # Поиск и нажатие на dispatch_all.png внутри station_coin.png
                result, x, y = click_static_template(page, dispatch_all_image, best_scale)
                if result:
                    # Выходим со станции и запускаем таймер ожидания
                    press_escape(page)
                    last_station_collect_coin_time = current_time
                    # Включаем отсчёт на 7 минут до сбора монеток
                    station_collect_coin_interval = 420

            last_station_coin_time = current_time

        # Сбор золотых монеток после отправки и ожидания 7 минут
        if station_collect_coin_interval > -1:
            if current_time - last_station_coin_time >= station_collect_coin_interval:
                print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Пришло время собрать монетки')
                station_x, station_y = station_pos
                collect_coins(page, station_x, station_y, best_scale)
                last_station_collect_coin_time = current_time
                station_collect_coin_interval = -1
        #'''

        # Проверка и нажатие на basket изображение каждые 30 минут
        if current_time - last_basket_time >= basket_interval:
            print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Пришло время проверить корзину')
            result = click_static_template(page, basket_image, best_scale, save_screenshot=False)
            if result:
                time.sleep(0.5)
                # Нажатие на кнопку сбора безрекламных шестерёнок
                result = click_static_template(page, basket_free_image, best_scale, offset_y=160, threshold=0.7, save_screenshot=False)
                if result:
                    # Нажатие на кнопку сбора безрекламных шестерёнок
                    result, gear_pos_x, gear_pos_y = click_static_template(page, start_free_gear_image, best_scale, offset_y=160, threshold=0.7, save_screenshot=False)
                    if result:
                        if(get_gear_pos == 0,0):
                            get_gear_pos = gear_pos_x, gear_pos_y
                        time.sleep(5)
                        press_escape(page)
                        time.sleep(1)
                # Нажатие на кнопку сбора рекламных шестерёнок
                result = click_static_template(page, basket_advert_image, best_scale, offset_y=160, threshold=0.7, save_screenshot=False)
                if result:
                    # Нажатие на кнопку сбора рекламных шестерёнок
                    if(get_gear_pos[0] != 0):
                        click_by_pos(get_gear_pos[0], get_gear_pos[1])
                        time.sleep(2)
                        close_advert(page, best_scale)
                        time.sleep(3)
                # Закрытие окна корзины
                press_escape(page)
                last_basket_time = current_time
        # Задержка перед следующей итерацией
        time.sleep(1)
    #key_thread.join()
    #'''
#endregion
