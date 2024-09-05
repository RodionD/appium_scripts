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
#endregion

#region Путь к изображениям
images_path = './images/'
world_image = f'{images_path}world.png'
loot_images = [f'{images_path}loot1.png', f'{images_path}loot2.png']
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
#endregion

#region Обявление переменных
screenshot_counter = 0
screenshot_directory = "./screenshots" 
# Максимальное количество сохраняемых скриншотов
MAX_SCREENSHOTS = 10

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
#endregion

#region Функция для очистки папки со скриншотами
def clear_screenshot_directory(directory=screenshot_directory):
    """Очищает папку со скриншотами при запуске скрипта."""
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Удаление папки со всеми файлами
    os.makedirs(directory)  # Создание пустой папки
#endregion

#region Функция для сохранения скриншота с меткой клика
def save_screenshot_with_marker(screenshot_array, x, y, directory=screenshot_directory, prefix="screenshot", region_size=100):
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
    print(f"Скриншот сохранён: {screenshot_filename}")

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
        print(f"Не удалось получить скриншот без мерцания: {e}")
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
        print("Скриншот не найден.")
        return None, None, None

    # Проход по диапазону масштабов
    for scale in np.arange(lower_scale, upper_scale + step, step):
        found, location, screenshot_with_marker = find_template_in_image(screenshot_array, template_image, scale, threshold=0.75, mark_center=mark_center)
        
        if found:
            # Проверка, если найденное совпадение лучше, чем текущее лучшее
            result = cv2.matchTemplate(screenshot_array, cv2.resize(cv2.imread(template_image), (0, 0), fx=scale, fy=scale), cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_scale = scale
                best_val = max_val
                best_location = location
                best_screenshot = screenshot_with_marker

    if best_scale is not None:
        print(f"Лучший масштаб найден: {best_scale}, уверенность: {best_val:.2f}.")
        return best_scale, best_location, best_screenshot
    else:
        print("Шаблон не найден ни на одном из масштабов.")
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
            print(f"Ошибка при получении заголовка страницы: {e}")
            # Если контекст выполнения разрушен, продолжаем искать дальше
            continue
    return None
#endregion

#region Функция для эмуляции нажатия клавиши Esc в фрейме
def press_escape(frame):
    try:
        # Эмулируем нажатие клавиши Esc
        frame.keyboard.press("Escape")
        print("Клавиша Escape нажата.")
    except Exception as e:
        print(f"Ошибка при нажатии клавиши Escape: {e}")
#endregion

#region Функция для прокрутки колесом мыши внутри фрейма
def scroll_wheel(frame, delta_y, steps=10):
    try:
        # Получаем размеры окна браузера (это может быть полезно для расчета центра)
        window_size = frame.evaluate("({ width: window.innerWidth, height: window.innerHeight })")
        center_x = window_size['width'] / 2
        center_y = window_size['height'] / 2

        # Перемещаем виртуальный курсор в центр игрового поля
        frame.mouse.move(center_x, center_y)

        # Разбиваем прокрутку на несколько шагов
        step_size = delta_y // steps  # Количество пикселей для прокрутки за один шаг
        for _ in range(steps):
            frame.mouse.wheel(0, step_size)
            time.sleep(0.05)  # Небольшая задержка между прокрутками

        direction = "вниз" if delta_y > 0 else "вверх"
        print(f"Колесо прокручено {direction} в main_frame на {delta_y} пикселей в {steps} шагов.")
    except Exception as e:
        print(f"Ошибка при прокрутке колесом в main_frame: {e}")
#endregion

#region Функция поиска статического объекта
def find_template(page, template_image, best_scale, threshold=0.8):
    """Функция поиска статического объекта."""
    
    screenshot_array = get_screenshot(page)
    if screenshot_array is None:
        print("Ошибка: скриншот не найден.")
        return False, None, None

    found, location, marked_screenshot = find_template_in_image(screenshot_array, template_image, best_scale, threshold=threshold, mark_center=True)

    if found:
        x, y = location
        print(f"Шаблон найден в координатах ({x}, {y}) с использованием масштаба {best_scale}.")
        return True, location, marked_screenshot
    else:
        return False, None, None
#endregion

#region Функция для клика по статическому объекту с использованием найденного масштаба
def click_by_pos(page, x, y):
    """Функция клика по статическому объекту по координатам."""

    page.mouse.move(x, y)
    page.mouse.click(x, y)

    print(f"Клик по статическому объекту в координатах ({x}, {y}).")
#endregion

#region Функция для клика по статическому объекту с использованием найденного масштаба
def click_static_template(page, template_image, best_scale, threshold=0.8, delay_time=0, offset_x=0, offset_y=0):
    """Функция клика по статическому объекту с сохранением скриншота с меткой."""

    time.sleep(delay_time)
    found, location, marked_screenshot = find_template(page=page, template_image=template_image, best_scale=best_scale, threshold=threshold)

    if found:
        x, y = location
        x += offset_x
        y += offset_y
        page.mouse.move(x, y)
        page.mouse.click(x, y)

        # Сохраняем скриншот с меткой
        save_screenshot_with_marker(marked_screenshot, x, y, screenshot_directory)

        print(f"Клик по статическому объекту в координатах ({x}, {y}) с использованием масштаба {best_scale}.")
        return True, x, y
    else:
        print(f"Шаблон не найден на статическом объекте с масштабом {best_scale}.")
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
        print(f"Ошибка: не удалось загрузить изображение {template_image}.")
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
        print("Ошибка: скриншот не найден.")
        return False

    found, location1, marked_screenshot1 = find_template_in_image(screenshot_array, template_image, best_scale, threshold=threshold, mark_center=False)

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

    print(f"Область интереса для второго скриншота: ({roi_left}, {roi_top}) - ({roi_right}, {roi_bottom})")

    # Второй скриншот
    time.sleep(0.1)  # Короткая пауза для фиксации движения
    screenshot_array2 = get_screenshot(page)
    if screenshot_array2 is None:
        return False

    # Ограничиваем изображение второй области интереса
    roi_screenshot = screenshot_array2[roi_top:roi_bottom, roi_left:roi_right]

    # Поиск шаблона в области второго скриншота
    found, location2, marked_screenshot2 = find_template_in_image(roi_screenshot, template_image, best_scale, threshold=threshold, mark_center=False)

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

    print(f"Клик по движущемуся объекту в предполагаемых координатах ({adjusted_x}, {adjusted_y}) с использованием масштаба {best_scale}.")

    # Сохранение скриншота с маркерами при необходимости
    if save_screenshot:
        # Используем функцию save_screenshot_with_marker для сохранения скриншота с отметками
        screenshot_counter = save_screenshot_with_marker(screenshot_array2, adjusted_x, adjusted_y, region_size=search_radius)

    press_escape(page)
    return True
#endregion

#region Стартовые приготовления
# Создание папки для скриншотов, если её нет
if not os.path.exists(screenshot_directory):
    os.makedirs(screenshot_directory)

# Очистка папки со скриншотами при запуске
clear_screenshot_directory(screenshot_directory)
#endregion

#region Функция закрытия рекламы
def close_advert(page, best_scale):
    no_ads = click_static_template(page, failed_ads_image, best_scale=best_scale)
    if no_ads[0]:
        time.sleep(20)
        click_static_template(page, close_failed_ads_image, best_scale=best_scale)
        time.sleep(5)
    else:
        time.sleep(30)
        press_escape(page)
#endregion

#region Перезагрузка страницы
def reload_page(page):
    try:
        page.reload()
        print("Страница успешно перезагружена.")
    except Exception as e:
        print(f"Ошибка при перезагрузке страницы: {e}")
#endregion

#region Сбор монеток на станции
def collect_coins(page, x, y, best_scale):
    click_by_pos(page,x,y)
    time.sleep(0.5)
    click_static_template(page, collect_all_image, best_scale, threshold=0.85)
    press_escape(page)
#endregion

#region Логика скрипта
with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp("http://localhost:8888")
    context = browser.contexts[0]  # Берем первый контекст (окно)

    # Пытаемся переключиться на вкладку с нужным заголовком
    page = switch_to_tab(context, "Play Trainstation 2 on PC")
    
    if not page:
        # Если вкладка не найдена, открываем новую
        page = context.new_page()
        page.goto("https://portal.pixelfederation.com/en/trainstation2")
    
    # Выполняем скроллинг в пределах фрейма на 7% видимого окна
    #scroll_wheel(page, delta_y=300, steps=3)

    # Прокрутка колесом вниз (например, для уменьшения масштаба) внутри фрейма
    #scroll_wheel_in_main_frame(page, delta_y=300, steps=3)

    # Запуск функции поиска наилучшего масштаба
    best_scale, best_location, screenshot_with_marker = find_best_scale(page, build_shop_image, lower_scale=0.5, upper_scale=2.0, step=0.1, mark_center=True)

    # Бесконечный цикл
    while True:
        current_time = time.time()

        # Проверка и нажатие на loot изображения каждые 5 секунд
        if current_time - last_loot_time >= loot_interval:
            for loot_image in loot_images:
                click_moving_template(page, loot_image, best_scale, threshold=0.6)
            last_loot_time = current_time

        # Проверка и нажатие на advert изображение каждые 5 секунд
        if current_time - last_advert_time >= advert_interval:
            result = click_moving_template(page, advert_image, best_scale, threshold=0.6, save_screenshot=True)
            if result:
                close_advert(page, best_scale)

            last_advert_time = current_time

        # Проверка на запуск второй копии игры каждые 30 секунд
        if current_time - last_restart >= restart_interval:
            # Проверка и нажатие на restart изображение если оно есть, значит есть второй вход - ждём 5 минут
            result = find_template(page, restart_image, best_scale)
            if result:
                if result[0]:
                    print("Есть второй вход - ждём 5 минут.")
                    time.sleep(299)
                    reload_page(page)
                    time.sleep(120)
                    #perform_mouse_scroll(page, distance_percentage=0.11)
                    scroll_wheel(page, delta_y=300, steps=3)

            last_restart_time = current_time

        # Проверка и нажатие на station_coin изображение каждые 30 минут
        if current_time - last_station_coin_time >= station_coin_interval:
            found, station_x, station_y = click_static_template(page, station_coin_image, best_scale, threshold=0.85)

            found = True
            if found:
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
                collect_coins(page, station_x, station_y, best_scale)
                last_station_collect_coin_time = -1

        # Проверка и нажатие на basket изображение каждые 30 минут
        if current_time - last_basket_time >= basket_interval:
            result = click_static_template(page, basket_image, best_scale)
            if result:
                time.sleep(1)
                # Тут должен быть сборка шестерёнок с корзины в том числе с рекламой
                # Нажатие на кнопку сбора безрекламных шестерёнок
                result = click_static_template(page, basket_free_image, best_scale, offset_y=160, threshold=0.85)
                if result:
                    # Нажатие на кнопку сбора безрекламных шестерёнок
                    result = click_static_template(page, start_free_gear_image, best_scale, offset_y=160)
                    if result:
                        time.sleep(5)
                        # Нажатие на кнопку сбора шестерёнок
                        click_static_template(page, collect_gear_image, best_scale)
                        time.sleep(3)
                    press_escape(page)
                # Нажатие на кнопку сбора рекламных шестерёнок
                result = click_static_template(page, basket_advert_image, best_scale, offset_y=160)
                if result:
                    # Нажатие на кнопку сбора рекламных шестерёнок
                    result = click_static_template(page, start_advert_gear_image, best_scale, offset_y=160)
                    if result:
                        time.sleep(2)
                        close_advert(page, best_scale)
                        # Нажатие на кнопку сбора шестерёнок
                        click_static_template(page, collect_gear_image, best_scale)
                        time.sleep(3)
                        press_escape(page)
                # Закрытие окна корзины
                press_escape(page)
                last_basket_time = current_time

        # Задержка перед следующей итерацией
        time.sleep(1)
#endregion