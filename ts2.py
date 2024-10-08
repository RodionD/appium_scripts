import time
from appium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.pointer_input import PointerInput
from selenium.webdriver.common.actions.action_builder import ActionBuilder
import numpy as np
import cv2
from appium.options.android import UiAutomator2Options
import base64
import os
import threading
import shutil

# Настройки для подключения к Appium серверу и запуска приложения
options = UiAutomator2Options()
options.app_package = 'com.pixelfederation.ts2'  # Уникальный идентификатор пакета приложения
options.app_activity = 'com.google.firebase.MessagingUnityPlayerActivity'  # Основная активность приложения
options.device_name = 'device'  # Имя устройства (если требуется)
options.no_reset = True  # Сохраняет данные приложения между запусками
options.full_reset = False  # Полный сброс приложения при запуске отключен


# Функция закрытия рекламы
def close_advert():
    time.sleep(35)
    find_and_tap_image(driver, advert_close1_image)
    time.sleep(10)
    find_and_tap_image(driver, advert_close2_image)

# Функция для очистки папки со скриншотами
def clear_screenshot_directory(directory):
    """Очищает папку со скриншотами при запуске скрипта."""
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Удаление папки со всеми файлами
    os.makedirs(directory)  # Создание пустой папки

def is_app_running(driver, package_name):
    """Функция для проверки, запущено ли приложение на устройстве."""
    output = driver.execute_script("mobile: shell", {
        'command': f'pidof {package_name}',
        'args': [],
        'includeStderr': True,
        'timeout': 5000
    })
    return bool(output.get('stdout', '').strip())  # Если PID найден, возвращается True, иначе False

def start_app(driver, package_name, activity_name):
    """Функция для запуска приложения."""
    driver.execute_script('mobile: startActivity', {
        'appPackage': package_name,
        'appActivity': activity_name
    })

def perform_pinch_or_zoom(driver, action='pinch'):
    """Функция для выполнения масштабирования экрана: уменьшение или увеличение."""
    window_size = driver.get_window_size()  # Получение размеров экрана устройства
    center_x = window_size['width'] / 2
    center_y = window_size['height'] / 2

    if action == 'pinch':
        # Уменьшение масштаба (pinch)
        driver.execute_script('mobile: pinchCloseGesture', {
            'elementId': None,
            'left': center_x - 100,
            'top': center_y - 100,
            'width': 200,
            'height': 200,
            'percent': 0.5,
            'speed': 200
        })
    elif action == 'zoom':
        # Увеличение масштаба (zoom)
        driver.execute_script('mobile: pinchOpenGesture', {
            'elementId': None,
            'left': center_x - 100,
            'top': center_y - 100,
            'width': 200,
            'height': 200,
            'percent': 0.5,
            'speed': 200
        })

def find_and_tap_image(driver, template_path, threshold=0.6, save_screenshot=True, offset_y=0, use_gray=False):
    """
    Функция для поиска изображения на экране и клика по нему с возможностью смещения нажатия вниз от центра.
    
    :param driver: WebDriver instance
    :param template_path: Путь к изображению шаблона
    :param threshold: Порог совпадения для определения успешного поиска
    :param save_screenshot: Сохранять ли скриншот с меткой при успешном клике
    :param offset_y: Смещение вниз от центра найденного шаблона (в пикселях)
    :return: True если изображение найдено и клик выполнен, иначе False
    """
    # Получение скриншота экрана
    screenshot = driver.get_screenshot_as_png()
    screenshot = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
    
    # Загрузка шаблона изображения
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)

    if template is None:
        print(f"Ошибка: шаблон {template_path} изображения не найден.")
        return False, None, None

    if use_gray:
        # Конвертируем изображения в черно-белые
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Сопоставление шаблона с текущим изображением экрана
    result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        # Вычисление координат центра найденного шаблона с учетом смещения
        tap_x = max_loc[0] + template.shape[1] // 2
        tap_y = max_loc[1] + template.shape[0] // 2 + offset_y

        # Выполнение клика по найденным координатам
        finger = PointerInput('touch', "finger")
        actions = ActionBuilder(driver, mouse=finger)
        actions.pointer_action.move_to_location(tap_x, tap_y).pointer_down().pointer_up()
        actions.perform()

        if save_screenshot:
            # Сохранение скриншота с меткой только при успешном нажатии
            save_screenshot_with_marker(driver, tap_x, tap_y)

        print(f"Изображение '{template_path}' найдено с максимальным значением совпадения {max_val:.2f}. Выполнен клик по координатам ({tap_x}, {tap_y}) со смещением {offset_y} пикселей.")
        return True, tap_x, tap_y
    else:
        print(f"Изображение '{template_path}' не найдено. Максимальное значение совпадения: {max_val:.2f}. Пороговое значение: {threshold}.")
        return False, None, None

def perform_action_and_check_close(driver, action_template, close_template, use_gray=True, use_motion=True):
    """Функция выполняет действие по заданному шаблону и проверяет, появился ли элемент закрытия (close_template)."""
    if use_motion:
        action_successful = find_and_predict_tap_multiscale(driver, action_template, use_gray=use_gray)
    else:
        action_successful = find_and_tap_image(driver, action_template, use_gray=use_gray)
    
    if action_successful:
        time.sleep(2)  # Небольшая пауза перед проверкой close1.png
        # Проверяем наличие close1.png и нажимаем на него, если он найден
        close_found = find_and_tap_image(driver, close_template, save_screenshot=False)
        if close_found:
            return True
        else:
            return False
    else:
        return False

def perform_delayed_actions(driver, station_x, station_y, delay_time):
    """Функция для выполнения отложенных действий через заданное время."""
    time.sleep(delay_time)  # Ожидание указанного времени (например, 7 минут)

    # Нажатие на station_coin.png по старым координатам
    finger = PointerInput('touch', "finger")
    actions = ActionBuilder(driver, mouse=finger)
    actions.pointer_action.move_to_location(station_x, station_y).pointer_down().pointer_up()
    actions.perform()

    # Нажатие на collect_all.png
    find_and_predict_tap_multiscale(driver, collect_all_image, use_gray=False)

    # Нажатие на close1.png после collect_all.png
    find_and_predict_tap_multiscale(driver, close1_image, use_gray=False)

def take_screenshot(driver):
    """Функция для получения скриншота и конвертации его в изображение OpenCV."""
    screenshot = driver.get_screenshot_as_png()
    image = np.frombuffer(screenshot, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def find_and_predict_tap_multiscale(driver, template_path, use_gray=True, threshold=0.60, scales=[1.0, 0.9, 0.8]):
    """Функция для поиска и предсказания нажатия на объект, с учетом возможного изменения его масштаба."""
    # Снимаем первый скриншот
    img1 = take_screenshot(driver)
    
    # Загружаем шаблон
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    
    if use_gray:
        # Конвертируем изображения в черно-белые
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    best_match = None
    best_val = 0
    best_loc = None
    best_scale = 1.0
    
    # Проходим по каждому масштабу
    for scale in scales:
        # Масштабируем шаблон
        resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
        
        # Выполняем сопоставление шаблона
        res = cv2.matchTemplate(img1, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # Сохраняем лучшие результаты
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_match = resized_template
            best_scale = scale

    # Проверяем, если лучший результат превышает порог
    if best_val < threshold:
        print(f"Шаблон {template_path} не найден.")
        return False, None, None
    
    print(f"Шаблон {template_path} найден с масштабом {best_scale} и уверенностью {best_val:.2f}.")
    
    # Снимаем второй скриншот через небольшую задержку
    time.sleep(0.1)
    img2 = take_screenshot(driver)
    
    if use_gray:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Выполняем сопоставление шаблона на втором изображении
    res2 = cv2.matchTemplate(img2, best_match, cv2.TM_CCOEFF_NORMED)
    min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
    
    if max_val2 < threshold:
        print(f"Шаблон {template_path} не найден на втором изображении.")
        return False, None, None
    
    template_height, template_width = template.shape[:2]
    
    # Определяем движение объекта
    delta_x = max_loc2[0] - best_loc[0]
    delta_y = max_loc2[1] - best_loc[1]
    
    # Вычисляем координаты предполагаемого местоположения объекта
    predicted_x = int(max_loc2[0] + delta_x + template_width * best_scale / 2) + 3 * (1 if delta_x > 0 else -1)
    predicted_y = int(max_loc2[1] + delta_y + template_height * best_scale / 2) + 3 * (1 if delta_y > 0 else -1)

    print(f"Тап по координатам ({predicted_x}, {predicted_y}).")

    # Выполнение клика с использованием PointerInput
    finger = PointerInput('touch', "finger")
    actions = ActionBuilder(driver, mouse=finger)
    actions.pointer_action.move_to_location(predicted_x, predicted_y).pointer_down().pointer_up()
    actions.perform()

    # Сохранение скриншота с меткой только при успешном нажатии
    save_screenshot_with_marker(driver, predicted_x, predicted_y)

    return True, predicted_x, predicted_y

def press_esc_key(driver):
    """
    Функция для нажатия физической клавиши Esc на устройстве.
    
    :param driver: WebDriver instance
    """
    driver.execute_script("mobile: shell", {
        'command': 'input',
        'args': ['keyevent', '111'],
        'includeStderr': True,
        'timeout': 5000
    })
    print("Нажата клавиша Esc.")
    
def save_screenshot_with_marker(driver, tap_x, tap_y):
    """Функция для сохранения скриншота с меткой нажатия."""
    global screenshot_counter
    screenshot_path = os.path.join(screenshot_directory, f"screenshot_{screenshot_counter % 10}.png")
    
    screenshot = driver.get_screenshot_as_base64()
    nparr = np.frombuffer(base64.b64decode(screenshot), np.uint8)
    img_rgb = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Добавление метки на изображении
    cv2.circle(img_rgb, (tap_x, tap_y), 10, (0, 0, 255), -1)  # Красная точка

    # Сохранение изображения
    cv2.imwrite(screenshot_path, img_rgb)
    screenshot_counter += 1
    print(f"Скриншот с меткой сохранен: {screenshot_path}")

# Запуск сессии WebDriver
driver = webdriver.Remote('http://127.0.0.1:4723', options=options)

# Проверяем, запущено ли приложение
if not is_app_running(driver, options.app_package):
    print("Приложение не запущено, выполняется запуск.")
    start_app(driver, options.app_package, options.app_activity)
    time.sleep(30)  # Ожидаем запуска приложения
else:
    print("Приложение уже запущено, продолжаем работу.")

# Выполняем масштабирование (уменьшение масштаба)
perform_pinch_or_zoom(driver, action='pinch')

# Путь к изображениям
world_image = './images/world.png'
loot_images = ['./images/loot1.png', './images/loot2.png']
advert_image = './images/advert.png'
advert_close1_image = './images/advert_close1.png'
advert_close2_image = './images/advert_close2.png'
station_coin_image = './images/station_coin.png'
close1_image = './images/close1.png'
restart_image = './images/restart.png'
collect_all_image = './images/collect_all.png'
dispatch_all_image = './images/dispatch_all.png'
basket_image = './images/basket.png'
basket_free_image = './images/basket_free.png'
basket_advert_image = './images/basket_advert.png'
start_free_gear_image = './images/start_free_gear.png'
start_advert_gear_image = './images/start_advert_gear.png'
collect_free_gear_image = './images/collect_free_gear.png'
station_image = './images/station.png'

screenshot_counter = 0
screenshot_directory = "./screenshots" 

# Время в секундах для интервалов
loot_interval = 5  # 5 секунд
advert_interval = 5  # 30 секунд
basket_interval = 1800 # 30 минут
station_coin_interval = 1800  # 30 минут
restart_interval = 30 # 30 секунд

# Время последнего выполнения
last_loot_time = time.time()
last_advert_time = time.time()
last_basket_time = time.time() - 900 # Запуск первой проверки через 15 минут после запуска скрипта
last_station_coin_time = time.time() - 1800 # Запуск первой проверки сразу после запуска скрипта
last_restart = time.time()

# Создание папки для скриншотов, если её нет
if not os.path.exists(screenshot_directory):
    os.makedirs(screenshot_directory)

# Очистка папки со скриншотами при запуске
clear_screenshot_directory(screenshot_directory)

# Возвращаемся на станцию, если накуролесили руками
find_and_tap_image(driver, close1_image)
find_and_tap_image(driver, station_image)
find_and_tap_image(driver, close1_image)
find_and_tap_image(driver, station_image)

# Бесконечный цикл
while True:
    current_time = time.time()

    # Проверка и нажатие на loot изображения каждые 5 секунд
    if current_time - last_loot_time >= loot_interval:
        for loot_image in loot_images:
            perform_action_and_check_close(driver, loot_image, use_gray=False, use_motion=True, close_template=close1_image)
        last_loot_time = current_time

    # Проверка и нажатие на advert изображение каждые 5 секунд
    if current_time - last_advert_time >= advert_interval:
        result = find_and_tap_image(driver, advert_image)
        if result:
            close_advert
        last_advert_time = current_time

    # Проверка и нажатие на basket изображение каждые 30 минут
    if current_time - last_basket_time >= basket_interval:
        result = find_and_tap_image(driver, basket_image)
        if result:
            time.sleep(1)
            # Тут должен быть сборка шестерёнок с корзины в том числе с рекламой
            # Нажатие на кнопку сбора безрекламных шестерёнок
            result = find_and_tap_image(driver, basket_free_image, offset_y=160, threshold=0.85)
            if result:
                # Нажатие на кнопку сбора безрекламных шестерёнок
                result = find_and_tap_image(driver, start_free_gear_image, offset_y=200)
                if result:
                    time.sleep(1)
                    # Нажатие на кнопку старта писка безрекламных шестерёнок
                    result = find_and_tap_image(driver, start_free_gear_image)
                    if result:
                        time.sleep(6)
                        # Нажатие на кнопку сбора шестерёнок
                        find_and_tap_image(driver, collect_free_gear_image)
                        time.sleep(1)
                        find_and_tap_image(driver, close1_image)

            # Нажатие на кнопку сбора рекламных шестерёнок
            result = find_and_tap_image(driver, basket_advert_image, offset_y=160, threshold=0.85)
            if result:
                # Нажатие на кнопку сбора рекламных шестерёнок
                result = find_and_tap_image(driver, start_advert_gear_image, offset_y=200)
                if result:
                    close_advert
                    # Нажатие на кнопку старта писка рекламных шестерёнок
                    result = find_and_tap_image(driver, start_free_gear_image)
                    if result:
                        # Нажатие на кнопку сбора шестерёнок
                        find_and_tap_image(driver, collect_free_gear_image)
                        time.sleep(1)
                        find_and_tap_image(driver, close1_image)
            # Закрытие окна корзины
            find_and_tap_image(driver, close1_image)
            last_basket_time = current_time

    # Проверка и нажатие на station_coin изображение каждые 30 минут
    if current_time - last_station_coin_time >= station_coin_interval:
        found, station_x, station_y = find_and_tap_image(driver, station_coin_image, use_gray=False, threshold=0.75)

        if found:
            # Поиск и нажатие на dispatch_all.png внутри station_coin.png
            result = find_and_tap_image(driver, dispatch_all_image, use_gray=False)

            # Нажатие на close1.png после dispatch_all
            find_and_tap_image(driver, close1_image, use_gray=False)

            if result:
                # Запуск отложенных действий через 7 минут в отдельном потоке
                threading.Thread(target=perform_delayed_actions, args=(driver, station_x, station_y, 420)).start()

        last_station_coin_time = current_time

    # Проверка на запуск второй копии игры каждые 30 секунд
    if current_time - last_restart >= restart_interval:
        # Проверка и нажатие на restart изображение если оно есть, значит есть второй вход - ждём 5 минут
        if find_and_tap_image(driver, restart_image):
            print("Есть второй вход - ждём 5 минут.")
            time.sleep(299)

    # Задержка перед следующей итерацией
    time.sleep(1)
