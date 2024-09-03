# Скрипт для автматизации сбора лута и золота для TS2

# ВНИМАНИЕ! Все манипуляции были произведены на Gentoo Linux для других дистрибутивов, нужна будет адаптация под конкретный пакетный менеджер и версии пакетов.
# ВНИМАНИЕ! На устройстве нужно включить редим разработчика и в нём включить "Отладка по USB".

# 1.  Установка зависимостей:
    
    emerge dev-lang/python dev-util/android-studio dev-python/pip net-libs/nodejs dev-java/openjdk-bin
    

# 2.  Установка компонентов Android Studio:
    
    sdkmanager "build-tools;35.0.0" "platforms;android-35"
    
   
# 3.  Добавление путей в переменным окружения:
    
    echo 'export ANDROID_HOME=~/Android/Sdk' >> ~/.bashrc
    echo 'export PATH="$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools:$PATH"' >> ~/.bashrc
    

# 4.  Установка Appium server:
    
    npm install -g appium
    

# 5.  Установка драйверов в Appium для использования в скрипте
    
    appium driver install uiautomator2
    

# 6.  Запуск appium сервера:
    
    appium --allow-insecure=adb_shell
    

# 7.  Создание виртуальной среды для работы в Python:
    
    mkdir ~/appium
    python -m venv ~/appium
    

# 8.  Подключение к созданной виртуальной среде:
    
    source ~/appium
    
   
# 9.  Установка доп. библиотек с помощью pip:
    
    pip install appium-python-client opencv-python selenium
    

# 10. Сканирование подключённых android устройств и поиск имени нужного:
    
    adb devices
    

# 11. Имя устройства нужно прописать в скрипт в строку, вместо слова device:
    
    options.device_name = 'device'
    

# 12. Запуск скрипта:
    
    python ts2.py
    
    

Оба окна, с запущенным сервером и с запущенным скриптом, должны работать, скрипт можно прервать по нажатаю Ctrl-C и запустить снова. Если устройство будет отключено, то сервер тоже нужно будет перезапустить.
