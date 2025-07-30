@echo off
echo ====================================
echo WebPage AI Chatbot Extension Setup
echo ====================================
echo.

echo Checking if Chrome is installed...
where chrome >nul 2>nul
if %errorlevel% neq 0 (
    echo Chrome not found in PATH. Please ensure Chrome is installed.
    pause
    exit /b 1
)

echo Chrome found!
echo.

echo Opening Chrome Extensions page...
start chrome --new-window "chrome://extensions/"

echo.
echo ====================================
echo SETUP INSTRUCTIONS:
echo ====================================
echo 1. Enable 'Developer mode' (top-right toggle)
echo 2. Click 'Load unpacked'
echo 3. Select the 'WebPageChatbot_Extension' folder
echo 4. Click 'Select Folder'
echo.
echo The extension should now be loaded and ready to use!
echo.

echo Current extension folder location:
echo %~dp0WebPageChatbot_Extension
echo.

echo Press any key to open the extension folder...
pause >nul
explorer "%~dp0WebPageChatbot_Extension"

echo.
echo Setup complete! The extension should now be visible in Chrome.
echo.
pause
