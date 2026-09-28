@echo off
echo ==========================================
echo     Starting MAORS Clinical AI Platform   
echo ==========================================

REM Check if Docker is installed
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] Docker could not be found. Please install Docker Desktop.
    pause
    exit /b 1
)

echo [*] Building and starting Docker containers...
docker-compose up --build -d

echo.
echo [*] MAORS is booting up! Please wait a few seconds.
echo [*] The application will be available at: http://localhost:8080
echo [*] To stop the application, run: docker-compose down
echo ==========================================
pause
