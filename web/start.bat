@echo off
REM Script para iniciar la interfaz web del detector de acordes en Windows

echo 🎸 Iniciando Detector de Acordes...

REM Cambiar al directorio del script
cd /d "%~dp0"

REM Verificar si el entorno virtual existe
if not exist "..\env" (
    echo ❌ Error: No se encontró el entorno virtual.
    echo    Por favor, crea el entorno virtual primero desde el directorio raíz del proyecto.
    pause
    exit /b 1
)

REM Activar entorno virtual
echo 📦 Activando entorno virtual...
call ..\env\Scripts\activate.bat

REM Verificar si Flask está instalado
python -c "import flask" 2>nul
if errorlevel 1 (
    echo 📥 Instalando dependencias de Flask...
    pip install -r requirements.txt
)

REM Verificar que existen los recursos necesarios
if not exist "..\analysis_out\baseline_mlp_model.h5" (
    echo ⚠️  Advertencia: No se encontró el modelo entrenado.
    echo    Asegúrate de haber entrenado el modelo antes de usar la interfaz web.
)

REM Crear directorio uploads si no existe
if not exist "uploads" mkdir uploads

REM Iniciar servidor
echo 🚀 Iniciando servidor en http://localhost:5000
echo    Presiona Ctrl+C para detener el servidor
echo.
python app.py

pause
































