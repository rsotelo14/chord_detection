#!/bin/bash

# Script para iniciar la interfaz web del detector de acordes

echo "🎸 Iniciando Detector de Acordes..."

# Verificar que estamos en el directorio correcto
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Verificar si el entorno virtual existe
if [ ! -d "../env" ]; then
    echo "❌ Error: No se encontró el entorno virtual."
    echo "   Por favor, crea el entorno virtual primero desde el directorio raíz del proyecto."
    exit 1
fi

# Activar entorno virtual
echo "📦 Activando entorno virtual..."
source ../env/bin/activate

# Verificar si Flask está instalado
if ! python -c "import flask" 2>/dev/null; then
    echo "📥 Instalando dependencias de Flask..."
    pip install -r requirements.txt
fi

# Verificar que existen los recursos necesarios
if [ ! -f "../analysis_out/baseline_mlp_model.h5" ]; then
    echo "⚠️  Advertencia: No se encontró el modelo entrenado."
    echo "   Asegúrate de haber entrenado el modelo antes de usar la interfaz web."
fi

# Crear directorio uploads si no existe
mkdir -p uploads

# Iniciar servidor
echo "🚀 Iniciando servidor en http://localhost:5000"
echo "   Presiona Ctrl+C para detener el servidor"
echo ""
python app.py
































