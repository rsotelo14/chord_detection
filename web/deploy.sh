#!/bin/bash

# Script de despliegue automatizado para EC2
# Uso: ./deploy.sh

set -e  # Salir si hay algún error

echo "🚀 Iniciando despliegue de Chord Detection Web..."

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar que estamos en el directorio correcto
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$SCRIPT_DIR"

if [ ! -f "app.py" ]; then
    echo -e "${RED}❌ Error: Debes ejecutar este script desde el directorio web/${NC}"
    exit 1
fi

# Verificar que existe .env
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  Advertencia: No se encontró archivo .env${NC}"
    echo "   Creando .env desde env.example..."
    cp env.example .env
    echo -e "${YELLOW}   Por favor, edita .env con tus credenciales antes de continuar${NC}"
    exit 1
fi

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Error: Python3 no está instalado${NC}"
    exit 1
fi

# Usar el entorno virtual de la raíz del proyecto
if [ ! -d "$PROJECT_ROOT/env" ]; then
    echo -e "${YELLOW}⚠️  No se encontró entorno virtual en la raíz del proyecto${NC}"
    echo "   Creando entorno virtual en $PROJECT_ROOT/env..."
    python3 -m venv "$PROJECT_ROOT/env"
fi

# Activar entorno virtual de la raíz
echo "🔌 Activando entorno virtual desde la raíz del proyecto..."
source "$PROJECT_ROOT/env/bin/activate"

# Actualizar pip
echo "⬆️  Actualizando pip..."
pip install --upgrade pip --quiet

# Instalar dependencias del proyecto web
echo "📥 Instalando dependencias de la aplicación web..."
pip install -r requirements.txt --quiet

# Instalar dependencias del proyecto principal si existe
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    echo "📥 Instalando dependencias del proyecto principal..."
    pip install -r "$PROJECT_ROOT/requirements.txt" --quiet
fi

# Verificar que los modelos existan
if [ ! -d "../analysis_out_frames" ]; then
    echo -e "${YELLOW}⚠️  Advertencia: No se encontró el directorio analysis_out_frames${NC}"
    echo "   Asegúrate de que los modelos entrenados estén disponibles"
fi

# Crear directorio uploads si no existe
mkdir -p uploads

# Verificar que Gunicorn esté instalado
if ! pip show gunicorn &> /dev/null; then
    echo "📦 Instalando Gunicorn..."
    pip install gunicorn --quiet
fi

echo -e "${GREEN}✅ Despliegue completado exitosamente!${NC}"
echo ""
echo "Para ejecutar la aplicación:"
echo "  cd $SCRIPT_DIR"
echo "  source $PROJECT_ROOT/env/bin/activate"
echo "  gunicorn -c gunicorn_config.py app:app"
echo ""
echo "O para desarrollo:"
echo "  cd $SCRIPT_DIR"
echo "  source $PROJECT_ROOT/env/bin/activate"
echo "  python app.py"

