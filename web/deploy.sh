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

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno virtual
echo "🔌 Activando entorno virtual..."
source venv/bin/activate

# Actualizar pip
echo "⬆️  Actualizando pip..."
pip install --upgrade pip --quiet

# Instalar dependencias
echo "📥 Instalando dependencias..."
pip install -r requirements.txt --quiet

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
echo "  source venv/bin/activate"
echo "  gunicorn -c gunicorn_config.py app:app"
echo ""
echo "O para desarrollo:"
echo "  python app.py"

