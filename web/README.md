# Interfaz Web - Detector de Acordes

Interfaz web para detectar acordes de canciones en tiempo real.

## Instalación

1. Asegúrate de tener el entorno virtual activado del proyecto principal:
```bash
cd ..
source env/bin/activate  # En Windows: env\Scripts\activate
```

2. Instala las dependencias adicionales para el servidor web:
```bash
pip install -r web/requirements.txt
```

3. Configura las variables de entorno para AWS Cognito:

**Opción recomendada:** Usar archivo `.env`
```bash
cd web
cp env.example .env
# Edita .env con tus credenciales
```

**Alternativa:** Variables de entorno del sistema
```bash
export AWS_ACCESS_KEY_ID=tu_access_key
export AWS_SECRET_ACCESS_KEY=tu_secret_key
export AWS_DEFAULT_REGION=us-east-1
export COGNITO_CLIENT_ID=tu_client_id  # Opcional
```

📖 **Para más detalles sobre dónde encontrar cada valor, consulta [CONFIGURACION.md](CONFIGURACION.md)**

⚠️ **Si encuentras errores, consulta [TROUBLESHOOTING.md](TROUBLESHOOTING.md) para soluciones comunes**

## Uso

1. Desde el directorio `web/`, ejecuta:
```bash
python app.py
```

2. Abre tu navegador en: `http://localhost:5000`

3. Sube un archivo de audio (MP3, WAV, OGG, M4A, FLAC)

4. Espera a que el modelo procese la canción

5. ¡Reproduce y toca junto con los acordes detectados!

## Características

- 🔐 Autenticación con AWS Cognito (login y registro)
- ✨ Interfaz moderna y responsive
- 🎵 Reproductor de audio integrado
- 🎸 Acordes sincronizados en tiempo real
- 📊 Línea de tiempo interactiva
- 🖱️ Drag & drop para subir archivos
- 🔄 Procesamiento automático con el modelo DNN

## Autenticación

La aplicación requiere autenticación para usar el detector de acordes:

1. **Registro**: Crea una cuenta nueva con tu correo electrónico y contraseña
2. **Verificación**: Verifica tu correo electrónico con el código que recibirás
3. **Login**: Inicia sesión con tus credenciales
4. **Uso**: Una vez autenticado, podrás subir y procesar archivos de audio

### Configuración de AWS Cognito

- **User Pool ID**: `us-east-1_o7y7iPcVz`
- **Región**: `us-east-1`
- El Client ID se obtiene automáticamente, pero puede configurarse manualmente con `COGNITO_CLIENT_ID`

## Limitaciones

- Tamaño máximo de archivo: 50MB
- Formatos soportados: MP3, WAV, OGG, M4A, FLAC
- Timeout de procesamiento: 5 minutos

## Despliegue en EC2

Para desplegar la aplicación en AWS EC2, consulta la guía completa:

📖 **[DEPLOY_EC2.md](DEPLOY_EC2.md)** - Guía paso a paso para desplegar en EC2

### Resumen Rápido:

1. **Crear instancia EC2** (Amazon Linux o Ubuntu)
2. **Conectarse por SSH**
3. **Clonar repositorio**: `git clone https://github.com/rsotelo14/chord_detection.git`
4. **Ejecutar script de despliegue**: `cd chord_detection/web && ./deploy.sh`
5. **Configurar `.env`** con tus credenciales
6. **Ejecutar con Gunicorn**: `gunicorn -c gunicorn_config.py app:app`
7. **Configurar Security Group** para permitir tráfico HTTP (puerto 80)

Para más detalles y opciones avanzadas (Nginx, SSL, systemd), consulta [DEPLOY_EC2.md](DEPLOY_EC2.md).



































