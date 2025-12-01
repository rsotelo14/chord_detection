# Guía de Despliegue en AWS EC2

Esta guía te ayudará a desplegar la aplicación web del Detector de Acordes en una instancia EC2 de AWS.

## Prerrequisitos

- Cuenta de AWS activa
- Acceso a AWS Console
- Clave SSH para conectarte a EC2
- Git instalado localmente

## Paso 1: Crear una Instancia EC2

1. Ve a [AWS EC2 Console](https://console.aws.amazon.com/ec2/)
2. Haz clic en **"Launch Instance"**
3. Configura la instancia:
   - **Name**: `chord-detection-web` (o el nombre que prefieras)
   - **AMI**: Selecciona **Amazon Linux 2023** o **Ubuntu 22.04 LTS** (recomendado)
   - **Instance type**: `t2.micro` (gratis en el tier gratuito) o `t2.small` para mejor rendimiento
   - **Key pair**: Crea o selecciona un key pair (necesitarás la clave `.pem` para conectarte)
   - **Network settings**: 
     - Crea un nuevo security group o edita uno existente
     - Agrega reglas de entrada:
       - **SSH (22)** desde tu IP
       - **HTTP (80)** desde cualquier lugar (0.0.0.0/0)
       - **HTTPS (443)** desde cualquier lugar (0.0.0.0/0) - opcional
       - **Custom TCP (5000)** desde cualquier lugar si quieres usar el puerto de Flask directamente
4. Haz clic en **"Launch Instance"**

## Paso 2: Conectarse a la Instancia

### Linux/macOS:
```bash
chmod 400 tu-clave.pem
ssh -i tu-clave.pem ec2-user@tu-ip-publica
# O si usas Ubuntu:
ssh -i tu-clave.pem ubuntu@tu-ip-publica
```

### Windows:
Usa PuTTY o WSL con el mismo comando de Linux.

**Nota:** Reemplaza `tu-clave.pem` con la ruta a tu archivo de clave y `tu-ip-publica` con la IP pública de tu instancia EC2.

## Paso 3: Actualizar el Sistema

```bash
# Para Amazon Linux 2023:
sudo yum update -y

# Para Ubuntu:
sudo apt update && sudo apt upgrade -y
```

## Paso 4: Instalar Dependencias del Sistema

### Amazon Linux 2023:
```bash
sudo yum install -y python3 python3-pip git
```

### Ubuntu:
```bash
sudo apt install -y python3 python3-pip python3-venv git
```

## Paso 5: Clonar el Repositorio

```bash
cd ~
git clone https://github.com/rsotelo14/chord_detection.git
cd chord_detection/web
```

## Paso 6: Crear Entorno Virtual en la Raíz

```bash
cd ~/chord_detection
python3 -m venv env
source env/bin/activate
```

## Paso 7: Instalar Dependencias de Python

```bash
# Desde la raíz del proyecto (con el entorno virtual activado)
pip install --upgrade pip

# Instalar dependencias del proyecto principal (TensorFlow, librosa, etc.)
pip install -r requirements.txt

# Instalar dependencias de la aplicación web
cd web
pip install -r requirements.txt
```

## Paso 8: Instalar Dependencias del Sistema para Audio (opcional)

Si necesitas procesar archivos de audio, instala las librerías necesarias:

### Amazon Linux 2023:
```bash
sudo yum install -y gcc gcc-c++ make libffi-devel
```

### Ubuntu:
```bash
sudo apt install -y build-essential libffi-dev
```

## Paso 9: Configurar Variables de Entorno

```bash
cd ~/chord_detection/web
cp env.example .env
nano .env  # o usa vi, vim, o tu editor preferido

# Asegúrate de estar en el directorio web antes de continuar
```

Edita el archivo `.env` con tus credenciales reales:

```env
AWS_ACCESS_KEY_ID=tu_access_key_real
AWS_SECRET_ACCESS_KEY=tu_secret_key_real
AWS_DEFAULT_REGION=us-east-1
COGNITO_CLIENT_ID=tu_client_id_real
COGNITO_CLIENT_SECRET=tu_client_secret_real
SECRET_KEY=genera-una-clave-secreta-segura-aqui
```

**Generar SECRET_KEY:**
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

## Paso 10: Verificar que los Modelos Estén Disponibles

Asegúrate de que los modelos entrenados estén en el repositorio o súbelos manualmente:

```bash
# Verificar que existan los modelos necesarios
ls -la ../analysis_out_frames/*.h5
```

Si no están, necesitarás subirlos o entrenarlos primero.

## Paso 11: Configurar Security Group

1. Ve a EC2 Console → **Security Groups**
2. Selecciona el security group de tu instancia
3. **Inbound rules** → **Edit inbound rules**
4. Agrega reglas si faltan:
   - **Type**: HTTP, **Port**: 80, **Source**: 0.0.0.0/0
   - **Type**: HTTPS, **Port**: 443, **Source**: 0.0.0.0/0 (opcional)
   - **Type**: Custom TCP, **Port**: 5000, **Source**: 0.0.0.0/0 (si usas Flask directamente)

## Paso 12: Ejecutar la Aplicación (Desarrollo)

Para probar rápidamente:

```bash
cd ~/chord_detection/web
source ../env/bin/activate  # Activar entorno virtual de la raíz
python app.py
```

La aplicación estará disponible en `http://tu-ip-publica:5000`

**Nota:** Esto es solo para pruebas. Para producción, usa Gunicorn (ver siguiente paso).

## Paso 13: Configurar Gunicorn para Producción

### Instalar Gunicorn:
```bash
pip install gunicorn
```

### Crear archivo de configuración:
```bash
nano ~/chord_detection/web/gunicorn_config.py
```

Contenido:
```python
bind = "0.0.0.0:5000"
workers = 2
timeout = 300
worker_class = "sync"
```

### Ejecutar con Gunicorn:
```bash
cd ~/chord_detection/web
source ../env/bin/activate  # Activar entorno virtual de la raíz
gunicorn -c gunicorn_config.py app:app
```

## Paso 14: Configurar como Servicio Systemd (Opcional pero Recomendado)

Crea un servicio systemd para que la aplicación se inicie automáticamente:

```bash
sudo nano /etc/systemd/system/chord-detection.service
```

Contenido:
```ini
[Unit]
Description=Chord Detection Web Application
After=network.target

[Service]
User=ec2-user
Group=ec2-user
WorkingDirectory=/home/ec2-user/chord_detection/web
Environment="PATH=/home/ec2-user/chord_detection/env/bin"
ExecStart=/home/ec2-user/chord_detection/env/bin/gunicorn -c gunicorn_config.py app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

**Nota:** Ajusta `User`, `Group` y las rutas según tu usuario (puede ser `ubuntu` en lugar de `ec2-user`).

Habilitar y iniciar el servicio:
```bash
sudo systemctl daemon-reload
sudo systemctl enable chord-detection
sudo systemctl start chord-detection
sudo systemctl status chord-detection
```

## Paso 15: Configurar Nginx como Proxy Reverso (Opcional pero Recomendado)

### Instalar Nginx:

**Amazon Linux 2023:**
```bash
sudo yum install -y nginx
```

**Ubuntu:**
```bash
sudo apt install -y nginx
```

### Configurar Nginx:

```bash
sudo nano /etc/nginx/conf.d/chord-detection.conf
```

Contenido:
```nginx
server {
    listen 80;
    server_name tu-dominio.com tu-ip-publica;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Para archivos grandes (uploads)
        client_max_body_size 50M;
    }
}
```

Iniciar Nginx:
```bash
sudo systemctl start nginx
sudo systemctl enable nginx
sudo nginx -t  # Verificar configuración
```

## Paso 16: Configurar SSL con Let's Encrypt (Opcional)

Si tienes un dominio:

```bash
# Instalar certbot
sudo yum install -y certbot python3-certbot-nginx
# O en Ubuntu:
sudo apt install -y certbot python3-certbot-nginx

# Obtener certificado
sudo certbot --nginx -d tu-dominio.com
```

## Verificación

1. Abre tu navegador y ve a `http://tu-ip-publica` (o `http://tu-dominio.com`)
2. Deberías ver la página de login del Detector de Acordes
3. Prueba registrarte y usar la aplicación

## Comandos Útiles

### Ver logs de la aplicación:
```bash
sudo journalctl -u chord-detection -f
```

### Reiniciar la aplicación:
```bash
sudo systemctl restart chord-detection
```

### Ver logs de Nginx:
```bash
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Detener la aplicación:
```bash
sudo systemctl stop chord-detection
```

## Solución de Problemas

### La aplicación no inicia:
- Verifica que las variables de entorno estén configuradas: `cat .env`
- Verifica los logs: `sudo journalctl -u chord-detection -n 50`
- Verifica que el puerto 5000 esté libre: `sudo netstat -tlnp | grep 5000`

### No puedo acceder desde el navegador:
- Verifica el Security Group: debe permitir tráfico HTTP (puerto 80) o el puerto que uses
- Verifica que la aplicación esté corriendo: `sudo systemctl status chord-detection`
- Verifica los logs de Nginx si lo usas: `sudo tail -f /var/log/nginx/error.log`

### Error de permisos:
- Asegúrate de que el usuario tenga permisos en el directorio: `sudo chown -R ec2-user:ec2-user ~/chord_detection`

## Costos Estimados

- **t2.micro**: Gratis durante 12 meses (si es tu primera vez con AWS)
- **t2.small**: ~$0.02/hora (~$15/mes)
- **Transferencia de datos**: Primeros 100GB gratis, luego ~$0.09/GB

## Seguridad

⚠️ **Importante:**
- Nunca subas el archivo `.env` al repositorio
- Usa Security Groups restrictivos en producción
- Considera usar AWS Secrets Manager para credenciales en producción
- Configura SSL/HTTPS para producción
- Limita el acceso SSH solo desde tu IP

## Próximos Pasos

- Configurar un dominio personalizado
- Configurar backups automáticos
- Configurar monitoreo con CloudWatch
- Configurar auto-scaling si es necesario

