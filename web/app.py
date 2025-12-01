import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from werkzeug.utils import secure_filename
from functools import wraps
import subprocess
import re
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import hmac
import hashlib
import base64

# Cargar variables de entorno desde archivo .env
load_dotenv()

# Agregar el directorio raíz al path para importar inference
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# AWS Cognito configuration
COGNITO_USER_POOL_ID = 'us-east-1_o7y7iPcVz'
COGNITO_REGION = 'us-east-1'
COGNITO_CLIENT_ID = os.environ.get('COGNITO_CLIENT_ID')  # Intentar obtener de variable de entorno primero
COGNITO_CLIENT_SECRET = os.environ.get('COGNITO_CLIENT_SECRET')  # Client Secret si está habilitado

# Inicializar cliente de Cognito
cognito_client = boto3.client('cognito-idp', region_name=COGNITO_REGION)

# Función para calcular SECRET_HASH cuando el Client tiene un secret
def calculate_secret_hash(username):
    """
    Calcula el SECRET_HASH requerido cuando el App Client tiene un Client Secret.
    El hash se calcula como: HMAC-SHA256(username + client_id, client_secret)
    """
    if not COGNITO_CLIENT_SECRET:
        return None
    
    message = username + COGNITO_CLIENT_ID
    dig = hmac.new(
        COGNITO_CLIENT_SECRET.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).digest()
    return base64.b64encode(dig).decode('utf-8')

# Obtener el Client ID del User Pool si no está configurado
if not COGNITO_CLIENT_ID:
    try:
        response = cognito_client.describe_user_pool(UserPoolId=COGNITO_USER_POOL_ID)
        app_clients = cognito_client.list_user_pool_clients(UserPoolId=COGNITO_USER_POOL_ID, MaxResults=1)
        if app_clients['UserPoolClients']:
            COGNITO_CLIENT_ID = app_clients['UserPoolClients'][0]['ClientId']
            print(f"Client ID obtenido automáticamente: {COGNITO_CLIENT_ID}")
    except Exception as e:
        print(f"Warning: No se pudo obtener el Client ID automáticamente: {e}")
        print("Por favor, configura COGNITO_CLIENT_ID como variable de entorno")

if not COGNITO_CLIENT_ID:
    print("ERROR: COGNITO_CLIENT_ID no está configurado. La autenticación no funcionará.")

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Decorador para proteger rutas que requieren autenticación
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required
def index():
    return render_template('index.html', user_email=session.get('user_email'))

@app.route('/login')
def login():
    if 'user_email' in session:
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/register')
def register():
    if 'user_email' in session:
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/verify')
def verify():
    if 'user_email' in session:
        return redirect(url_for('index'))
    email = request.args.get('email')
    if not email:
        return redirect(url_for('register'))
    return render_template('verify.html', email=email)

@app.route('/api/signup', methods=['POST'])
def api_signup():
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        confirm_password = data.get('confirm_password')
        
        if not email or not password or not confirm_password:
            return jsonify({'error': 'Todos los campos son requeridos'}), 400
        
        if password != confirm_password:
            return jsonify({'error': 'Las contraseñas no coinciden'}), 400
        
        if len(password) < 8:
            return jsonify({'error': 'La contraseña debe tener al menos 8 caracteres'}), 400
        
        if not COGNITO_CLIENT_ID:
            return jsonify({'error': 'Configuración de autenticación no disponible'}), 500
        
        # Preparar parámetros para sign_up
        sign_up_params = {
            'ClientId': COGNITO_CLIENT_ID,
            'Username': email,
            'Password': password,
            'UserAttributes': [
                {'Name': 'email', 'Value': email}
            ]
        }
        
        # Agregar SECRET_HASH si el Client tiene un secret
        secret_hash = calculate_secret_hash(email)
        if secret_hash:
            sign_up_params['SecretHash'] = secret_hash
        
        # Registrar usuario en Cognito
        response = cognito_client.sign_up(**sign_up_params)
        
        return jsonify({
            'success': True,
            'message': 'Usuario registrado. Por favor verifica tu correo electrónico.',
            'email': email
        })
    
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        
        if error_code == 'UsernameExistsException':
            return jsonify({'error': 'Este correo electrónico ya está registrado'}), 400
        elif error_code == 'InvalidPasswordException':
            return jsonify({'error': 'La contraseña no cumple con los requisitos de seguridad'}), 400
        elif error_code == 'InvalidParameterException':
            return jsonify({'error': 'El correo electrónico no es válido'}), 400
        elif error_code == 'NotAuthorizedException' and 'SignUp' in error_message:
            return jsonify({
                'error': 'El registro no está permitido en este User Pool. Por favor, contacta al administrador o verifica la configuración de Cognito.',
                'details': 'El User Pool o el App Client no tienen habilitado el permiso de registro. Consulta CONFIGURACION.md para más información.'
            }), 403
        else:
            return jsonify({'error': f'Error al registrar usuario: {error_message}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Error inesperado: {str(e)}'}), 500

@app.route('/api/verify', methods=['POST'])
def api_verify():
    try:
        data = request.json
        email = data.get('email')
        code = data.get('code')
        
        if not email or not code:
            return jsonify({'error': 'Email y código son requeridos'}), 400
        
        if not COGNITO_CLIENT_ID:
            return jsonify({'error': 'Configuración de autenticación no disponible'}), 500
        
        # Preparar parámetros para confirm_sign_up
        confirm_params = {
            'ClientId': COGNITO_CLIENT_ID,
            'Username': email,
            'ConfirmationCode': code
        }
        
        # Agregar SECRET_HASH si el Client tiene un secret
        secret_hash = calculate_secret_hash(email)
        if secret_hash:
            confirm_params['SecretHash'] = secret_hash
        
        # Confirmar registro con código de verificación
        response = cognito_client.confirm_sign_up(**confirm_params)
        
        return jsonify({
            'success': True,
            'message': 'Cuenta verificada exitosamente. Ya puedes iniciar sesión.'
        })
    
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        
        if error_code == 'CodeMismatchException':
            return jsonify({'error': 'El código de verificación es incorrecto'}), 400
        elif error_code == 'ExpiredCodeException':
            return jsonify({'error': 'El código de verificación ha expirado. Solicita uno nuevo.'}), 400
        elif error_code == 'NotAuthorizedException':
            return jsonify({'error': 'Este usuario ya está verificado'}), 400
        else:
            return jsonify({'error': f'Error al verificar: {error_message}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Error inesperado: {str(e)}'}), 500

@app.route('/api/resend_code', methods=['POST'])
def api_resend_code():
    try:
        data = request.json
        email = data.get('email')
        
        if not email:
            return jsonify({'error': 'Email es requerido'}), 400
        
        if not COGNITO_CLIENT_ID:
            return jsonify({'error': 'Configuración de autenticación no disponible'}), 500
        
        # Preparar parámetros para resend_confirmation_code
        resend_params = {
            'ClientId': COGNITO_CLIENT_ID,
            'Username': email
        }
        
        # Agregar SECRET_HASH si el Client tiene un secret
        secret_hash = calculate_secret_hash(email)
        if secret_hash:
            resend_params['SecretHash'] = secret_hash
        
        # Reenviar código de verificación
        response = cognito_client.resend_confirmation_code(**resend_params)
        
        return jsonify({
            'success': True,
            'message': 'Código de verificación reenviado a tu correo electrónico.'
        })
    
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        
        if error_code == 'UserNotFoundException':
            return jsonify({'error': 'Usuario no encontrado'}), 404
        elif error_code == 'InvalidParameterException':
            return jsonify({'error': 'Email inválido'}), 400
        else:
            return jsonify({'error': f'Error al reenviar código: {error_message}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Error inesperado: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email y contraseña son requeridos'}), 400
        
        if not COGNITO_CLIENT_ID:
            return jsonify({'error': 'Configuración de autenticación no disponible'}), 500
        
        # Preparar parámetros de autenticación
        auth_parameters = {
            'USERNAME': email,
            'PASSWORD': password
        }
        
        # Agregar SECRET_HASH si el Client tiene un secret
        secret_hash = calculate_secret_hash(email)
        if secret_hash:
            auth_parameters['SECRET_HASH'] = secret_hash
        
        # Autenticar usuario
        response = cognito_client.initiate_auth(
            ClientId=COGNITO_CLIENT_ID,
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters=auth_parameters
        )
        
        # Guardar información del usuario en la sesión
        session['user_email'] = email
        session['access_token'] = response['AuthenticationResult']['AccessToken']
        session['id_token'] = response['AuthenticationResult']['IdToken']
        session['refresh_token'] = response['AuthenticationResult']['RefreshToken']
        
        return jsonify({
            'success': True,
            'message': 'Inicio de sesión exitoso'
        })
    
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        
        if error_code == 'UserNotFoundException':
            return jsonify({'error': 'Usuario no encontrado'}), 404
        elif error_code == 'NotAuthorizedException':
            return jsonify({'error': 'Contraseña incorrecta o usuario no verificado'}), 401
        elif error_code == 'UserNotConfirmedException':
            return jsonify({'error': 'Usuario no verificado. Por favor verifica tu correo electrónico.'}), 403
        else:
            return jsonify({'error': f'Error al iniciar sesión: {error_message}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Error inesperado: {str(e)}'}), 500

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({
        'success': True,
        'message': 'Sesión cerrada exitosamente'
    })

@app.route('/api/auth_status', methods=['GET'])
def api_auth_status():
    if 'user_email' in session:
        return jsonify({
            'authenticated': True,
            'email': session.get('user_email')
        })
    return jsonify({
        'authenticated': False
    })

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'audio' not in request.files:
        return jsonify({'error': 'No se encontró el archivo'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(filepath)
        
        # Guardar el nombre original para el archivo .lab
        original_filename = filename  # Ya sanitizado por secure_filename
        
        # Ejecutar inference_frames.py con el nuevo modelo DNN
        try:
            project_root = Path(__file__).parent.parent
            inference_script = project_root / 'inference_frames.py'
            #inference_script = project_root / 'inference_baseline_mlp.py'
            
            # Ejecutar el script de inferencia con el nuevo modelo DNN
            # Usar --smooth para segmentación por beats (reduce oscilaciones)
            result = subprocess.run(
                [sys.executable, str(inference_script), str(filepath), "--smooth", "--beat-sync"],
                #[sys.executable, str(inference_script), str(filepath)],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=300  # 5 minutos timeout
            )
            
            if result.returncode != 0:
                return jsonify({
                    'error': f'Error al procesar el audio: {result.stderr}'
                }), 500
            
            # Tomar la ruta exacta desde la salida del script
            lab_file = None
            m = re.search(r"✅ Archivo \.lab guardado en: (.+?\.lab)", result.stdout)
            if m:
                # Resolver contra el root del proyecto si es ruta relativa
                extracted = Path(m.group(1).strip())
                lab_file = extracted if extracted.is_absolute() else (project_root / extracted)
            else:
                # Fallback: buscar el último archivo que matchee el patrón con timestamp
                outputs_dir = (project_root / 'outputs')
                # Aceptar con o sin sufijo (timestamp). Ej: *_predicted.lab o *_predicted_*.lab
                pattern = f"{filepath.stem}_predicted*.lab"
                candidates = sorted(outputs_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
                if candidates:
                    lab_file = candidates[0]
            
            if lab_file is None or not lab_file.exists():
                outputs_dir = project_root / 'outputs'
                existing_files = list(outputs_dir.glob('*.lab')) if outputs_dir.exists() else []
                return jsonify({
                    'error': f'No se generó el archivo de acordes. Existentes: {[f.name for f in existing_files]}. Salida: {result.stdout}. Error: {result.stderr}'
                }), 500
            
            # Leer los acordes del archivo .lab
            chords = []
            with open(lab_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # inference_frames.py usa espacios, pero también probar tabs por compatibilidad
                        if '\t' in line:
                            parts = line.split('\t')
                        else:
                            parts = line.split()
                        
                        if len(parts) >= 3:
                            chords.append({
                                'start': float(parts[0]),
                                'end': float(parts[1]),
                                'chord': parts[2]
                            })
            
            return jsonify({
                'success': True,
                'filename': filename,
                'chords': chords
            })
        
        except subprocess.TimeoutExpired:
            return jsonify({'error': 'El procesamiento tardó demasiado tiempo'}), 500
        except Exception as e:
            return jsonify({'error': f'Error inesperado: {str(e)}'}), 500
    
    return jsonify({'error': 'Tipo de archivo no permitido'}), 400

@app.route('/audio/<filename>')
def serve_audio(filename):
    filepath = app.config['UPLOAD_FOLDER'] / secure_filename(filename)
    if filepath.exists():
        return send_file(filepath)
    return jsonify({'error': 'Archivo no encontrado'}), 404

if __name__ == '__main__':
    # En producción, usar Gunicorn en lugar de app.run()
    # Para desarrollo local, puedes usar: python app.py
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)




