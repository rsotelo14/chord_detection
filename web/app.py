import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import subprocess
import re

# Agregar el directorio raíz al path para importar inference
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
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
    app.run(debug=True, port=5000)




