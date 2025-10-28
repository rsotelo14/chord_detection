# rename_audio_files.py
"""
Renombra automáticamente archivos de audio para que matcheen con sus .lab correspondientes.
"""

import re
from pathlib import Path
from difflib import SequenceMatcher

AUDIO_ROOT = Path("The Beatles Annotations/audio/The Beatles")
LAB_ROOT = Path("The Beatles Annotations/chordlab/The Beatles")


def normalize_for_comparison(text):
    """Normaliza texto para comparación (lowercase, sin espacios extra, sin puntuación)."""
    text = text.lower()
    # Quitar "(Remastered XXXX)" y "- The Beatles"
    text = re.sub(r'\(remastered \d+\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'- the beatles', '', text, flags=re.IGNORECASE)
    # Quitar números de track al inicio (##_-_)
    text = re.sub(r'^\d+_-_', '', text)
    text = re.sub(r'^cd\d+_-_\d+_-_', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s]', '', text)  # Quitar puntuación
    text = re.sub(r'\s+', ' ', text).strip()  # Normalizar espacios
    return text


def find_best_lab_match(audio_name, lab_files):
    """
    Encuentra el archivo .lab que mejor matchea con el nombre del audio.
    
    Args:
        audio_name: nombre del archivo de audio (sin extensión)
        lab_files: lista de paths a archivos .lab
    
    Returns:
        (best_lab_path, similarity_score)
    """
    audio_normalized = normalize_for_comparison(audio_name)
    
    best_match = None
    best_score = 0
    
    for lab_path in lab_files:
        lab_name = lab_path.stem  # Nombre sin extensión
        lab_normalized = normalize_for_comparison(lab_name)
        
        # Calcular similitud
        score = SequenceMatcher(None, audio_normalized, lab_normalized).ratio()
        
        if score > best_score:
            best_score = score
            best_match = lab_path
    
    return best_match, best_score


def rename_audio_files(dry_run=True):
    """
    Renombra archivos de audio para matchear con sus .lab correspondientes.
    
    Args:
        dry_run: si True, solo muestra lo que haría sin renombrar
    """
    print("=" * 70)
    print("🔄 Renombrado automático de archivos de audio")
    print("=" * 70)
    print(f"Modo: {'DRY RUN (solo vista previa)' if dry_run else 'RENOMBRAR ARCHIVOS'}")
    print()
    
    renamed_count = 0
    skipped_count = 0
    
    # Recorrer cada álbum
    for album_dir in sorted(AUDIO_ROOT.iterdir()):
        if not album_dir.is_dir():
            continue
        
        album_name = album_dir.name
        lab_album_dir = LAB_ROOT / album_name
        
        if not lab_album_dir.exists():
            print(f"⚠️  No hay folder de anotaciones para: {album_name}")
            continue
        
        # Obtener todos los .lab de este álbum
        lab_files = list(lab_album_dir.glob("*.lab"))
        
        if not lab_files:
            continue
        
        print(f"\n📁 {album_name}")
        print(f"   Archivos .lab disponibles: {len(lab_files)}")
        
        # Buscar archivos de audio que necesiten renombrado
        for audio_file in sorted(album_dir.glob("*.mp3")):
            audio_name = audio_file.stem
            
            # Si ya tiene formato correcto (##_-_*), skip
            if re.match(r'^\d+_-_', audio_name):
                continue
            
            # Buscar mejor match con .lab
            best_lab, score = find_best_lab_match(audio_name, lab_files)
            
            if best_lab is None or score < 0.7:  # Umbral más alto para mayor confianza
                if score > 0:
                    print(f"   ⚠️  Score bajo: {audio_name[:50]}")
                    print(f"       Mejor match: {best_lab.stem} (score={score:.2f})")
                else:
                    print(f"   ⚠️  No match: {audio_name}")
                skipped_count += 1
                continue
            
            # Nuevo nombre basado en el .lab
            new_name = best_lab.stem + audio_file.suffix
            new_path = audio_file.parent / new_name
            
            # Verificar si ya existe
            if new_path.exists() and new_path != audio_file:
                print(f"   ⚠️  Ya existe: {new_name}")
                skipped_count += 1
                continue
            
            # Mostrar cambio
            print(f"   ✓ {audio_file.name}")
            print(f"      → {new_name}  (score={score:.2f})")
            
            # Renombrar (si no es dry run)
            if not dry_run:
                audio_file.rename(new_path)
            
            renamed_count += 1
    
    # Resumen
    print("\n" + "=" * 70)
    print("📊 Resumen")
    print("=" * 70)
    print(f"Archivos renombrados: {renamed_count}")
    print(f"Archivos omitidos:    {skipped_count}")
    
    if dry_run and renamed_count > 0:
        print("\n💡 Para aplicar los cambios, ejecuta:")
        print("   python rename_audio_files.py --apply")
    
    return renamed_count, skipped_count


if __name__ == "__main__":
    import sys
    
    # Primero hacer dry run
    if "--apply" in sys.argv:
        print("\n⚠️  APLICANDO CAMBIOS...")
        if "--yes" not in sys.argv and "-y" not in sys.argv:
            input("Presiona Enter para continuar o Ctrl+C para cancelar...")
        rename_audio_files(dry_run=False)
    else:
        rename_audio_files(dry_run=True)

