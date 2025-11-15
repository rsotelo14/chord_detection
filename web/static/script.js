let currentChords = [];
let currentFilename = '';

// Elementos del DOM
const uploadArea = document.getElementById('uploadArea');
const audioInput = document.getElementById('audioInput');
const uploadSection = document.getElementById('uploadSection');
const loadingSection = document.getElementById('loading');
const playerSection = document.getElementById('playerSection');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const audioPlayer = document.getElementById('audioPlayer');
const currentChord = document.getElementById('chordValue');
const timeline = document.getElementById('timeline');
const resetBtn = document.getElementById('resetBtn');

// Event listeners para drag & drop
uploadArea.addEventListener('click', () => audioInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileUpload(files[0]);
    }
});

audioInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

resetBtn.addEventListener('click', () => {
    resetPlayer();
});

// Función para manejar la subida del archivo
async function handleFileUpload(file) {
    // Validar tipo de archivo
    const allowedTypes = ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/mp4', 'audio/flac', 'audio/x-m4a'];
    if (!allowedTypes.includes(file.type) && !file.name.match(/\.(mp3|wav|ogg|m4a|flac)$/i)) {
        showError('Tipo de archivo no soportado. Por favor usa MP3, WAV, OGG, M4A o FLAC.');
        return;
    }

    // Validar tamaño (50MB)
    if (file.size > 50 * 1024 * 1024) {
        showError('El archivo es demasiado grande. El tamaño máximo es 50MB.');
        return;
    }

    hideError();
    showLoading();

    const formData = new FormData();
    formData.append('audio', file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Error al procesar el archivo');
        }

        currentChords = data.chords;
        currentFilename = data.filename;

        // Configurar el reproductor de audio
        audioPlayer.src = `/audio/${currentFilename}`;
        
        // Renderizar acordes en la línea de tiempo
        renderTimeline();
        
        // Mostrar el reproductor
        showPlayer();

    } catch (error) {
        showError(error.message);
        resetPlayer();
    }
}

// Función para renderizar la línea de tiempo de acordes
function renderTimeline() {
    timeline.innerHTML = '';
    
    currentChords.forEach((chord, index) => {
        const chordItem = document.createElement('div');
        chordItem.className = 'chord-item';
        chordItem.dataset.index = index;
        
        const startTime = formatTime(chord.start);
        const endTime = formatTime(chord.end);
        
        chordItem.innerHTML = `
            <span class="chord-time">${startTime} - ${endTime}</span>
            <span class="chord-name">${formatChord(chord.chord)}</span>
        `;
        
        // Hacer clic en un acorde para saltar a ese momento
        chordItem.addEventListener('click', () => {
            audioPlayer.currentTime = chord.start;
            if (audioPlayer.paused) {
                audioPlayer.play();
            }
        });
        
        timeline.appendChild(chordItem);
    });
}

// Actualizar acorde actual mientras se reproduce
audioPlayer.addEventListener('timeupdate', () => {
    const currentTime = audioPlayer.currentTime;
    
    // Encontrar el acorde actual
    const activeChord = currentChords.find(chord => 
        currentTime >= chord.start && currentTime < chord.end
    );
    
    if (activeChord) {
        currentChord.textContent = formatChord(activeChord.chord);
        
        // Resaltar en la línea de tiempo
        const allItems = timeline.querySelectorAll('.chord-item');
        allItems.forEach(item => item.classList.remove('active'));
        
        const activeIndex = currentChords.indexOf(activeChord);
        const activeItem = timeline.querySelector(`[data-index="${activeIndex}"]`);
        if (activeItem) {
            activeItem.classList.add('active');
            // Scroll automático para mantener visible el acorde activo
            activeItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }
});

// Formatear tiempo en formato MM:SS
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// Formatear nombre del acorde
function formatChord(chord) {
    if (chord === 'N') {
        return 'Sin acorde';
    }
    return chord;
}

// Funciones de UI
function showLoading() {
    uploadSection.style.display = 'none';
    playerSection.style.display = 'none';
    loadingSection.style.display = 'block';
}

function showPlayer() {
    uploadSection.style.display = 'none';
    loadingSection.style.display = 'none';
    playerSection.style.display = 'block';
}

function showError(message) {
    errorText.textContent = message;
    errorMessage.style.display = 'flex';
    setTimeout(() => {
        hideError();
    }, 5000);
}

function hideError() {
    errorMessage.style.display = 'none';
}

function resetPlayer() {
    uploadSection.style.display = 'block';
    loadingSection.style.display = 'none';
    playerSection.style.display = 'none';
    audioPlayer.pause();
    audioPlayer.src = '';
    currentChords = [];
    currentFilename = '';
    audioInput.value = '';
    currentChord.textContent = '-';
    timeline.innerHTML = '';
}
































