# Configuración de Gunicorn para producción

# Dirección y puerto donde escuchará Gunicorn
# 0.0.0.0 permite conexiones desde cualquier IP (necesario para acceso externo)
bind = "0.0.0.0:5000"

# Número de workers (procesos)
# Regla general: (2 x CPU cores) + 1
workers = 2

# Timeout para requests largos (procesamiento de audio puede tardar)
timeout = 300

# Clase de worker
worker_class = "sync"

# Logging
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = "info"

# Preload app para mejor rendimiento
preload_app = True

# Máximo de requests por worker antes de reciclar
max_requests = 1000
max_requests_jitter = 50

