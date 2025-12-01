# Guía de Configuración de Variables de Entorno

Esta guía te ayudará a configurar las variables de entorno necesarias para la autenticación con AWS Cognito.

## Archivo de Configuración

**Archivo principal:** `web/.env` (debes crearlo tú mismo)

1. Copia el archivo de ejemplo:
   ```bash
   cd web
   cp env.example .env
   ```

2. Edita el archivo `.env` con tus credenciales reales.

## Variables Requeridas

### 1. Credenciales de AWS

Estas credenciales son necesarias para que la aplicación pueda acceder a AWS Cognito.

#### `AWS_ACCESS_KEY_ID`
**¿Qué es?** Tu Access Key ID de AWS  
**¿Dónde encontrarlo?**
1. Ve a [AWS Console](https://console.aws.amazon.com/)
2. Haz clic en tu nombre de usuario (arriba a la derecha)
3. Selecciona "Security credentials"
4. En la sección "Access keys", haz clic en "Create access key"
5. Copia el **Access key ID**

#### `AWS_SECRET_ACCESS_KEY`
**¿Qué es?** Tu Secret Access Key de AWS (solo se muestra una vez)  
**¿Dónde encontrarlo?**
1. En la misma página donde creaste el Access Key
2. Haz clic en "Show" para revelar el **Secret access key**
3. **IMPORTANTE:** Guárdalo de forma segura, no se puede recuperar después

#### `AWS_DEFAULT_REGION`
**¿Qué es?** La región donde está tu User Pool de Cognito  
**Valor:** `us-east-1` (ya configurado en el ejemplo)

### 2. Configuración de Cognito

#### `COGNITO_CLIENT_ID` (Opcional)
**¿Qué es?** El Client ID de tu aplicación en Cognito  
**¿Dónde encontrarlo?**
1. Ve a [AWS Cognito Console](https://console.aws.amazon.com/cognito/)
2. Selecciona tu User Pool: `us-east-1_o7y7iPcVz`
3. En el menú lateral, ve a "App integration" → "App clients"
4. Verás una lista de aplicaciones cliente
5. Copia el **Client ID** de la aplicación que quieras usar

**Nota:** Si no configuras esta variable, la aplicación intentará obtenerla automáticamente. Si tienes problemas, configúrala manualmente.

#### `COGNITO_CLIENT_SECRET` (Requerido si el App Client tiene secret habilitado)
**¿Qué es?** El Client Secret de tu aplicación en Cognito  
**¿Dónde encontrarlo?**
1. Ve a [AWS Cognito Console](https://console.aws.amazon.com/cognito/)
2. Selecciona tu User Pool: `us-east-1_o7y7iPcVz`
3. En el menú lateral, ve a "App integration" → "App clients"
4. Haz clic en el nombre de tu aplicación cliente
5. En la sección "App client information", busca "Client secret"
6. Si hay un secret configurado, haz clic en "Show" para revelarlo
7. **IMPORTANTE:** Si no ves un Client Secret, significa que tu App Client no tiene secret habilitado y no necesitas esta variable

**Nota:** Si tu App Client tiene un Client Secret habilitado, DEBES configurar esta variable o recibirás el error: "Client is configured with secret but SECRET_HASH was not received"

#### User Pool ID
**¿Qué es?** El ID de tu User Pool de Cognito  
**Valor:** `us-east-1_o7y7iPcVz` (ya está configurado en `app.py`, no necesitas cambiarlo)

### 3. Configuración de Flask

#### `SECRET_KEY`
**¿Qué es?** Clave secreta para las sesiones de Flask  
**¿Dónde encontrarlo?** Genera una clave aleatoria segura:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

**Importante:** En producción, usa una clave segura y única. No uses la clave por defecto.

## Ejemplo de Archivo .env

```env
# Credenciales de AWS
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION=us-east-1

# Cognito Client ID (opcional)
COGNITO_CLIENT_ID=1234567890abcdefghijklmn

# Cognito Client Secret (requerido si el App Client tiene secret habilitado)
COGNITO_CLIENT_SECRET=tu-client-secret-aqui-si-es-necesario

# Secret Key para Flask
SECRET_KEY=tu-clave-secreta-generada-aqui
```

## Verificación de Configuración

Después de configurar el archivo `.env`, puedes verificar que todo esté correcto:

1. Inicia el servidor:
   ```bash
   cd web
   python app.py
   ```

2. Busca en la salida del servidor:
   - Si ves: `"Client ID obtenido automáticamente: ..."` → ✅ Todo está bien
   - Si ves: `"ERROR: COGNITO_CLIENT_ID no está configurado"` → ⚠️ Necesitas configurar el Client ID

## Alternativa: Variables de Entorno del Sistema

Si prefieres no usar un archivo `.env`, puedes configurar las variables directamente en tu sistema:

### Linux/macOS:
```bash
export AWS_ACCESS_KEY_ID=tu_access_key
export AWS_SECRET_ACCESS_KEY=tu_secret_key
export AWS_DEFAULT_REGION=us-east-1
export COGNITO_CLIENT_ID=tu_client_id
export SECRET_KEY=tu_secret_key
```

### Windows (PowerShell):
```powershell
$env:AWS_ACCESS_KEY_ID="tu_access_key"
$env:AWS_SECRET_ACCESS_KEY="tu_secret_key"
$env:AWS_DEFAULT_REGION="us-east-1"
$env:COGNITO_CLIENT_ID="tu_client_id"
$env:SECRET_KEY="tu_secret_key"
```

### Windows (CMD):
```cmd
set AWS_ACCESS_KEY_ID=tu_access_key
set AWS_SECRET_ACCESS_KEY=tu_secret_key
set AWS_DEFAULT_REGION=us-east-1
set COGNITO_CLIENT_ID=tu_client_id
set SECRET_KEY=tu_secret_key
```

## Seguridad

⚠️ **IMPORTANTE:**
- **NUNCA** subas el archivo `.env` a un repositorio Git (ya está en `.gitignore`)
- **NUNCA** compartas tus credenciales de AWS
- Usa credenciales con permisos mínimos necesarios (solo Cognito)
- En producción, usa AWS IAM Roles en lugar de Access Keys cuando sea posible

## Solución de Problemas

### Error: "No se pudo obtener el Client ID automáticamente"
**Solución:** Configura manualmente `COGNITO_CLIENT_ID` en el archivo `.env`

### Error: "Unable to locate credentials"
**Solución:** Verifica que `AWS_ACCESS_KEY_ID` y `AWS_SECRET_ACCESS_KEY` estén configurados correctamente

### Error: "User Pool does not exist"
**Solución:** Verifica que el User Pool ID `us-east-1_o7y7iPcVz` exista en tu cuenta de AWS

### Error: "Client is configured with secret but SECRET_HASH was not received"
**Solución:** Tu App Client tiene un Client Secret habilitado. Debes configurar `COGNITO_CLIENT_SECRET` en tu archivo `.env` con el valor del Client Secret de tu aplicación en Cognito.

### Error: "SignUp is not permitted for this user pool"
**Solución:** El registro está deshabilitado en tu User Pool o el App Client no tiene permisos. Sigue estos pasos:

1. **Habilitar auto-sign-up en el User Pool:**
   - Ve a [AWS Cognito Console](https://console.aws.amazon.com/cognito/)
   - Selecciona tu User Pool: `us-east-1_o7y7iPcVz`
   - Ve a "Sign-up experience" en el menú lateral
   - Asegúrate de que "Enable self-registration" esté habilitado
   - Si está deshabilitado, haz clic en "Edit" y habilítalo
   - Guarda los cambios

2. **Verificar permisos del App Client:**
   - En el mismo User Pool, ve a "App integration" → "App clients"
   - Haz clic en tu aplicación cliente (`3vevdg11rr2v2avvbhf0umjofr`)
   - Desplázate hasta la sección "Hosted UI settings" o "Authentication flows"
   - Asegúrate de que "ALLOW_USER_SRP_AUTH" o "ALLOW_USER_PASSWORD_AUTH" esté habilitado
   - Si hay una sección "OAuth 2.0 grant types", asegúrate de que "Authorization code grant" esté habilitado

3. **Verificar políticas de seguridad:**
   - En el User Pool, ve a "Sign-up experience" → "Security"
   - Verifica que no haya políticas que bloqueen el registro
   - Si hay un "Account recovery" configurado, asegúrate de que esté habilitado

**Nota:** Si no tienes acceso para modificar estas configuraciones, contacta al administrador de AWS que configuró el User Pool.

