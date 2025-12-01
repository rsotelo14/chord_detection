# Guía de Solución de Problemas - AWS Cognito

Esta guía te ayudará a resolver los errores más comunes relacionados con AWS Cognito.

## Error: "SignUp is not permitted for this user pool"

Este error indica que el registro está deshabilitado en tu User Pool o que el App Client no tiene los permisos necesarios.

### Solución Paso a Paso

#### Paso 1: Habilitar Auto-Sign-Up en el User Pool

1. Ve a [AWS Cognito Console](https://console.aws.amazon.com/cognito/)
2. Selecciona tu User Pool: `us-east-1_o7y7iPcVz`
3. En el menú lateral izquierdo, haz clic en **"Sign-up experience"**
4. Verifica la sección **"Self-service sign-up"**
5. Si está deshabilitado:
   - Haz clic en **"Edit"**
   - Habilita **"Enable self-registration"**
   - Guarda los cambios

#### Paso 2: Verificar Permisos del App Client

1. En el mismo User Pool, ve a **"App integration"** → **"App clients"**
2. Haz clic en el nombre de tu aplicación cliente (`3vevdg11rr2v2avvbhf0umjofr`)
3. Desplázate hasta la sección **"Hosted UI settings"** o busca **"Authentication flows"**
4. Verifica que estén habilitados:
   - **ALLOW_USER_SRP_AUTH** (recomendado)
   - O **ALLOW_USER_PASSWORD_AUTH** (si usas USER_PASSWORD_AUTH)
5. Si necesitas cambiarlos:
   - Haz clic en **"Edit"**
   - Habilita los flujos de autenticación necesarios
   - Guarda los cambios

#### Paso 3: Verificar Configuración de OAuth (si aplica)

Si tu App Client tiene OAuth habilitado:

1. En la misma página del App Client, busca **"OAuth 2.0 grant types"**
2. Asegúrate de que esté habilitado:
   - **Authorization code grant** (recomendado)
   - O **Implicit grant** (menos seguro)
3. Verifica que los **"Allowed callback URLs"** estén configurados correctamente

#### Paso 4: Verificar Políticas de Seguridad

1. En el User Pool, ve a **"Sign-up experience"** → **"Security"**
2. Verifica que no haya políticas que bloqueen el registro
3. Revisa la configuración de **"Password policy"** para asegurarte de que sea compatible con tus requisitos

### Verificación con AWS CLI (Opcional)

Si tienes AWS CLI configurado, puedes verificar la configuración:

```bash
# Verificar configuración del User Pool
aws cognito-idp describe-user-pool \
  --user-pool-id us-east-1_o7y7iPcVz \
  --region us-east-1

# Verificar configuración del App Client
aws cognito-idp describe-user-pool-client \
  --user-pool-id us-east-1_o7y7iPcVz \
  --client-id 3vevdg11rr2v2avvbhf0umjofr \
  --region us-east-1
```

### Si No Tienes Permisos

Si no tienes acceso para modificar estas configuraciones:

1. **Contacta al administrador de AWS** que configuró el User Pool
2. Comparte este documento con ellos
3. Solicita que habiliten:
   - Self-service sign-up en el User Pool
   - Permisos de autenticación en el App Client

## Error: "Client is configured with secret but SECRET_HASH was not received"

**Solución:** Configura `COGNITO_CLIENT_SECRET` en tu archivo `.env`. Ver [CONFIGURACION.md](CONFIGURACION.md) para más detalles.

## Error: "User Pool does not exist"

**Solución:** Verifica que el User Pool ID `us-east-1_o7y7iPcVz` exista en tu cuenta de AWS y región correcta.

## Error: "Unable to locate credentials"

**Solución:** Verifica que `AWS_ACCESS_KEY_ID` y `AWS_SECRET_ACCESS_KEY` estén configurados correctamente en tu archivo `.env`.

## Error: "UserNotFoundException"

**Solución:** El usuario no existe en el User Pool. Asegúrate de que el correo electrónico sea correcto y que el usuario se haya registrado previamente.

## Error: "NotAuthorizedException"

**Solución:** 
- Verifica que la contraseña sea correcta
- Asegúrate de que el usuario haya verificado su correo electrónico
- Verifica que el usuario no esté bloqueado en el User Pool

## Recursos Adicionales

- [Documentación oficial de AWS Cognito](https://docs.aws.amazon.com/cognito/)
- [Guía de configuración de User Pools](https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools.html)
- [Troubleshooting de Cognito](https://docs.aws.amazon.com/cognito/latest/developerguide/troubleshooting.html)

