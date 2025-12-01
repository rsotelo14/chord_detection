// Funciones de autenticación

function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');
    if (errorMessage && errorText) {
        errorText.textContent = message;
        errorMessage.style.display = 'flex';
        setTimeout(() => {
            errorMessage.style.display = 'none';
        }, 5000);
    }
}

function hideError() {
    const errorMessage = document.getElementById('errorMessage');
    if (errorMessage) {
        errorMessage.style.display = 'none';
    }
}

function showSuccess(message) {
    const successMessage = document.getElementById('successMessage');
    const successText = document.getElementById('successText');
    if (successMessage && successText) {
        successText.textContent = message;
        successMessage.style.display = 'flex';
    }
}

// Manejo del formulario de registro
const registerForm = document.getElementById('registerForm');
if (registerForm) {
    registerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        hideError();
        
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;
        const confirmPassword = document.getElementById('confirm_password').value;
        
        if (password !== confirmPassword) {
            showError('Las contraseñas no coinciden');
            return;
        }
        
        try {
            const response = await fetch('/api/signup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    email,
                    password,
                    confirm_password: confirmPassword
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Redirigir a la página de verificación
                window.location.href = `/verify?email=${encodeURIComponent(email)}`;
            } else {
                showError(data.error || 'Error al registrar usuario');
            }
        } catch (error) {
            showError('Error de conexión. Por favor intenta nuevamente.');
        }
    });
}

// Manejo del formulario de verificación
const verifyForm = document.getElementById('verifyForm');
if (verifyForm) {
    verifyForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        hideError();
        
        const email = document.getElementById('email').value;
        const code = document.getElementById('code').value;
        
        try {
            const response = await fetch('/api/verify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    email,
                    code
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                showSuccess('Cuenta verificada exitosamente. Redirigiendo...');
                setTimeout(() => {
                    window.location.href = '/login';
                }, 2000);
            } else {
                showError(data.error || 'Error al verificar código');
            }
        } catch (error) {
            showError('Error de conexión. Por favor intenta nuevamente.');
        }
    });
    
    // Reenviar código
    const resendLink = document.getElementById('resendLink');
    if (resendLink) {
        resendLink.addEventListener('click', async (e) => {
            e.preventDefault();
            hideError();
            
            const email = document.getElementById('email').value;
            
            try {
                const response = await fetch('/api/resend_code', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ email })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    showSuccess('Código reenviado. Revisa tu correo electrónico.');
                } else {
                    showError(data.error || 'Error al reenviar código');
                }
            } catch (error) {
                showError('Error de conexión. Por favor intenta nuevamente.');
            }
        });
    }
}

// Manejo del formulario de login
const loginForm = document.getElementById('loginForm');
if (loginForm) {
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        hideError();
        
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;
        
        try {
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    email,
                    password
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Redirigir a la página principal
                window.location.href = '/';
            } else {
                showError(data.error || 'Error al iniciar sesión');
            }
        } catch (error) {
            showError('Error de conexión. Por favor intenta nuevamente.');
        }
    });
}

// Manejo del botón de logout
const logoutBtn = document.getElementById('logoutBtn');
if (logoutBtn) {
    logoutBtn.addEventListener('click', async () => {
        try {
            const response = await fetch('/api/logout', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (response.ok) {
                window.location.href = '/login';
            } else {
                showError(data.error || 'Error al cerrar sesión');
            }
        } catch (error) {
            showError('Error de conexión. Por favor intenta nuevamente.');
        }
    });
}

