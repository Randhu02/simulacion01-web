// Función de login
function login() {
    const user = document.getElementById('username').value.trim();
    const pass = document.getElementById('password').value.trim();
    const errorMsg = document.getElementById('error-message');

    if (user === "operaciones" && pass === "tisur") {
        window.location.href = "home.html";
    } else {
        errorMsg.textContent = "Usuario o contraseña incorrectos";
    }
}

// Mostrar / ocultar contraseña
document.getElementById('showPassword').addEventListener('change', function() {
    const passwordField = document.getElementById('password');
    if (this.checked) {
        passwordField.type = "text";
    } else {
        passwordField.type = "password";
    }
});

