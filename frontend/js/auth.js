function checkAuth() {
    const token = localStorage.getItem("analytica_token");
    if (!token) {
        if (!window.location.pathname.endsWith("login.html") && !window.location.pathname.endsWith("register.html")) {
            window.location.href = "login.html";
        }
    }
}

function logoutUser() {
    localStorage.removeItem("analytica_token");
    localStorage.removeItem("analytica_user");
    localStorage.removeItem("analytica_session");
    window.location.href = "login.html";
}

async function fetchUserProfile() {
    try {
        const user = await apiRequest("/auth/me");
        localStorage.setItem("analytica_user", JSON.stringify(user));
        return user;
    } catch (err) {
        logoutUser();
    }
}
