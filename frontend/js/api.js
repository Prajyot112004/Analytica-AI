const API_BASE_URL = "http://127.0.0.1:8000/api/v1";

function getAuthHeaders() {
    const token = localStorage.getItem("analytica_token");
    return token ? { "Authorization": `Bearer ${token}` } : {};
}

async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const headers = {
        ...getAuthHeaders(),
        ...(options.headers || {})
    };

    if (options.body && !(options.body instanceof FormData)) {
        headers["Content-Type"] = "application/json";
        options.body = JSON.stringify(options.body);
    }

    const response = await fetch(url, { ...options, headers });
    
    if (response.status === 401 && !endpoint.includes("/auth/")) {
        localStorage.removeItem("analytica_token");
        window.location.href = "login.html";
        throw new Error("Unauthorized");
    }

    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.detail || "API Request Failed");
    }
    return data;
}
