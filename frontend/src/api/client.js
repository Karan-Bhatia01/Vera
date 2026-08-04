import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:5000",
  // Without a timeout a stalled backend leaves the UI spinning forever
  // (e.g. Data Info stuck on "Loading…"). Fail fast so callers can show
  // an error and let the user retry. Long jobs are polled, not held open,
  // so 20s is plenty for any single request.
  timeout: 20000,
});

// Attach the token to every outgoing request, if one exists.
api.interceptors.request.use((config) => { // Intercetors are middlemen that run before every request or after every response 
  const token = localStorage.getItem("token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// If the backend ever says the token is invalid/expired, clear it and
// send the user back to login instead of leaving them stuck on a broken page.
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem("token");
      localStorage.removeItem("email");
      if (window.location.pathname !== "/login") {
        window.location.href = "/login";
      }
    }
    return Promise.reject(error);
  }
);

export default api;