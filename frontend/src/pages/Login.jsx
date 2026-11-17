import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import api, { saveSession } from "../api/client";
import AuthLayout, { AuthFields, AuthError } from "../components/auth/AuthLayout";
import GoogleAuthButton from "../components/auth/GoogleAuthButton";
import validateCredentials from "../components/auth/validateCredentials";

const fields = [
  { name: "email", type: "text", label: "Email address", placeholder: "you@example.com" },
  { name: "password", type: "password", label: "Password", placeholder: "min 6 characters" },
];

export default function Login() {
  const navigate = useNavigate();
  const [form, setForm] = useState({ email: "", password: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handle = (e) => setForm({ ...form, [e.target.name]: e.target.value });

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    const err = validateCredentials(form);
    if (err) {
      setError(err);
      return;
    }
    setLoading(true);
    try {
      const res = await api.post("/auth/login", form);
      saveSession(res.data);
      navigate("/upload");
    } catch (err) {
      setError(err.response?.data?.detail || "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout title="Welcome back." subtitle="Log in to your account.">
      <form onSubmit={submit} className="space-y-6">
        <AuthFields fields={fields} form={form} onChange={handle} />
        <AuthError message={error} />
        <button
          type="submit"
          disabled={loading}
          className="mt-2 w-full rounded-xl bg-[var(--text)] py-4 text-base font-bold text-[var(--bg)] transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Authenticating..." : "Login"}
        </button>
      </form>

      <GoogleAuthButton failMessage="Google Login failed" setError={setError} setLoading={setLoading} />

      <p className="mt-8 text-center text-sm text-[var(--muted)]">
        Don't have an account?{" "}
        <Link to="/signup" className="font-bold text-[var(--text)] hover:underline transition-all">
          Sign up
        </Link>
      </p>
    </AuthLayout>
  );
}
