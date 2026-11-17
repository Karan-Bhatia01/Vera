import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import api, { saveSession } from "../api/client";
import AuthLayout, { AuthFields, AuthError, AuthSubmit } from "../components/auth/AuthLayout";
import GoogleAuthButton from "../components/auth/GoogleAuthButton";
import validateCredentials from "../components/auth/validateCredentials";

const fields = [
  { name: "email", type: "email", label: "Email address", placeholder: "you@example.com", autoComplete: "email" },
  { name: "password", type: "password", label: "Password", placeholder: "At least 6 characters", autoComplete: "current-password" },
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
      setError(err.response?.data?.detail || "Couldn't reach Vera. Check your connection and try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout title="Log in to Vera" subtitle="Pick up where you left off.">
      <form onSubmit={submit} noValidate className="space-y-5">
        <AuthFields fields={fields} form={form} onChange={handle} />
        <AuthError message={error} />
        <AuthSubmit loading={loading} loadingText="Logging in…">Log in</AuthSubmit>
      </form>

      <GoogleAuthButton failMessage="Google sign-in didn't finish. Try again." setError={setError} setLoading={setLoading} />

      <p className="mt-8 text-center text-sm text-[var(--muted)]">
        Don't have an account?{" "}
        <Link to="/signup" className="font-bold text-[var(--text)] underline underline-offset-4">
          Sign up
        </Link>
      </p>
    </AuthLayout>
  );
}
