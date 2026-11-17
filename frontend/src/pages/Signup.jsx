import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import api, { saveSession } from "../api/client";
import AuthLayout, { AuthFields, AuthError } from "../components/auth/AuthLayout";
import GoogleAuthButton from "../components/auth/GoogleAuthButton";
import validateCredentials from "../components/auth/validateCredentials";

const fields = [
  { name: "email", type: "text", label: "Email address", placeholder: "you@example.com" },
  { name: "password", type: "password", label: "Password", placeholder: "min 6 characters" },
  { name: "confirm_password", type: "password", label: "Confirm password", placeholder: "re-enter password" },
];

export default function Signup() {
  const navigate = useNavigate();
  const [form, setForm] = useState({ email: "", password: "", confirm_password: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handle = (e) => setForm({ ...form, [e.target.name]: e.target.value });

  const validate = () => {
    const err = validateCredentials(form);
    if (err) return err;
    if (form.password !== form.confirm_password) return "Passwords do not match";
    if (!form.consent) return "You must agree to the Terms of Service and Privacy Policy";
    return null;
  };

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    const err = validate();
    if (err) {
      setError(err);
      return;
    }
    setLoading(true);
    try {
      const res = await api.post("/auth/signup", form);
      saveSession(res.data);
      navigate("/upload");
    } catch (err) {
      setError(err.response?.data?.detail || "Signup failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout title="Create account." subtitle="Join Vera and start analyzing your data.">
      <form onSubmit={submit} className="space-y-6">
        <AuthFields fields={fields} form={form} onChange={handle} />
        <AuthError message={error} />

        <div className="flex items-start gap-3 mt-4">
          <input
            type="checkbox"
            id="consent"
            name="consent"
            checked={form.consent || false}
            onChange={(e) => setForm({ ...form, consent: e.target.checked })}
            className="mt-1 h-4 w-4 shrink-0 rounded border-[var(--text)]/[0.2] bg-[var(--text)]/[0.05] text-[var(--text)] focus:ring-[var(--text)]"
          />
          <label htmlFor="consent" className="text-sm text-[var(--muted)] leading-relaxed">
            I agree to the{" "}
            <Link to="/terms-of-service" className="text-[var(--text)] underline hover:opacity-80">Terms of Service</Link>
            {" "}and{" "}
            <Link to="/privacy-policy" className="text-[var(--text)] underline hover:opacity-80">Privacy Policy</Link>.
          </label>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="mt-2 w-full rounded-xl bg-[var(--text)] py-4 text-base font-bold text-[var(--bg)] transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Creating Account..." : "Create Account"}
        </button>
      </form>

      <GoogleAuthButton failMessage="Google Signup failed" setError={setError} setLoading={setLoading} />

      <p className="mt-8 text-center text-sm text-[var(--muted)]">
        Already have an account?{" "}
        <Link to="/login" className="font-bold text-[var(--text)] hover:underline transition-all">
          Log in
        </Link>
      </p>
    </AuthLayout>
  );
}
