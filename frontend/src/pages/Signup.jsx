import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import api, { saveSession } from "../api/client";
import AuthLayout, { AuthFields, AuthError, AuthSubmit } from "../components/auth/AuthLayout";
import GoogleAuthButton from "../components/auth/GoogleAuthButton";
import validateCredentials from "../components/auth/validateCredentials";

const fields = [
  { name: "email", type: "email", label: "Email address", placeholder: "you@example.com", autoComplete: "email" },
  { name: "password", type: "password", label: "Password", placeholder: "At least 6 characters", autoComplete: "new-password" },
  { name: "confirm_password", type: "password", label: "Confirm password", placeholder: "Type it again", autoComplete: "new-password" },
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
      setError(err.response?.data?.detail || "Couldn't reach Vera. Check your connection and try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout title="Create your account" subtitle="Free, and you can upload your first CSV right after.">
      <form onSubmit={submit} noValidate className="space-y-5">
        <AuthFields fields={fields} form={form} onChange={handle} />
        <AuthError message={error} />

        <div className="flex items-start gap-3">
          <input
            type="checkbox"
            id="consent"
            name="consent"
            checked={form.consent || false}
            onChange={(e) => setForm({ ...form, consent: e.target.checked })}
            className="mt-1 h-4 w-4 shrink-0 accent-[var(--sky-ink)]"
          />
          <label htmlFor="consent" className="text-[15px] text-[var(--muted)] leading-relaxed">
            I agree to the{" "}
            <Link to="/terms-of-service" className="text-[var(--text)] underline underline-offset-2">Terms of Service</Link>
            {" "}and{" "}
            <Link to="/privacy-policy" className="text-[var(--text)] underline underline-offset-2">Privacy Policy</Link>.
          </label>
        </div>

        <AuthSubmit loading={loading} loadingText="Creating account…">Create account</AuthSubmit>
      </form>

      <GoogleAuthButton failMessage="Google sign-up didn't finish. Try again." setError={setError} setLoading={setLoading} />

      <p className="mt-8 text-center text-sm text-[var(--muted)]">
        Already have an account?{" "}
        <Link to="/login" className="font-bold text-[var(--text)] underline underline-offset-4">
          Log in
        </Link>
      </p>
    </AuthLayout>
  );
}
