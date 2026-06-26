import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import axios from "axios";
import Background from "../components/Background";

const fields = [
  { name: "email", type: "text", label: "EMAIL", placeholder: "you@example.com" },
  { name: "password", type: "password", label: "PASSWORD", placeholder: "min 6 characters" },
];

export default function Login() {
  const navigate = useNavigate();
  const [form, setForm] = useState({ email: "", password: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handle = (e) => setForm({ ...form, [e.target.name]: e.target.value });

  const validate = () => {
    if (!form.email) return "Email is required";
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email)) return "Invalid email format";
    if (!form.password) return "Password is required";
    if (form.password.length < 6) return "Password must be at least 6 characters";
    return null;
  };

  const submit = async () => {
    setError("");
    const err = validate();
    if (err) {
      setError(err);
      return;
    }
    setLoading(true);
    try {
      const res = await axios.post("http://localhost:5000/auth/login", form);
      localStorage.setItem("token", res.data.token);
      localStorage.setItem("email", res.data.email);
      navigate("/upload");
    } catch (err) {
      setError(err.response?.data?.detail || "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen flex items-center justify-center bg-[#0d1117] text-[#f0ece2] font-['Space_Grotesk'] px-6">
      <Background />

      <div className="relative z-10 w-full max-w-[420px] rounded-3xl border border-white/15 bg-[#0d1117]/80 backdrop-blur-sm p-10 sm:p-12">
        <p className="font-['JetBrains_Mono'] text-xs tracking-[0.15em] text-[#9a9a93] mb-2">
          AUTHENTICATION
        </p>
        <h1 className="text-3xl font-bold tracking-tight mb-8">LOGIN</h1>

        {fields.map((field) => (
          <div key={field.name} className="mb-4">
            <p className="font-['JetBrains_Mono'] text-[11px] tracking-[0.15em] text-[#9a9a93] mb-1.5">
              {field.label}
            </p>
            <input
              name={field.name}
              type={field.type}
              value={form[field.name]}
              onChange={handle}
              placeholder={field.placeholder}
              className="w-full rounded-md border border-white/20 bg-transparent px-4 py-3 font-['JetBrains_Mono'] text-sm text-[#f0ece2] outline-none transition-colors focus:border-[#c97539] placeholder:text-[#6e6e66]"
            />
          </div>
        ))}

        {error && (
          <p className="mb-4 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 font-['JetBrains_Mono'] text-xs text-red-400">
            ✗ {error}
          </p>
        )}

        <button
          onClick={submit}
          disabled={loading}
          className="mt-2 w-full rounded-md bg-[#b56126] py-3.5 font-['JetBrains_Mono'] text-sm font-bold tracking-[0.1em] text-[#0d1117] transition-opacity disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "AUTHENTICATING..." : "[ LOGIN ]"}
        </button>

        <p className="mt-6 text-center font-['JetBrains_Mono'] text-xs text-[#9a9a93]">
          NO ACCOUNT?{" "}
          <Link to="/signup" className="text-[#f0ece2] hover:text-[#c97539] transition-colors">
            SIGNUP →
          </Link>
        </p>
      </div>
    </div>
  );
}