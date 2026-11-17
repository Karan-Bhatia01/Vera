/** Client-side email/password checks shared by Login and Signup. Returns an error message or null. */
export default function validateCredentials({ email, password }) {
  if (!email) return "Email is required";
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return "Invalid email format";
  if (!password) return "Password is required";
  if (password.length < 6) return "Password must be at least 6 characters";
  return null;
}
