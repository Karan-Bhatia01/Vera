import { Routes, Route } from "react-router-dom";
import LandingPage from "./components/LandingPage";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Upload from "./pages/Upload";
import DashboardLayout from "./components/dashboard/DashboardLayout";
import DashboardHome from "./pages/dashboard/DashboardHome";
import DataInfo from "./pages/dashboard/DataInfo";
import DataEDA from "./pages/dashboard/DataEDA";
import MLModelling from "./pages/dashboard/MLModelling";
import ProtectedRoute from "./components/ProtectedRoute";

function App() {
  return (
    <Routes>
      {/* Public */}
      <Route path="/" element={<LandingPage />} />
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />

      {/* Require a valid login */}
      <Route path="/upload" element={<ProtectedRoute><Upload /></ProtectedRoute>} />
      <Route
        path="/dashboard"
        element={<ProtectedRoute><DashboardLayout /></ProtectedRoute>}
      >
        <Route index element={<DashboardHome />} />
        <Route path="info" element={<DataInfo />} />
        <Route path="eda" element={<DataEDA />} />
        <Route path="ml" element={<MLModelling />} />
      </Route>
    </Routes>
  );
}

export default App;