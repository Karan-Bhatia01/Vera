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
import CookieConsent from "./components/CookieConsent";

import PrivacyPolicy from "./pages/policies/PrivacyPolicy";
import TermsOfService from "./pages/policies/TermsOfService";
import CookiePolicy from "./pages/policies/CookiePolicy";

import { useEffect } from "react";

function App() {
  useEffect(() => {
    const theme = localStorage.getItem("app-theme") || "dark";
    document.documentElement.dataset.theme = theme;
  }, []);

  return (
    <>
      <CookieConsent />
      <Routes>
        {/* Public */}
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        
        {/* Policies */}
        <Route path="/privacy-policy" element={<PrivacyPolicy />} />
        <Route path="/terms-of-service" element={<TermsOfService />} />
        <Route path="/cookie-policy" element={<CookiePolicy />} />

        {/* Require a valid login */}
        <Route
          element={
            <ProtectedRoute>
              <DashboardLayout />
            </ProtectedRoute>
          }
        >
          <Route path="/upload" element={<Upload />} />
          <Route path="/dashboard" element={<DashboardHome />} />
          <Route path="/dashboard/info" element={<DataInfo />} />
          <Route path="/dashboard/eda" element={<DataEDA />} />
          <Route path="/dashboard/ml" element={<MLModelling />} />
        </Route>
      </Routes>
    </>
  );
}

export default App;
