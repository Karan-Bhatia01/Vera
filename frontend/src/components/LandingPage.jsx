import Background from "./Background";
import ConnectionStatus from "./ConnectionStatus";
import Navbar from "./landing/Navbar";
import Hero from "./landing/Hero";
import Stats from "./landing/Stats";
import Pipeline from "./landing/Pipeline";
import Features from "./landing/Features";
import CTA from "./landing/CTA";
import Footer from "./landing/Footer";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[#0d1117] text-[#f0ece2] font-['Space_Grotesk'] relative">
      <Background />
      <ConnectionStatus />
      <div className="relative z-10">
        <Navbar />
        <Hero />
        <Stats />
        <Pipeline />
        <Features />
        <CTA />
        <Footer />
      </div>
    </div>
  );
}