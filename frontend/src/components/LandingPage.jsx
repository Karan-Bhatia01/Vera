import Navbar from "./landing/Navbar";
import Hero from "./landing/Hero";
import HowItWorks from "./landing/HowItWorks";
import CTA from "./landing/CTA";
import FAQ from "./landing/FAQ";
import Footer from "./landing/Footer";

export default function LandingPage() {
  return (
    <div className="min-h-screen">
      <Navbar />
      <main>
        <Hero />
        <HowItWorks />
        <FAQ />
        <CTA />
      </main>
      <Footer />
    </div>
  );
}
