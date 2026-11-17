
import PolicyLayout, { PolicySection } from './PolicyLayout';

export default function PrivacyPolicy() {
  return (
    <PolicyLayout title="Privacy Policy" lastUpdated="October 2026">
      <PolicySection number={1} title="Information We Collect">
        <p>We collect information you provide directly to us, such as your email address when creating an account. We also process datasets you upload to provide our machine learning analysis services.</p>
      </PolicySection>
      
      <PolicySection number={2} title="How We Use Your Information">
        <p>We use the information we collect to provide, maintain, and improve our services, including processing your datasets to generate models and insights. Your uploaded data is not used to train global AI models without your explicit consent.</p>
      </PolicySection>

      <PolicySection number={3} title="Data Security and Sub-processors">
        <p>We implement appropriate security measures to protect your data. We use trusted third-party sub-processors (such as MongoDB for secure database storage and Groq for LLM processing) to operate our service. All sub-processors are vetted for strict privacy compliance.</p>
      </PolicySection>

      <PolicySection number={4} title="Your Rights">
        <p>You have the right to access, update, or delete your personal information. You can delete uploaded datasets at any time from Your Library, and contact us to delete your account.</p>
      </PolicySection>
    </PolicyLayout>
  );
}
