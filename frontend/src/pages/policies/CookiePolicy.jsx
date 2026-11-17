
import PolicyLayout, { PolicySection } from './PolicyLayout';

export default function CookiePolicy() {
  return (
    <PolicyLayout title="Cookie Policy" lastUpdated="October 2026">
      <PolicySection number={1} title="What Are Cookies?">
        <p>Cookies are small text files that are stored on your device when you visit our website. Vera does not set cookies of its own; it uses your browser's local storage to make the app work and remember your preferences.</p>
      </PolicySection>
      
      <PolicySection number={2} title="Strictly Necessary Storage">
        <p>We store your sign-in token in local storage so you stay logged in. Clearing your browser storage will sign you out.</p>
      </PolicySection>

      <PolicySection number={3} title="Preference Cookies">
        <p>We use local storage to remember your preferences, such as your chosen theme (dark/light mode). This prevents you from having to re-select your preferences on every visit.</p>
      </PolicySection>

      <PolicySection number={4} title="No Unnecessary Tracking">
        <p>We believe in minimal data collection. We do not use third-party tracking cookies or advertising pixels to track your behavior across other websites.</p>
      </PolicySection>
    </PolicyLayout>
  );
}
