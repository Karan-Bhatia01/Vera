
import PolicyLayout, { PolicySection } from './PolicyLayout';

export default function TermsOfService() {
  return (
    <PolicyLayout title="Terms of Service" lastUpdated="October 2026">
      <PolicySection number={1} title="Acceptance of Terms">
        <p>By accessing and using Vera, you agree to be bound by these Terms of Service. If you do not agree to these terms, please do not use our services.</p>
      </PolicySection>
      
      <PolicySection number={2} title="Use of Service">
        <p>Vera provides automated machine learning and data analysis tools. You agree to use the service only for lawful purposes. You must have the legal right and necessary permissions to upload and process any dataset you submit to Vera.</p>
      </PolicySection>

      <PolicySection number={3} title="User Data and Content">
        <p>You retain all rights to the datasets you upload. By uploading data, you grant Vera a temporary, limited license to process the data solely for the purpose of providing the requested analysis and models to you.</p>
      </PolicySection>

      <PolicySection number={4} title="Acceptable Use Policy">
        <p>You agree not to upload malicious code, attempt to breach our security, reverse-engineer our models, or use Vera to process illegal, highly sensitive, or non-consensual personal data.</p>
      </PolicySection>

      <PolicySection number={5} title="Disclaimer of Warranties">
        <p>Our machine learning analysis is provided "as is". While we strive for accuracy, Vera does not guarantee the absolute correctness of any generated model or insight. You are responsible for validating the models before using them in critical environments.</p>
      </PolicySection>
    </PolicyLayout>
  );
}
