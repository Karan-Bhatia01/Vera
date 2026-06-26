import { Link } from "react-router-dom";

export default function CTA() {
  return (
    <section className="px-6 pb-24 text-center">
      <h2 className="text-4xl font-extrabold">
        Ready to transform your data?
      </h2>
      <p className="mt-3 text-[#9a9a93]">
        Start with a free upload. No credit card required.
      </p>
      <Link
        to="#"
        onClick={(e) => e.preventDefault()}
        className="mt-7 inline-block rounded-full bg-[#b56126] px-8 py-3.5 font-semibold text-[#0d1117]"
      >
        Upload Your First Dataset
      </Link>
    </section>
  );
}