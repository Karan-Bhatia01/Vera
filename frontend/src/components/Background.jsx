export default function Background() {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none z-[-1]">
      <div
        className="absolute inset-0 opacity-[0.25]"
        style={{ filter: "url(#noiseFilter)" }}
      />
    </div>
  );
}
