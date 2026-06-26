import { useEffect, useRef } from "react";

export default function Background() {
  const glowRef = useRef(null);
  const pos = useRef({ x: 0.5, y: 0.3 });
  const target = useRef({ x: 0.5, y: 0.3 });

  useEffect(() => {
    const handleMove = (e) => {
      target.current = {
        x: e.clientX / window.innerWidth,
        y: e.clientY / window.innerHeight,
      };
    };
    window.addEventListener("mousemove", handleMove);

    let frame;
    const animate = () => {
      pos.current.x += (target.current.x - pos.current.x) * 0.08;
      pos.current.y += (target.current.y - pos.current.y) * 0.08;

      if (glowRef.current) {
        glowRef.current.style.background = `
          radial-gradient(40% 32% at ${pos.current.x * 100}% ${pos.current.y * 100}%, #b5612638 0%, transparent 70%),
          radial-gradient(55% 42% at 50% 100%, #2a3a4a4d 0%, transparent 70%),
          radial-gradient(45% 35% at 85% 8%, #8a6a3a2e 0%, transparent 70%)
        `;
      }
      frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener("mousemove", handleMove);
      cancelAnimationFrame(frame);
    };
  }, []);

  return (
    <div className="fixed inset-0 overflow-hidden">
      <div
        className="absolute inset-0"
        style={{
          backgroundImage:
            "radial-gradient(circle at 1px 1px, #ffffff10 1px, transparent 0)",
          backgroundSize: "24px 24px",
        }}
      />
      <div
        ref={glowRef}
        className="absolute inset-0 transition-[background] duration-100"
      />
    </div>
  );
}