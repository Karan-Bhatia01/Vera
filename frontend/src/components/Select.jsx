import { useEffect, useRef, useState } from "react";

const MENU_HEIGHT = 288; // max-h-72

/**
 * Styled single-select (ARIA "select-only combobox" pattern).
 * Keyboard: ↑/↓ move, Home/End jump, Enter/Space choose, Esc closes, typing jumps to a match.
 */
export default function Select({ id, value, onChange, options, placeholder = "Choose…", labelledBy }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [openUp, setOpenUp] = useState(false);
  const rootRef = useRef(null);
  const buttonRef = useRef(null);
  const optionRefs = useRef([]);
  const typeahead = useRef({ text: "", timer: null });

  const selectedIndex = options.indexOf(value);
  const listId = `${id}-listbox`;

  const openMenu = (index = selectedIndex >= 0 ? selectedIndex : 0) => {
    // Flip upwards when there isn't room below the button.
    const rect = buttonRef.current.getBoundingClientRect();
    setOpenUp(window.innerHeight - rect.bottom < MENU_HEIGHT && rect.top > window.innerHeight - rect.bottom);
    setActive(index);
    setOpen(true);
  };

  const closeMenu = () => {
    setOpen(false);
    buttonRef.current?.focus();
  };

  const choose = (index) => {
    onChange(options[index]);
    closeMenu();
  };

  // Keep the highlighted option visible while moving with the keyboard.
  useEffect(() => {
    if (open) optionRefs.current[active]?.scrollIntoView({ block: "nearest" });
  }, [open, active]);

  // Close on a click anywhere outside.
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e) => {
      if (!rootRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [open]);

  const jumpToTyped = (key) => {
    const t = typeahead.current;
    clearTimeout(t.timer);
    t.text += key.toLowerCase();
    t.timer = setTimeout(() => (t.text = ""), 600);
    const start = open ? active + 1 : 0;
    const ordered = [...options.slice(start), ...options.slice(0, start)];
    const hit = ordered.findIndex((o) => o.toLowerCase().startsWith(t.text));
    if (hit === -1) return;
    const index = (hit + start) % options.length;
    if (open) setActive(index);
    else onChange(options[index]);
  };

  const onKeyDown = (e) => {
    if (!options.length) return;
    const last = options.length - 1;
    const keys = {
      ArrowDown: () => (open ? setActive((i) => Math.min(last, i + 1)) : openMenu()),
      ArrowUp: () => (open ? setActive((i) => Math.max(0, i - 1)) : openMenu()),
      Home: () => open && setActive(0),
      End: () => open && setActive(last),
      Enter: () => (open ? choose(active) : openMenu()),
      " ": () => (open ? choose(active) : openMenu()),
      Escape: () => open && closeMenu(),
      Tab: () => open && setOpen(false),
    };
    if (keys[e.key]) {
      if (e.key !== "Tab") e.preventDefault();
      keys[e.key]();
    } else if (e.key.length === 1 && /\S/.test(e.key)) {
      jumpToTyped(e.key);
    }
  };

  return (
    <div ref={rootRef} className="relative">
      <button
        ref={buttonRef}
        id={id}
        type="button"
        role="combobox"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listId}
        aria-labelledby={labelledBy ? `${labelledBy} ${id}` : undefined}
        aria-activedescendant={open ? `${id}-opt-${active}` : undefined}
        onClick={() => (open ? setOpen(false) : openMenu())}
        onKeyDown={onKeyDown}
        className={`flex w-full items-center justify-between gap-3 rounded-xl border-2 bg-[var(--surface)] px-4 py-3 text-left text-base transition-colors ${
          open ? "border-[var(--sky-ink)]" : "border-[var(--line)] hover:border-[var(--muted)]/50"
        }`}
      >
        <span className={`truncate ${value ? "font-semibold" : "text-[var(--muted)]"}`}>{value || placeholder}</span>
        <svg
          width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
          className={`shrink-0 text-[var(--muted)] transition-transform duration-150 ${open ? "rotate-180" : ""}`}
          aria-hidden="true"
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {open && (
        <ul
          id={listId}
          role="listbox"
          aria-labelledby={labelledBy}
          className={`absolute z-50 max-h-72 w-full overflow-y-auto rounded-xl border border-[var(--line)] bg-[var(--surface)] p-1.5 shadow-[0_12px_32px_rgba(30,36,82,0.18)] ${
            openUp ? "bottom-full mb-2" : "top-full mt-2"
          }`}
        >
          {options.map((option, i) => {
            const selected = option === value;
            return (
              <li
                key={option}
                id={`${id}-opt-${i}`}
                ref={(el) => (optionRefs.current[i] = el)}
                role="option"
                aria-selected={selected}
                onPointerEnter={() => setActive(i)}
                onPointerDown={(e) => e.preventDefault()} // keep focus on the button
                onClick={() => choose(i)}
                className={`flex cursor-pointer items-center gap-3 rounded-lg px-3 py-2.5 ${
                  i === active ? "bg-[var(--surface-2)]" : ""
                } ${selected ? "font-bold" : ""}`}
              >
                <span className="flex h-4 w-4 shrink-0 items-center justify-center" aria-hidden="true">
                  {selected && (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--sky-ink)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  )}
                </span>
                <span className="truncate">{option}</span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
