#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vector Quest — a tiny GUI game to practice 2D vectors
No external dependencies (only Python stdlib + tkinter).
Tested on Python 3.8+.

How to run:
  python vector_quest.py
"""

import sys
import math
import random
import json
import os
import time

# --- Dependency / Environment check (tkinter) ---
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception as e:
    print("This game needs Tkinter (usually included with Python).")
    print("On Debian/Ubuntu:    sudo apt-get install python3-tk")
    print("On Fedora:            sudo dnf install python3-tkinter")
    print("On Arch:              sudo pacman -S tk")
    print("On macOS (Homebrew):  brew install python-tk@3")
    print("On Windows: Tkinter is included in the normal Python installer from python.org.")
    print("\nOriginal error:", e)
    sys.exit(1)


APP_TITLE = "Vector Quest"
SAVE_FILE = "vector_quest_highscore.json"

# -------------------- Utility functions --------------------

def vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1])

def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1])

def vec_scale(k, a):
    return (k * a[0], k * a[1])

def vec_dot(a, b):
    return a[0]*b[0] + a[1]*b[1]

def vec_cross_2d(a, b):
    # 2D cross "z-component" (scalar)
    return a[0]*b[1] - a[1]*b[0]

def vec_len(a):
    return math.sqrt(a[0]**2 + a[1]**2)

def is_parallel(a, b):
    # two non-zero vectors are parallel if cross product is zero
    # allow small tolerance
    if a == (0,0) or b == (0,0):
        return False
    return abs(vec_cross_2d(a, b)) < 1e-9

def is_antiparallel(a, b):
    # antiparallel if parallel and dot < 0
    if not is_parallel(a, b):
        return False
    return vec_dot(a, b) < 0

def simplify_tuple(t):
    # Pretty printing tuple (int if near integer)
    def simp(x):
        if abs(x - round(x)) < 1e-9:
            return int(round(x))
        return round(x, 4)
    return (simp(t[0]), simp(t[1]))

def round_if_close(x):
    return int(round(x)) if abs(x - round(x)) < 1e-9 else round(x, 4)

# -------------------- Problem generation --------------------

PROBLEM_TYPES = [
    "add",          # u + v
    "scale",        # k * u
    "dot",          # u · v
    "cross2d",      # 2D cross (scalar z)
    "length",       # |u|
    "parallel",     # are u and v parallel? (yes/no)
    "antiparallel", # are u and v anti-parallel? (yes/no)
]

def random_vector(lo=-6, hi=6, avoid_zero=False):
    while True:
        a = (random.randint(lo, hi), random.randint(lo, hi))
        if avoid_zero and a == (0,0):
            continue
        return a

def random_scalar(lo=-6, hi=6, avoid_zero=False):
    while True:
        k = random.randint(lo, hi)
        if avoid_zero and k == 0:
            continue
        return k

def gen_problem(difficulty=1):
    """
    difficulty 1..5 influences magnitude & numeric type.
    For now we keep integer components to keep entry simple.
    """
    max_abs = {1:4, 2:6, 3:8, 4:10, 5:12}.get(difficulty, 6)
    ptype = random.choice(PROBLEM_TYPES)

    if ptype == "add":
        u = random_vector(-max_abs, max_abs, avoid_zero=True)
        v = random_vector(-max_abs, max_abs, avoid_zero=True)
        answer = vec_add(u, v)
        prompt = f"Compute u + v for u={u}, v={v}. Answer as x,y"
        return {"type":"vector", "ptype":ptype, "u":u, "v":v, "answer":answer, "prompt":prompt, "explain": f"(u+v) = ({u[0]}+{v[0]}, {u[1]}+{v[1]}) = {answer}"}

    if ptype == "scale":
        u = random_vector(-max_abs, max_abs, avoid_zero=True)
        k = random_scalar(-max_abs, max_abs, avoid_zero=True)
        answer = vec_scale(k, u)
        prompt = f"Compute k·u for k={k}, u={u}. Answer as x,y"
        return {"type":"vector", "ptype":ptype, "u":u, "k":k, "answer":answer, "prompt":prompt, "explain": f"k·u = {k}·{u} = {answer}"}

    if ptype == "dot":
        u = random_vector(-max_abs, max_abs, avoid_zero=True)
        v = random_vector(-max_abs, max_abs, avoid_zero=True)
        answer = vec_dot(u, v)
        prompt = f"Compute the dot product u·v for u={u}, v={v}. Answer as a number"
        return {"type":"scalar", "ptype":ptype, "u":u, "v":v, "answer":answer, "prompt":prompt, "explain": f"u·v = {u[0]}·{v[0]} + {u[1]}·{v[1]} = {answer}"}

    if ptype == "cross2d":
        u = random_vector(-max_abs, max_abs, avoid_zero=True)
        v = random_vector(-max_abs, max_abs, avoid_zero=True)
        answer = vec_cross_2d(u, v)
        prompt = f"Compute the 2D cross (z-component) u×v for u={u}, v={v}. Answer as a number"
        return {"type":"scalar", "ptype":ptype, "u":u, "v":v, "answer":answer, "prompt":prompt, "explain": f"u×v (2D z) = {u[0]}·{v[1]} - {u[1]}·{v[0]} = {answer}"}

    if ptype == "length":
        u = random_vector(-max_abs, max_abs, avoid_zero=True)
        answer = vec_len(u)
        prompt = f"Compute the length |u| for u={u}. Answer as a number (you can use sqrt -> decimal)"
        return {"type":"float", "ptype":ptype, "u":u, "answer":answer, "prompt":prompt, "explain": f"|u| = sqrt({u[0]}² + {u[1]}²) = {round(answer,4)}"}

    if ptype == "parallel":
        u = random_vector(-max_abs, max_abs, avoid_zero=True)
        v = random_vector(-max_abs, max_abs, avoid_zero=True)
        # Ensure not ambiguous by retrying if length or near-parallel but not exact
        # We'll enforce integer multiples  v = m*u  sometimes, else clearly not parallel.
        if random.random() < 0.5:
            m = random_scalar(-5, 5, avoid_zero=True)
            v = vec_scale(m, u)
            answer = True
            explain = f"v = {m}·u → parallel"
        else:
            # try until clearly not parallel
            while is_parallel(u, v):
                v = random_vector(-max_abs, max_abs, avoid_zero=True)
            answer = False
            explain = "Cross product != 0 → not parallel"
        prompt = f"Are u and v parallel? Answer 'yes' or 'no' for u={u}, v={v}"
        return {"type":"yesno", "ptype":ptype, "u":u, "v":v, "answer":answer, "prompt":prompt, "explain": explain}

    if ptype == "antiparallel":
        u = random_vector(-max_abs, max_abs, avoid_zero=True)
        if random.random() < 0.5:
            m = random_scalar(-5, 5, avoid_zero=True)
            m = -abs(m)  # ensure opposite direction
            v = vec_scale(m, u)
            answer = True
            explain = f"v = {m}·u with m<0 → anti-parallel"
        else:
            # choose a non antiparallel option
            v = random_vector(-max_abs, max_abs, avoid_zero=True)
            # reject antiparallel cases
            while is_antiparallel(u, v):
                v = random_vector(-max_abs, max_abs, avoid_zero=True)
            answer = False
            explain = "Not a negative multiple → not anti-parallel"
        prompt = f"Are u and v anti-parallel? Answer 'yes' or 'no' for u={u}, v={v}"
        return {"type":"yesno", "ptype":ptype, "u":u, "v":v, "answer":answer, "prompt":prompt, "explain": explain}

    # fallback
    return gen_problem(difficulty)


# -------------------- Main App --------------------

class VectorQuestApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.reveal_answer = False
        self.title(APP_TITLE)
        self.geometry("980x680")
        self.minsize(900, 620)
        self.configure(bg="#0f172a")  # slate-900

        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except:
            pass

        self.score = 0
        self.streak = 0
        self.best = self.load_high_score()
        self.difficulty = tk.IntVar(value=2)
        self.current = None
        self.start_time = time.time()

        self._build_ui()
        self.new_problem()

    # ---------- Persistence ----------
    def load_high_score(self):
        try:
            if os.path.exists(SAVE_FILE):
                with open(SAVE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return int(data.get("highscore", 0))
        except:
            pass
        return 0

    def save_high_score(self):
        try:
            with open(SAVE_FILE, "w", encoding="utf-8") as f:
                json.dump({"highscore": self.best}, f, indent=2)
        except Exception as e:
            print("Couldn't save high score:", e)

    # ---------- UI ----------
    def _build_ui(self):
        # Top header
        header = tk.Frame(self, bg="#0f172a")
        header.pack(fill="x", pady=8)

        tk.Label(header, text="Vector Quest", fg="white", bg="#0f172a",
                 font=("Segoe UI", 22, "bold")).pack(side="left", padx=16)

        self.lbl_score = tk.Label(header, text="Score: 0   Streak: 0   Best: 0",
                                  fg="#93c5fd", bg="#0f172a", font=("Consolas", 14))
        self.lbl_score.pack(side="right", padx=16)

        # Main content split: left controls, right canvas
        main = tk.Frame(self, bg="#0b1220")
        main.pack(fill="both", expand=True, padx=12, pady=8)

        left = tk.Frame(main, bg="#0b1220")
        left.pack(side="left", fill="y", padx=(8, 8), pady=8)

        right = tk.Frame(main, bg="#0b1220")
        right.pack(side="right", fill="both", expand=True, padx=(8, 8), pady=8)

        # Left controls
        card = tk.Frame(left, bg="#111827", bd=0, highlightbackground="#1f2937", highlightthickness=1)
        card.pack(fill="x", pady=6)

        tk.Label(card, text="Difficulty", fg="white", bg="#111827",
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=12, pady=(10, 4))

        s = ttk.Scale(card, from_=1, to=5, orient="horizontal", command=self.on_difficulty_change,
                      value=self.difficulty.get())
        s.pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(card, text="1 = Easy ··· 5 = Hard", fg="#9ca3af", bg="#111827",
                 font=("Segoe UI", 9)).pack(anchor="w", padx=12, pady=(0, 10))

        # Problem / Answer card
        qa = tk.Frame(left, bg="#111827", bd=0, highlightbackground="#1f2937", highlightthickness=1)
        qa.pack(fill="both", expand=True, pady=6)

        tk.Label(qa, text="Task", fg="white", bg="#111827", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=12, pady=(10, 4))

        self.prompt_text = tk.Text(qa, height=4, wrap="word", fg="#e5e7eb", bg="#0b1220", bd=0, padx=10, pady=8)
        self.prompt_text.pack(fill="x", padx=12, pady=(0, 8))
        self.prompt_text.config(state="disabled")

        tk.Label(qa, text="Your answer:", fg="#d1d5db", bg="#111827").pack(anchor="w", padx=12)

        self.answer_entry = ttk.Entry(qa, font=("Consolas", 12))
        self.answer_entry.pack(fill="x", padx=12, pady=6)
        self.answer_entry.bind("<Return>", lambda e: self.check_answer())

        btns = tk.Frame(qa, bg="#111827")
        btns.pack(fill="x", padx=12, pady=(4, 12))

        ttk.Button(btns, text="Check", command=self.check_answer).pack(side="left")
        ttk.Button(btns, text="Hint", command=self.show_hint).pack(side="left", padx=6)
        ttk.Button(btns, text="Show Answer", command=self.show_answer).pack(side="left")
        ttk.Button(btns, text="Next ▶", command=self.new_problem).pack(side="right")

        self.feedback = tk.Label(qa, text="", fg="#a7f3d0", bg="#111827", font=("Segoe UI", 11))
        self.feedback.pack(anchor="w", padx=12, pady=(0, 6))

        # Learn card
        learn = tk.Frame(left, bg="#111827", bd=0, highlightbackground="#1f2937", highlightthickness=1)
        learn.pack(fill="x", pady=6)

        tk.Label(learn, text="Quick Reference", fg="white", bg="#111827",
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=12, pady=(10, 4))

        ref = (
            "Vector u=(ux,uy)\n"
            "Addition: u+v=(ux+vx, uy+vy)\n"
            "Scalar: k·u=(k·ux, k·uy)\n"
            "Dot: u·v=ux·vx+uy·vy → angle via cosθ=(u·v)/(|u||v|)\n"
            "2D Cross (z): ux·vy - uy·vx (sign = orientation)\n"
            "Length: |u|=√(ux²+uy²)\n"
            "Parallel: cross=0 & both ≠0\n"
            "Anti-parallel: parallel & dot<0"
        )
        ref_lbl = tk.Text(learn, height=10, wrap="word", fg="#e5e7eb", bg="#0b1220", bd=0, padx=10, pady=8)
        ref_lbl.insert("1.0", ref)
        ref_lbl.config(state="disabled")
        ref_lbl.pack(fill="x", padx=12, pady=(0, 10))

        # Right canvas panel
        canvas_card = tk.Frame(right, bg="#111827", bd=0, highlightbackground="#1f2937", highlightthickness=1)
        canvas_card.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_card, bg="#0b1220", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        self.status = tk.Label(self, text="Have fun! 🧠➕➡️", fg="#c7d2fe", bg="#0f172a")
        self.status.pack(fill="x")

        # resize handling
        self.canvas.bind("<Configure>", lambda e: self.redraw_vectors())

    # ---------- Events ----------
    def on_difficulty_change(self, val):
        try:
            self.difficulty.set(int(float(val)))
        except:
            pass

    def set_prompt(self, text):
        self.prompt_text.config(state="normal")
        self.prompt_text.delete("1.0", "end")
        self.prompt_text.insert("1.0", text)
        self.prompt_text.config(state="disabled")

    def update_scorebar(self):
        self.lbl_score.config(text=f"Score: {self.score}   Streak: {self.streak}   Best: {self.best}")

    def new_problem(self):
        self.current = gen_problem(self.difficulty.get())
        self.reveal_answer = False
        self.set_prompt(self.current["prompt"])
        self.answer_entry.delete(0, "end")
        self.feedback.config(text="", fg="#a7f3d0")
        self.status.config(text=f"New task: {self.current['ptype']}")
        self.start_time = time.time()
        self.redraw_vectors()

    def show_hint(self):
        if not self.current:
            return
        p = self.current
        hint = ""
        if p["ptype"] == "add":
            hint = "Add x-components and y-components separately."
        elif p["ptype"] == "scale":
            hint = "Multiply each component by k."
        elif p["ptype"] == "dot":
            hint = "ux·vx + uy·vy"
        elif p["ptype"] == "cross2d":
            hint = "ux·vy - uy·vx (sign shows rotation)."
        elif p["ptype"] == "length":
            hint = "√(ux²+uy²)"
        elif p["ptype"] == "parallel":
            hint = "Parallel if the cross (2D) is 0 (and both non-zero)."
        elif p["ptype"] == "antiparallel":
            hint = "Anti-parallel if parallel AND dot product < 0."
        self.feedback.config(text="Hint: " + hint, fg="#fde68a")

    def parse_answer(self, text):
        text = text.strip().lower()
        # vector format "x,y"
        if "," in text:
            parts = text.split(",")
            if len(parts) == 2:
                try:
                    x = float(parts[0].strip())
                    y = float(parts[1].strip())
                    return ("vector", (x, y))
                except:
                    pass
        # yes/no
        if text in ("yes", "y", "ja", "true"):
            return ("yesno", True)
        if text in ("no", "n", "nein", "false"):
            return ("yesno", False)
        # number
        try:
            val = float(text)
            return ("number", val)
        except:
            return (None, None)

    def check_answer(self):
        if not self.current:
            return
        user_text = self.answer_entry.get()
        kind, val = self.parse_answer(user_text)
        p = self.current

        correct = False

        if p["type"] == "vector" and kind == "vector":
            exp = p["answer"]
            # accept minor numeric tolerance
            correct = (abs(val[0] - exp[0]) < 1e-6) and (abs(val[1] - exp[1]) < 1e-6)

        elif p["type"] in ("scalar", "float") and kind == "number":
            exp = p["answer"]
            # for float (length), allow small tolerance
            tol = 1e-6 if p["type"] == "scalar" else 1e-3
            correct = abs(val - exp) <= tol

        elif p["type"] == "yesno" and kind == "yesno":
            correct = (val == p["answer"])

        if correct:
            elapsed = time.time() - self.start_time
            bonus = max(0, int(10 - elapsed))  # quick-answer bonus
            gained = 10 + bonus + self.streak
            self.score += gained
            self.streak += 1
            self.best = max(self.best, self.score)
            self.save_high_score()
            self.reveal_answer = True
            self.feedback.config(text=f"✅ Correct! +{gained} points (bonus {bonus}). {p.get('explain','')}", fg="#a7f3d0")
            self.update_scorebar()
            self.redraw_vectors(success=True)
        else:
            self.streak = 0
            self.update_scorebar()
            self.feedback.config(text="❌ Not quite. Try 'Hint' or 'Show Answer'.", fg="#fecaca")
            self.redraw_vectors(success=False)

    def show_answer(self):
        if not self.current:
            return
        self.reveal_answer = True
        self.redraw_vectors()
        p = self.current
        if p["type"] == "vector":
            ans = simplify_tuple(p["answer"])
            msg = f"Answer: {ans}   •  {p.get('explain','')}"
        elif p["type"] in ("scalar", "float"):
            msg = f"Answer: {round_if_close(p['answer'])}   •  {p.get('explain','')}"
        elif p["type"] == "yesno":
            msg = f"Answer: {'yes' if p['answer'] else 'no'}   •  {p.get('explain','')}"
        else:
            msg = "No answer."
        self.feedback.config(text=msg, fg="#c7d2fe")

    # ---------- Drawing ----------
    def redraw_vectors(self, success=None):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 100 or h < 100:
            return
        # draw axes
        cx, cy = w//2, h//2
        self.canvas.create_line(0, cy, w, cy, fill="#334155", width=1)
        self.canvas.create_line(cx, 0, cx, h, fill="#334155", width=1)
        for dx in range(-int(w/2), int(w/2), 20):
            self.canvas.create_line(cx+dx, cy-3, cx+dx, cy+3, fill="#1f2937")
        for dy in range(-int(h/2), int(h/2), 20):
            self.canvas.create_line(cx-3, cy+dy, cx+3, cy+dy, fill="#1f2937")

        if not self.current:
            return

        scale = 18  # pixels per unit
        p = self.current

        # vectors to draw:
        vecs = []
        colors = ["#93c5fd", "#a7f3d0", "#fca5a5", "#fde68a"]
        labels = []

        if "u" in p:
            vecs.append(p["u"]); labels.append("u")
        if "v" in p:
            vecs.append(p["v"]); labels.append("v")

        # For addition, also show u+v
        if self.reveal_answer and p["ptype"] == "add":
            vecs.append(p["answer"]); labels.append("u+v")
        if self.reveal_answer and p["ptype"] == "scale":
            vecs.append(p["answer"]); labels.append("k·u")

        # Draw arrows
        for i, v in enumerate(vecs):
            color = colors[i % len(colors)]
            self.draw_arrow(cx, cy, v, scale, color, labels[i])

        # status flair
        if success is True:
            self.status.config(text="Great! ✅", fg="#86efac")
        elif success is False:
            self.status.config(text="Keep trying… ❌", fg="#fecaca")

    def draw_arrow(self, cx, cy, v, scale, color, label):
        x = cx + v[0]*scale
        y = cy - v[1]*scale
        self.canvas.create_line(cx, cy, x, y, fill=color, width=3, arrow="last", arrowshape=(12,14,5))
        self.canvas.create_oval(x-3, y-3, x+3, y+3, fill=color, outline=color)
        self.canvas.create_text(x + 10, y - 10, text=f"{label}={simplify_tuple(v)}", fill=color, anchor="w", font=("Segoe UI", 10, "bold"))


def main():
    app = VectorQuestApp()
    app.mainloop()


if __name__ == "__main__":
    main()
