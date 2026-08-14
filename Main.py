import cv2
import numpy as np
import mediapipe as mp
import random
import time
import math

# --- Setup & Config ---
mp_hands = mp.tasks.vision.HandLandmarker
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=1, min_hand_detection_confidence=0.5)
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

ACCENT = (255, 100, 200)
TEXT_C = (250, 250, 250)
SKIN = (120, 180, 240)
SHADOW = (70, 110, 160)


# --- Game Logic ---
def get_winner(c1, c2):
    return c1 if {"rock": "scissors", "paper": "rock", "scissors": "paper"}[c1] == c2 else c2


def eval_round(players, choices):
    uniq = list(set(choices.values()))
    if len(uniq) != 2: return "DRAW! No one eliminated."
    win_choice = get_winner(uniq[0], uniq[1])
    elim = [p for p, c in choices.items() if c != win_choice]
    for p in elim: players[p] = False
    return f"{win_choice.upper()} WINS! Eliminated: {', '.join(elim)}"


def dist(p1, p2): return math.hypot(p1.x - p2.x, p1.y - p2.y)


def detect_gesture(lm):
    fingers = sum([dist(lm[tip], lm[0]) > dist(lm[pip], lm[0]) * 1.1
                   for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]])
    if fingers <= 1: return "rock"
    if fingers >= 4: return "paper"
    return "scissors" if dist(lm[16], lm[0]) < dist(lm[14], lm[0]) * 1.1 else "paper"


# --- Graphics & 3D Background ---
def draw_3d_beach(canvas, t):
    canvas[:] = (25, 10, 35)
    cv2.circle(canvas, (640, 250), 90, (0, 120, 255), -1, cv2.LINE_AA)
    cv2.line(canvas, (0, 250), (1280, 250), (255, 0, 150), 2)

    for i in range(-10, 11):
        x_bottom = 640 + i * 180
        cv2.line(canvas, (640, 250), (x_bottom, 720), (100, 30, 80), 1)

    for i in range(1, 24):
        depth = i + (t * 2.5) % 1
        if depth < 0.1: continue
        y_base = 250 + (depth ** 2) * 1.2
        if y_base > 720: continue

        pts = []
        amp = depth * 1.8
        for x in range(0, 1281, 30):
            wave = math.sin(x * 0.015 - t * 3 + depth) * amp
            pts.append([x, int(y_base + wave)])

        if len(pts) > 1:
            bright = min(255, 50 + int(depth * 10))
            cv2.polylines(canvas, [np.array(pts)], False, (255, 100, bright), max(1, int(depth / 4)), cv2.LINE_AA)


def draw_finger(img, base, angle, length, w):
    rad = math.radians(angle - 90)
    mid = (int(base[0] + length * 0.55 * math.cos(rad)), int(base[1] + length * 0.55 * math.sin(rad)))
    tip = (int(mid[0] + length * 0.45 * math.cos(rad)), int(mid[1] + length * 0.45 * math.sin(rad)))
    cv2.line(img, base, mid, SKIN, w, cv2.LINE_AA)
    cv2.line(img, mid, tip, SKIN, int(w * 0.85), cv2.LINE_AA)
    cv2.circle(img, mid, int(w * 0.4), SHADOW, 2, cv2.LINE_AA)
    cv2.circle(img, tip, int(w * 0.42), SKIN, -1, cv2.LINE_AA)


def make_icon(gesture, size=160):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy, scale, w = size // 2, int(size * 0.65), size / 300.0, max(4, int(20 * (size / 300.0)))
    pw, ph = int(65 * scale), int(75 * scale)

    cv2.ellipse(img, (cx, cy), (pw, ph), 0, 0, 360, SKIN, -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (pw, ph), 0, 0, 360, SHADOW, 2, cv2.LINE_AA)
    draw_finger(img, (int(cx - pw * 0.85), int(cy + ph * 0.15)), -55, 55 * scale, int(w * 0.9))

    base_y = cy - ph + int(8 * scale)
    if gesture == "rock":
        for dx in [-1.3, -0.4, 0.5, 1.4]:
            bx = int(cx + dx * pw * 0.35)
            cv2.circle(img, (bx, base_y), int(w * 0.6), SKIN, -1, cv2.LINE_AA)
    elif gesture == "paper":
        for a, l, dx in zip([-18, -6, 6, 18], [95, 110, 105, 85], [-1.2, -0.4, 0.45, 1.25]):
            draw_finger(img, (int(cx + dx * pw * 0.3), base_y), a, l * scale, w)
    else:
        draw_finger(img, (int(cx - 0.35 * pw), base_y), -12, 120 * scale, w)
        draw_finger(img, (int(cx + 0.1 * pw), base_y), 10, 125 * scale, w)
        for dx in [0.9, 1.5]:
            cv2.circle(img, (int(cx + dx * pw * 0.35), base_y + 5), int(w * 0.5), SKIN, -1, cv2.LINE_AA)
    return img


def overlay(bg, fg, x, y):
    h, w = bg.shape[:2]
    ih, iw = fg.shape[:2]
    if x >= w or y >= h or x + iw <= 0 or y + ih <= 0: return
    xs, ys, xe, ye = max(x, 0), max(y, 0), min(x + iw, w), min(y + ih, h)
    crop = fg[ys - y:ys - y + (ye - ys), xs - x:xs - x + (xe - xs)]
    mask = np.any(crop > 10, axis=2)
    bg[ys:ye, xs:xe][mask] = crop[mask]


def draw_text(img, text, pos, scale=1.0, col=TEXT_C, thk=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thk + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thk, cv2.LINE_AA)


icons_bob = {g: make_icon(g) for g in ["rock", "paper", "scissors"]}
icons_jerry = {g: cv2.flip(icons_bob[g], 1) for g in icons_bob}

# --- Mouse Click Event Handling for Buttons ---
manual_choice = None


def click_event(event, x, y, flags, param):
    global manual_choice
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check click bounds for bottom control buttons (y: 650 to 710)
        if 650 <= y <= 710:
            if 360 <= x <= 480:
                manual_choice = "rock"
            elif 500 <= x <= 620:
                manual_choice = "paper"
            elif 640 <= x <= 760:
                manual_choice = "scissors"


# --- Main Engine ---
cap = cv2.VideoCapture(0)
# Request maximum possible hardware resolution for crisp quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow("RPS Retro Royale", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("RPS Retro Royale", click_event)

state, countdown, scores = "start", 3, {"Player": 0, "Bob": 0, "Jerry": 0}
active = {"Player": True, "Bob": True, "Jerry": True}
choices, msg, champ = {}, "", None

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    draw_3d_beach(canvas, time.time())

    lms = None
    if active["Player"]:
        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if res.hand_landmarks:
            lms = res.hand_landmarks[0]
            for lm in lms: cv2.circle(frame, (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])), 5, ACCENT, -1,
                                      cv2.LINE_AA)

    # Scaling video feed cleanly to max bounds without distortion
    fh, fw = frame.shape[:2]
    scale = min(960 / fw, 540 / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    feed = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    ox = 160 + (960 - nw) // 2
    oy = 180 + (540 - nh) // 2


    def draw_bot(name, x, y, act, icon=None):
        cv2.rectangle(canvas, (x, y), (x + 320, y + 180), (25, 25, 35), -1)
        cv2.rectangle(canvas, (x, y), (x + 320, y + 180), (100, 100, 150), 2)
        draw_text(canvas, name, (x + 15, y + 35), 0.7)
        if not act:
            draw_text(canvas, "OUT", (x + 120, y + 110), 1.2, (50, 50, 255), 3)
        elif icon is not None:
            overlay(canvas, icon, x + (320 - icon.shape[1]) // 2, y + 20)


    draw_bot("Bob (Bot 1)", 0, 0, active["Bob"], icons_bob.get(choices.get("Bob")) if state == "res" else None)
    draw_bot("Jerry (Bot 2)", 960, 0, active["Jerry"],
             icons_jerry.get(choices.get("Jerry")) if state == "res" else None)

    # Scoreboard Header
    cv2.rectangle(canvas, (320, 0), (960, 180), (15, 15, 20), -1)
    cv2.rectangle(canvas, (320, 0), (960, 180), (100, 100, 150), 2)
    draw_text(canvas, "SCOREBOARD", (550, 40), 0.9, ACCENT)
    score_txt = f"You: {scores['Player']}  |  Bob: {scores['Bob']}  |  Jerry: {scores['Jerry']}"
    draw_text(canvas, score_txt, (380, 100), 0.8)
    draw_text(canvas, "[W] Next  |  [R] Reset  |  [Q] Quit", (450, 150), 0.6, (150, 150, 150))

    # Player Feed Panel
    if active["Player"]:
        canvas[oy:oy + nh, ox:ox + nw] = feed
    else:
        gray = cv2.cvtColor(cv2.cvtColor(feed, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        canvas[oy:oy + nh, ox:ox + nw] = gray

    cv2.rectangle(canvas, (ox, oy), (ox + nw, oy + nh), ACCENT, 3)
    status_label = f"You [{choices.get('Player', '')}]" if state == "res" else "You (Webcam & Buttons)"
    draw_text(canvas, status_label, (ox + 10, oy + 35), 0.8)
    if not active["Player"]:
        draw_text(canvas, "ELIMINATED", (ox + nw // 2 - 120, oy + nh // 2), 1.5, (50, 50, 255), 4)

    # --- Bottom Interactive Choice Bar (Clickable Buttons) ---
    cv2.rectangle(canvas, (350, 640), (770, 710), (30, 30, 45), -1)
    cv2.rectangle(canvas, (350, 640), (770, 710), (120, 120, 180), 2)

    # Render Button Options
    buttons = [("ROCK", 360), ("PAPER", 500), ("SCISSORS", 640)]
    for name, bx in buttons:
        is_selected = (manual_choice == name.lower())
        btn_color = (80, 180, 80) if is_selected else (50, 50, 70)
        cv2.rectangle(canvas, (bx, 650), (bx + 120, 700), btn_color, -1)
        cv2.rectangle(canvas, (bx, 650), (bx + 120, 700), (200, 200, 200), 1)
        draw_text(canvas, name, (bx + 12, 683), 0.55, TEXT_C, 1)

    if manual_choice:
        draw_text(canvas, f"Locked: {manual_choice.upper()}", (790, 678), 0.7, (100, 255, 100), 2)

    # Game Flow States
    if state == "start":
        draw_text(canvas, "Press 'W' or Click Button to Start", (420, 400), 1.0, ACCENT, 2)

    elif state == "count":
        cnt = 3 - int(time.time() - c_time)
        if cnt > 0:
            draw_text(canvas, str(cnt), (600, 450), 5.0, (50, 255, 255), 8)
        else:
            # Determine player choice: prioritize manual click selection, fallback to hand tracking
            p_choice = manual_choice
            if not p_choice and lms:
                p_choice = detect_gesture(lms)
            if not p_choice:
                p_choice = "rock"  # ultimate fallback default

            choices = {
                "Player": p_choice if active["Player"] else None,
                "Bob": random.choice(["rock", "paper", "scissors"]) if active["Bob"] else None,
                "Jerry": random.choice(["rock", "paper", "scissors"]) if active["Jerry"] else None
            }
            choices = {k: v for k, v in choices.items() if v}
            msg = eval_round(active, choices)
            champ = [p for p, a in active.items() if a][0] if sum(active.values()) == 1 else None
            if champ: scores[champ] += 1
            state = "res"

    elif state == "res":
        if champ:
            canvas = cv2.GaussianBlur(canvas, (45, 45), 0)
            draw_text(canvas, f"{champ.upper()} WINS MATCH!", (360, 450), 2.0, ACCENT, 6)
            draw_text(canvas, "Press 'R' to Reset Match", (450, 520), 0.9, TEXT_C, 2)
        else:
            draw_text(canvas, msg, (ox + 20, oy + nh - 30), 0.9, (0, 255, 255), 2)

    cv2.imshow("RPS Retro Royale", canvas)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    if k == ord('r'):
        scores = {k: 0 for k in scores}
        active = {k: True for k in active}
        manual_choice = None
        state = "start"
    if k == ord('w'):
        if champ:
            active = {k: True for k in active}
            champ = None
        state, c_time = "count", time.time()

cap.release()
cv2.destroyAllWindows()
