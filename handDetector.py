"""
Handy — Hand Gesture Cursor Control
====================================
Threading model:
  Main thread   → camera capture + MediaPipe inference + display  (~30 fps)
  Cursor thread → fixed 60 fps loop; One-Euro filter + interpolation → move / click

  The cursor thread runs at 60 fps regardless of camera/inference rate.
  When new inference data (30 fps) arrives, the One-Euro filtered target is
  updated.  Each 60 fps tick the cursor interpolates toward that target,
  making motion smooth even between inference frames.

Smoothing:
  One-Euro Filter (Casiez, Roussel, Vogel — CHI 2012):
    - Slow / still hand → heavy smoothing   (low jitter)
    - Fast motion       → minimal smoothing (low lag)
  Fixed EMA was replaced because it trades off jitter vs. lag globally;
  One-Euro adapts per-frame based on measured cursor speed.

Gestures:
  Hand visible, fingers open  → cursor tracks index finger tip
  Pinch (thumb ↔ index close) → left mouse button held (drag or click)
  Pinch released              → left mouse button released
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import math
import argparse

import pyautogui
pyautogui.PAUSE = 0  # Disable PyAutoGUI's built-in 0.1 s inter-call delay

# ── macOS low-latency cursor backend (Quartz CoreGraphics) ────────────────────
# Requires: pip install pyobjc-framework-Quartz
# NOTE: macOS will prompt for Accessibility permission on first run.
try:
    from Quartz.CoreGraphics import (
        CGEventCreateMouseEvent, CGEventPost,
        kCGEventMouseMoved, kCGEventLeftMouseDragged,
        kCGEventLeftMouseDown, kCGEventLeftMouseUp,
        kCGHIDEventTap, CGPointMake, kCGMouseButtonLeft,
    )
    QUARTZ_AVAILABLE = True
    print("[Info] Quartz CoreGraphics available — using low-latency cursor control")
except ImportError:
    # Stub out Quartz symbols — all usages are gated by QUARTZ_AVAILABLE
    CGEventCreateMouseEvent = CGEventPost = CGPointMake = None
    kCGEventMouseMoved = kCGEventLeftMouseDragged = None
    kCGEventLeftMouseDown = kCGEventLeftMouseUp = None
    kCGHIDEventTap = kCGMouseButtonLeft = None
    QUARTZ_AVAILABLE = False
    print("[Info] Quartz not available — falling back to PyAutoGUI")

# ── Tunable parameters (override with CLI flags) ──────────────────────────────
DEFAULTS = dict(
    camera=0,              # Camera index (0 = built-in, 2 = phone virtual cam)
    cam_width=640,
    cam_height=480,
    cam_fps=30,
    cam_margin=0.15,       # Ignore outer 15 % of frame to reduce edge jitter
    cursor_fps=60,         # Cursor thread rate — independent of camera FPS
    min_cutoff=1.0,        # One-Euro: smoothing at rest (Hz). Lower = smoother.
    speed_coef=0.007,      # One-Euro: speed factor β. Higher = less lag on fast moves.
    interp_alpha=0.6,      # 60fps interpolation toward One-Euro target (0=glide, 1=instant)
    lock_threshold=0.10,   # Pre-pinch cursor lock zone (must be > pinch_threshold).
                           # Cursor freezes when thumb-index distance drops below this
                           # so the click lands exactly where you aimed, not where your
                           # index finger drifted to during the pinch gesture.
    pinch_threshold=0.05,  # Normalised Euclidean distance for pinch detection
    detect_conf=0.7,       # MediaPipe detection confidence
    track_conf=0.7,        # MediaPipe tracking confidence
)


# ── One-Euro Filter ───────────────────────────────────────────────────────────

class OneEuroFilter:
    """
    Adaptive low-pass filter for noisy interactive input.
    Reference: Casiez, Roussel, Vogel — CHI 2012.
    https://cristal.univ-lille.fr/~casiez/1euro/

    Unlike a fixed EMA, this filter estimates the current speed of the signal
    and adjusts its cutoff frequency accordingly:
      - Signal is slow / still → low cutoff → heavy smoothing  → kills jitter
      - Signal is moving fast  → high cutoff → light smoothing → kills lag

    Args:
        freq:       Initial sampling frequency estimate in Hz.
        min_cutoff: Minimum cutoff frequency (Hz).
                    Lower = smoother when still. Good range: 0.5 – 3.0.
        beta:       Speed coefficient.
                    Higher = less lag during fast movement. Good range: 0.001 – 0.1.
        d_cutoff:   Derivative (speed estimate) filter cutoff. Rarely changed.
    """

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0,
                 beta: float = 0.007, d_cutoff: float = 1.0):
        self._freq       = freq
        self._min_cutoff = min_cutoff
        self._beta       = beta
        self._d_cutoff   = d_cutoff
        self._x_prev:  float | None = None
        self._dx_prev: float        = 0.0
        self._t_prev:  float | None = None

    @staticmethod
    def _alpha(freq: float, cutoff: float) -> float:
        """Compute EMA alpha for a first-order low-pass at given freq & cutoff."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te  = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, t: float) -> float:
        """Return the filtered value of x sampled at time t (seconds)."""
        # Auto-estimate sampling frequency from consecutive timestamps
        if self._t_prev is not None and t > self._t_prev:
            self._freq = 1.0 / (t - self._t_prev)
        self._t_prev = t

        if self._x_prev is None:          # First sample — no history yet
            self._x_prev = x
            return x

        # Step 1: estimate instantaneous speed via filtered derivative
        dx     = (x - self._x_prev) * self._freq
        a_d    = self._alpha(self._freq, self._d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Step 2: raise cutoff proportionally to speed (reduces lag when moving)
        cutoff = self._min_cutoff + self._beta * abs(dx_hat)

        # Step 3: apply main low-pass filter with adaptive cutoff
        a     = self._alpha(self._freq, cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev  = x_hat
        self._dx_prev = dx_hat
        return x_hat


# ── Thread-safe shared state ──────────────────────────────────────────────────

class HandState:
    """
    Passed by reference between the main (inference) thread and the cursor
    thread.  A Lock protects the data fields; an Event wakes the cursor thread
    whenever new landmarks arrive, avoiding the need for any sleep() call.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.lmlist: list = []        # [(id, norm_x, norm_y), ...]
        self.is_pinching: bool = False
        self.new_data = threading.Event()

    def update(self, lmlist: list, is_pinching: bool) -> None:
        """Called by the main thread after every inference frame."""
        with self._lock:
            self.lmlist = lmlist
            self.is_pinching = is_pinching
        self.new_data.set()   # Wake the cursor thread

    def consume(self) -> tuple:
        """
        Called by the cursor thread.  Clears the event flag and returns a
        snapshot of the current state.
        """
        self.new_data.clear()
        with self._lock:
            return list(self.lmlist), self.is_pinching


# ── MediaPipe hand detector ───────────────────────────────────────────────────

class HandDetector:
    """Thin MediaPipe Hands wrapper that returns normalised landmarks."""

    def __init__(self, maxHands: int = 1,
                 detectionCon: float = 0.7,
                 trackCon: float = 0.7):
        self._mp_hands = mp.solutions.hands
        # Use keyword args for MediaPipe 0.10.x compatibility.
        # BUG FIX: detectionCon kept as float — int(0.5) == 0 broke confidence.
        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=maxHands,
            min_detection_confidence=float(detectionCon),
            min_tracking_confidence=float(trackCon),
        )
        self._draw_utils = mp.solutions.drawing_utils
        self.results = None

    def find_hands(self, img: np.ndarray, draw: bool = True) -> np.ndarray:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks and draw:
            for hand_lms in self.results.multi_hand_landmarks:
                self._draw_utils.draw_landmarks(
                    img, hand_lms, self._mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, hand_no: int = 0) -> list:
        """
        Returns normalised landmarks for the requested hand:
            [(id, norm_x, norm_y), ...]   — values in [0, 1]

        Storing normalised coords (not pixel coords) makes downstream logic
        resolution-independent and eliminates coordinate-mapping bugs.
        """
        lmlist = []
        if self.results and self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                for lm_id, lm in enumerate(
                        self.results.multi_hand_landmarks[hand_no].landmark):
                    lmlist.append([lm_id, lm.x, lm.y])
        return lmlist


# ── Pinch detection ───────────────────────────────────────────────────────────

def check_pinch(lmlist: list, threshold: float = DEFAULTS["pinch_threshold"]) -> bool:
    """
    Scale-invariant pinch detection via normalised Euclidean distance between
    thumb tip (4) and index finger tip (8).
    """
    if len(lmlist) < 9:
        return False
    _, tx, ty = lmlist[4]   # thumb tip
    _, ix, iy = lmlist[8]   # index finger tip
    return math.hypot(tx - ix, ty - iy) < threshold


def is_pointer_mode(lmlist: list) -> bool:
    """
    Detects the 'pointer' hand pose: index finger clearly extended,
    middle / ring / pinky fingers folded in.  Thumb is unconstrained
    so it can move freely to perform pinch gestures while pointing.

    Without this guard the cursor moves whenever the hand is in frame,
    which makes every accidental gesture a cursor event.  Pointer mode
    means you have to consciously 'activate' cursor control — much more
    like how Vision Pro requires an explicit raised-finger pose.

    Detection: compare each fingertip y-coord to its PIP (second joint).
    In normalised MediaPipe coordinates y increases downward, so:
      tip.y  < pip.y  →  finger is extended (tip is higher in image)
      tip.y  > pip.y  →  finger is folded   (tip has curled below second joint)

    Landmarks used:
      Index : tip=8,  pip=6
      Middle: tip=12, pip=10
      Ring  : tip=16, pip=14
      Pinky : tip=20, pip=18
    """
    if len(lmlist) < 21:
        return False

    index_up  = lmlist[8][2]  < lmlist[6][2]   # index tip above index PIP
    middle_dn = lmlist[12][2] > lmlist[10][2]  # middle tip below middle PIP
    ring_dn   = lmlist[16][2] > lmlist[14][2]  # ring tip below ring PIP
    pinky_dn  = lmlist[20][2] > lmlist[18][2]  # pinky tip below pinky PIP

    return index_up and middle_dn and ring_dn and pinky_dn


# ── Cursor controller ─────────────────────────────────────────────────────────

class CursorController:
    """
    Two-stage cursor pipeline:

      Stage 1 — set_target()  (called ~30 fps, when new inference data arrives)
        raw landmark  →  coordinate remap  →  One-Euro filter  →  filtered target
        Pre-pinch lock: if thumb-index distance is in (pinch_threshold, lock_threshold)
        and not currently dragging, the cursor position is FROZEN so the click
        always lands where you aimed, not where your index drifted during the gesture.

      Stage 2 — tick()  (called at 60 fps by cursor thread)
        short EMA glide from current pos toward filtered target
        →  eliminates the discrete 33 ms position jumps visible at 30 fps
        →  dispatches Quartz / PyAutoGUI cursor event

    Pinch transitions (mouseDown / mouseUp) are driven from tick() so they
    fire at 60 fps resolution regardless of when inference completes.

    Pre-pinch lock state machine (all evaluated in cursor thread — no extra lock needed):
      OPEN       distance > lock_threshold          → cursor tracks index freely
      PRE-PINCH  lock_threshold ≥ dist > pinch_thr → cursor FROZEN, waiting for pinch
      PINCHED    distance ≤ pinch_threshold         → mouseDown, cursor unfreezes (drag)
      RELEASING  was pinched, now open              → mouseUp, back to OPEN
    """

    def __init__(self, cam_margin: float,
                 min_cutoff: float   = 1.0,
                 speed_coef: float   = 0.007,
                 interp_alpha: float = 0.6,
                 lock_threshold: float = 0.10):
        self.cam_margin      = cam_margin
        self.interp_alpha    = interp_alpha
        self._lock_threshold = lock_threshold
        self.screen_w, self.screen_h = pyautogui.size()

        # One-Euro Filters — independent instances for X and Y
        self._oef_x = OneEuroFilter(freq=30.0, min_cutoff=min_cutoff, beta=speed_coef)
        self._oef_y = OneEuroFilter(freq=30.0, min_cutoff=min_cutoff, beta=speed_coef)

        # Filtered target — written by set_target, read by tick (lock-protected)
        self._target_x:  float = self.screen_w / 2
        self._target_y:  float = self.screen_h / 2
        self._pinching:  bool  = False
        self._tgt_lock = threading.Lock()

        # Cursor interpolation state — only accessed from cursor thread
        self._smooth_x:   float = self.screen_w / 2
        self._smooth_y:   float = self.screen_h / 2
        self._pinch_held: bool  = False

    # ── Stage 1: called at inference rate (~30 fps) ───────────────────────────

    def set_target(self, lmlist: list, pinching: bool) -> None:
        """
        Applies One-Euro filter to the new landmark position and updates the
        shared target — unless the pre-pinch lock is active.

        Pre-pinch lock logic (runs entirely in cursor thread, no extra lock needed):
          - If thumb-index distance is in the zone (pinch_threshold, lock_threshold)
            AND the user is not currently dragging (_pinch_held=False)
            AND not yet fully pinched (pinching=False)
            → freeze the cursor so it doesn't drift during the pinch gesture.
          - Once fully pinched (_pinch_held becomes True in tick()), cursor unfreezes
            so the user can drag freely.
        """
        if not lmlist:
            # Hand left frame — release any held click; cursor stays put
            with self._tgt_lock:
                self._pinching = False
            return

        _, norm_x, norm_y = lmlist[8]   # index finger tip
        _, tnx,   tny     = lmlist[4]   # thumb tip
        t = time.perf_counter()

        # Thumb-index normalised distance (same metric as check_pinch)
        dist = math.hypot(norm_x - tnx, norm_y - tny)

        # ── Pre-pinch cursor lock ──────────────────────────────────────────────
        # Freeze cursor when hand enters the approach zone but is not yet pinched
        # and is not currently in a drag.  This ensures the click lands exactly
        # where you aimed before your index finger drifted downward.
        in_lock_zone = (dist < self._lock_threshold   # approaching pinch
                        and not pinching               # not yet clicked
                        and not self._pinch_held)      # not mid-drag

        if in_lock_zone:
            # Cursor position is frozen — only propagate the pinch flag
            with self._tgt_lock:
                self._pinching = pinching
            return

        # ── Normal: update One-Euro filtered target ────────────────────────────
        # Mirror X (webcam is front-facing) + remap [margin, 1-margin] → screen
        raw_x = np.interp(1.0 - norm_x,
                          [self.cam_margin, 1.0 - self.cam_margin],
                          [0, self.screen_w])
        raw_y = np.interp(norm_y,
                          [self.cam_margin, 1.0 - self.cam_margin],
                          [0, self.screen_h])

        # One-Euro filter — adaptive per measured speed
        fx = self._oef_x.filter(raw_x, t)
        fy = self._oef_y.filter(raw_y, t)

        with self._tgt_lock:
            self._target_x = fx
            self._target_y = fy
            self._pinching = pinching

    # ── Stage 2: called at cursor rate (60 fps) ───────────────────────────────

    def tick(self) -> None:
        """
        Interpolates the cursor toward the One-Euro filtered target and posts
        the OS cursor event.  Running at 60 fps means the cursor glides smoothly
        to each new 30 fps target position instead of jumping discretely.
        """
        with self._tgt_lock:
            tx       = self._target_x
            ty       = self._target_y
            pinching = self._pinching

        # Short-range EMA glide: covers `interp_alpha` of remaining gap each tick.
        # With alpha=0.6 at 60 fps: 60 % at tick 1, 84 % at tick 2 (~33 ms).
        a = self.interp_alpha
        self._smooth_x += (tx - self._smooth_x) * a
        self._smooth_y += (ty - self._smooth_y) * a

        cx = int(np.clip(self._smooth_x, 0, self.screen_w - 1))
        cy = int(np.clip(self._smooth_y, 0, self.screen_h - 1))

        # Click state machine
        if pinching and not self._pinch_held:
            self._mouse_down(cx, cy)
            self._pinch_held = True
        elif not pinching and self._pinch_held:
            self._mouse_up(cx, cy)
            self._pinch_held = False

        self._move(cx, cy)

    # ── OS event dispatch ─────────────────────────────────────────────────────

    def _move(self, x: int, y: int) -> None:
        if QUARTZ_AVAILABLE:
            event_type = (kCGEventLeftMouseDragged
                          if self._pinch_held else kCGEventMouseMoved)
            evt = CGEventCreateMouseEvent(
                None, event_type, CGPointMake(x, y), kCGMouseButtonLeft)
            CGEventPost(kCGHIDEventTap, evt)
        else:
            pyautogui.moveTo(x, y, _pause=False)

    def _mouse_down(self, x: int, y: int) -> None:
        if QUARTZ_AVAILABLE:
            evt = CGEventCreateMouseEvent(
                None, kCGEventLeftMouseDown, CGPointMake(x, y), kCGMouseButtonLeft)
            CGEventPost(kCGHIDEventTap, evt)
        else:
            pyautogui.mouseDown(x, y, _pause=False)

    def _mouse_up(self, x: int, y: int) -> None:
        if QUARTZ_AVAILABLE:
            evt = CGEventCreateMouseEvent(
                None, kCGEventLeftMouseUp, CGPointMake(x, y), kCGMouseButtonLeft)
            CGEventPost(kCGHIDEventTap, evt)
        else:
            pyautogui.mouseUp(x, y, _pause=False)


# ── Cursor thread ─────────────────────────────────────────────────────────────

def cursor_worker(hand_state: HandState,
                  stop_event: threading.Event,
                  controller: CursorController,
                  cursor_fps: int = 60) -> None:
    """
    Fixed-rate cursor loop running at `cursor_fps` Hz (default 60).

    Uses time.perf_counter() for precise frame timing.  On each tick it:
      1. Non-blockingly checks whether new inference data has arrived.
         If so, calls controller.set_target() to update the One-Euro target.
      2. Always calls controller.tick() to interpolate toward the target
         and post the cursor event — regardless of whether new data arrived.

    This means the cursor moves continuously at 60 fps even though MediaPipe
    only delivers new landmarks at 30 fps.
    """
    interval = 1.0 / cursor_fps
    print(f"[CursorThread] Started at {cursor_fps} fps")

    while not stop_event.is_set():
        tick_start = time.perf_counter()

        # Non-blocking: grab latest inference data if the main thread posted any
        if hand_state.new_data.is_set():
            lmlist, pinching = hand_state.consume()
            controller.set_target(lmlist, pinching)

        # Advance cursor at full cursor_fps rate
        controller.tick()

        # Sleep for the remainder of this 60fps frame
        elapsed    = time.perf_counter() - tick_start
        sleep_time = interval - elapsed
        if sleep_time > 0.001:   # Skip sub-millisecond sleeps (high OS overhead)
            time.sleep(sleep_time)

    print("[CursorThread] Stopped")


# ── Main / camera loop ────────────────────────────────────────────────────────

def main(args) -> None:
    print("[Main] Starting Handy — Hand Gesture Cursor Control")
    print("[Main] Press 'q' in the camera window to quit\n")

    # ── Camera setup ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        fallback = 2 if args.camera != 2 else 0
        print(f"[Main] Camera {args.camera} unavailable — trying {fallback}...")
        cap = cv2.VideoCapture(fallback)
    if not cap.isOpened():
        print("[Main] ERROR: No camera found. "
              "Run checkForAvailableCameras.py to list available indices.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    cap.set(cv2.CAP_PROP_FPS,          args.cam_fps)

    # ── Detector, shared state & cursor controller ────────────────────────────
    detector   = HandDetector(maxHands=1,
                              detectionCon=args.detect_conf,
                              trackCon=args.track_conf)
    hand_state = HandState()
    controller = CursorController(
        cam_margin=args.cam_margin,
        min_cutoff=args.min_cutoff,
        speed_coef=args.speed_coef,
        interp_alpha=args.interp_alpha,
        lock_threshold=args.lock_threshold,
    )
    stop_event = threading.Event()

    # ── Cursor thread ─────────────────────────────────────────────────────────
    c_thread = threading.Thread(
        target=cursor_worker,
        args=(hand_state, stop_event, controller, args.cursor_fps),
        name="CursorThread",
        daemon=True,   # Killed automatically if main thread exits unexpectedly
    )
    c_thread.start()

    # ── Inference loop ────────────────────────────────────────────────────────
    p_time = time.time()

    while True:
        ok, img = cap.read()
        if not ok:
            print("[Main] Camera read failed — retrying...")
            continue

        img    = detector.find_hands(img, draw=True)
        lmlist = detector.find_position(hand_no=0)

        # ── Pointer mode gate ──────────────────────────────────────────────────
        # Only activate cursor control when the user is explicitly pointing:
        # index finger up, middle/ring/pinky folded.  Pass an empty lmlist
        # when not pointing so the cursor freezes and any held click releases.
        pointer_active = is_pointer_mode(lmlist) if lmlist else False
        pinching       = (check_pinch(lmlist, threshold=args.pinch_threshold)
                          if pointer_active else False)

        hand_state.update(
            lmlist if pointer_active else [],
            pinching,
        )

        if lmlist:
            # Visual state indicator
            _, tnx, tny = lmlist[4]
            _, inx, iny = lmlist[8]
            dist = math.hypot(inx - tnx, iny - tny)

            if not pointer_active:
                label, color = "IDLE",     (160, 160, 160)     # grey  — hand seen, not pointing
            elif pinching:
                label, color = "PINCHING", (0, 255, 0)         # green — click / drag
            elif dist < args.lock_threshold:
                label, color = "LOCKED",   (0, 165, 255)       # orange — cursor frozen, aim confirmed
            else:
                label, color = "TRACKING", (255, 0, 255)       # purple — cursor following index tip

            cv2.putText(img, label, (10, 120),
                        cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        # FPS counter
        c_time = time.time()
        fps    = 1.0 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Handy", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    print("\n[Main] Shutting down...")
    stop_event.set()
    c_thread.join(timeout=2.0)
    cap.release()
    cv2.destroyAllWindows()
    print("[Main] Done.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Handy — control your cursor with hand gestures")
    parser.add_argument("--camera",          type=int,   default=DEFAULTS["camera"])
    parser.add_argument("--cam-width",       type=int,   default=DEFAULTS["cam_width"],    dest="cam_width")
    parser.add_argument("--cam-height",      type=int,   default=DEFAULTS["cam_height"],   dest="cam_height")
    parser.add_argument("--cam-fps",         type=int,   default=DEFAULTS["cam_fps"],      dest="cam_fps")
    parser.add_argument("--cursor-fps",      type=int,   default=DEFAULTS["cursor_fps"],   dest="cursor_fps",
                        help="Cursor update rate in Hz (default: 60). Independent of camera fps.")
    parser.add_argument("--cam-margin",      type=float, default=DEFAULTS["cam_margin"],   dest="cam_margin",
                        help="Edge dead-zone fraction (default: 0.15). Lower if hard to reach screen edges.")
    parser.add_argument("--min-cutoff",      type=float, default=DEFAULTS["min_cutoff"],   dest="min_cutoff",
                        help="One-Euro smoothing at rest in Hz (default: 1.0). Lower = smoother / less jitter.")
    parser.add_argument("--speed-coef",      type=float, default=DEFAULTS["speed_coef"],   dest="speed_coef",
                        help="One-Euro speed factor β (default: 0.007). Higher = less lag on fast moves.")
    parser.add_argument("--interp-alpha",    type=float, default=DEFAULTS["interp_alpha"], dest="interp_alpha",
                        help="60fps→30fps glide alpha (default: 0.6). Higher = snappier, lower = smoother glide.")
    parser.add_argument("--lock-threshold",  type=float, default=DEFAULTS["lock_threshold"], dest="lock_threshold",
                        help="Pre-pinch cursor lock zone (default: 0.10). Cursor freezes when thumb-index "
                             "distance drops below this so clicks land where you aimed. Must be > pinch-threshold.")
    parser.add_argument("--pinch-threshold", type=float, default=DEFAULTS["pinch_threshold"], dest="pinch_threshold",
                        help="Normalised pinch distance (default: 0.05). Lower = tighter pinch required.")
    parser.add_argument("--detect-conf",     type=float, default=DEFAULTS["detect_conf"],  dest="detect_conf")
    parser.add_argument("--track-conf",      type=float, default=DEFAULTS["track_conf"],   dest="track_conf")

    main(parser.parse_args())
