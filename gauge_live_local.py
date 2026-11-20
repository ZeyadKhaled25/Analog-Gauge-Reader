"""
Live analog-gauge reader using a locally trained YOLOv8 model (no cloud, no API key).

Model classes: center, gauge, max, min, tip

How the reading is produced
  1. The model finds the needle pivot (center), the needle tip and the 'min'/'max' marks.
  2. The needle's position between the min and max marks gives a fraction of the scale (0-1).
  3. The numbers printed at the min and max marks are read with OCR (EasyOCR, in a background
     thread) and voted over several frames, which turns the fraction into a real value
     (e.g. 4.2 on a 0-7 gauge).
  If OCR is unavailable or unsure, pass --vmin/--vmax yourself; with neither it shows % of scale.

Saving results (CSV, opens in Excel):
  Click the video window, then press
      l  save the current reading            n  start the next inspection round
      d  next device number                  a  previous device number
      q  quit
  Each saved row: date, time, round, device, reading, unit, scale, status (OK / ALARM).

Usage:
    python gauge_live_local.py                                   # scale read automatically (OCR)
    python gauge_live_local.py --vmin 0 --vmax 7 --unit bar      # manual scale (overrides OCR)
    python gauge_live_local.py --device 1 --round 2              # start at Device 1, round 2
    python gauge_live_local.py --auto-log 10                     # also save a row every 10 s
    python gauge_live_local.py --debug-ocr                       # print/save what OCR sees
    python gauge_live_local.py rtsp://ip/stream --ccw --alarm-high 15
"""
import argparse
import collections
import csv
import math
import os
import re
import statistics
import threading
import time

import cv2
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "runs", "gauge", "weights", "best.pt")
DEFAULT_LOG = os.path.join(HERE, "gauge_readings.csv")
DEBUG_DIR = os.path.join(HERE, "ocr_debug")
SMOOTHING_WINDOW = 5
ALARM_COOLDOWN = 1.0
MIN_SWEEP_DEG = 30   # min->max must span at least this much, otherwise detections are unreliable

try:
    import winsound

    def _beep():
        winsound.Beep(700, 150)
except ImportError:
    def _beep():
        print("\a", end="", flush=True)


class CameraStream:
    """Grabs frames in a background thread and keeps only the newest one."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera/stream: {source}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self.lock:
                self.frame = frame
                self.frame_id += 1

    def read(self):
        with self.lock:
            if self.frame is None:
                return None, 0
            return self.frame.copy(), self.frame_id

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_number(text):
    m = _NUM.search(text.replace(" ", ""))
    return float(m.group()) if m else None


class ScaleReader:
    """Reads the numbers printed at the gauge's min and max marks.
    OCR runs in its own thread so the live reading never stalls. The last few results per mark
    are kept and the most common one is used, so a single misread doesn't change the scale."""

    WINDOW = 6

    def __init__(self, debug=False):
        import easyocr   # raises ImportError if not installed
        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        self.debug = debug
        self.debug_saved = 0
        self.window = {k: collections.deque(maxlen=self.WINDOW) for k in ("min", "max")}
        self.last_try = {"min": 0.0, "max": 0.0}
        self._job = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def reset(self):
        for w in self.window.values():
            w.clear()

    def stop(self):
        self.running = False
        self._wake.set()
        self.thread.join(timeout=2)

    def _top(self, k):
        w = list(self.window[k])
        if not w:
            return None, 0
        cnt = collections.Counter(w)
        best = max(cnt, key=lambda v: (cnt[v], max(i for i, x in enumerate(w) if x == v)))
        return best, cnt[best]

    def value(self, k):
        v, n = self._top(k)
        return v if n >= 2 else None

    def update(self, frame, best):
        """Called every frame; hands the latest frame to the OCR thread and returns at once."""
        marks = {k: dict(best[k]) for k in ("min", "max")
                 if k in best and not best[k].get("remembered")}   # only marks really detected
        if not marks:
            return
        with self._lock:
            self._job = (frame, marks)
        self._wake.set()

    def _worker(self):
        while self.running:
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                continue
            frame, marks = job
            for k, d in marks.items():
                _, n = self._top(k)
                interval = 1.0 if n >= 4 else 0.4        # re-check slowly once confident
                if time.time() - self.last_try[k] < interval:
                    continue
                self.last_try[k] = time.time()
                try:
                    num = self._read(frame, d)
                except Exception as e:
                    if self.debug:
                        print(f"[OCR] error: {e}")
                    continue
                if num is not None:
                    self.window[k].append(num)

    def _read(self, frame, d):
        H, W = frame.shape[:2]
        px, py = d["w"] * 0.1 + 4, d["h"] * 0.1 + 4
        x1, x2 = int(max(d["x"] - d["w"] / 2 - px, 0)), int(min(d["x"] + d["w"] / 2 + px, W))
        y1, y2 = int(max(d["y"] - d["h"] / 2 - py, 0)), int(min(d["y"] + d["h"] / 2 + py, H))
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            return None
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        s = 64.0 / g.shape[0]
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        g = cv2.copyMakeBorder(g, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

        best_val, best_conf, raw = None, 0.0, []
        for img in (g, 255 - g):                          # dark-on-light and light-on-dark dials
            cands = []
            try:    # detect + read: separates the number from tick marks around it
                cands += self.reader.readtext(img, allowlist="0123456789.-", detail=1,
                                              min_size=10, text_threshold=0.5, low_text=0.3)
            except Exception:
                pass
            try:    # read the whole crop as one text line
                cands += self.reader.recognize(img, allowlist="0123456789.-", detail=1)
            except Exception:
                pass
            for _, text, conf in cands:
                raw.append((text, round(float(conf), 2)))
                n = parse_number(text)
                if n is not None and conf > best_conf:
                    best_val, best_conf = n, float(conf)

        if self.debug:
            print(f"[OCR] {d['name']}: raw={raw} -> {best_val} (conf {best_conf:.2f})")
            if self.debug_saved < 40:
                os.makedirs(DEBUG_DIR, exist_ok=True)
                cv2.imwrite(os.path.join(DEBUG_DIR, f"{d['name']}_{self.debug_saved:02d}.png"), g)
                self.debug_saved += 1
        return best_val if best_conf >= 0.2 else None


class ReadingLogger:
    """Appends readings to a CSV file (UTF-8 with BOM so Excel shows symbols like °C correctly)."""

    HEADER = ["date", "time", "round", "device", "reading", "unit",
              "scale_min", "scale_max", "scale_source", "status"]

    def __init__(self, path):
        self.path = path

    def log(self, device, rnd, value, scale, unit, status):
        now = time.localtime()
        row = [time.strftime("%d/%m/%Y", now), time.strftime("%H:%M:%S", now), rnd, device,
               f"{value:.2f}", unit,
               f"{scale[0]:g}" if scale else "", f"{scale[1]:g}" if scale else "",
               scale[2] if scale else "none", status]
        new_file = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(self.HEADER)
            w.writerow(row)


def angle_deg(c, p, ccw):
    """Angle of point p around center c, 0-360. Image y points down, so atan2 grows clockwise on screen."""
    a = math.degrees(math.atan2(p[1] - c[1], p[0] - c[0]))
    if ccw:
        a = -a
    return a % 360


def compute_value(center, tip, mn, mx, vmin, vmax, ccw):
    a_tip = angle_deg(center, tip, ccw)
    a_min = angle_deg(center, mn, ccw)
    a_max = angle_deg(center, mx, ccw)
    sweep = (a_max - a_min) % 360
    if sweep < MIN_SWEEP_DEG:
        return None
    rel = (a_tip - a_min) % 360            # degrees travelled from min toward max
    if rel > sweep:                        # needle is in the dead zone outside the scale
        over = rel - sweep                 # past max
        under = 360 - rel                  # before min
        rel = -under if under < over else sweep + over
    return vmin + rel / sweep * (vmax - vmin)


def fmt(v, scale):
    rng = abs(scale[1] - scale[0]) if scale else 100
    return f"{v:.2f}" if rng <= 10 else f"{v:.1f}"


class Inferencer:
    def __init__(self, stream, model, args, scale_reader):
        self.stream, self.model, self.args = stream, model, args
        self.scale_reader = scale_reader
        self.lock = threading.Lock()
        self.result = None
        self.history = collections.deque(maxlen=SMOOTHING_WINDOW)   # needle position as fraction of scale
        self.last_alarm = 0.0
        self.marks = {}        # remembered min/max positions, relative to the needle pivot
        self.last_gw = None    # last seen gauge width in pixels
        self.last_center_t = time.time()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        last_id = 0
        a = self.args
        manual = a.vmin is not None and a.vmax is not None
        while self.running:
            frame, fid = self.stream.read()
            if frame is None or fid == last_id:
                time.sleep(0.005)
                continue
            last_id = fid
            t0 = time.time()
            r = self.model.predict(frame, conf=min(a.conf, a.conf_marks), imgsz=a.imgsz,
                                   device=a.device_id, verbose=False)[0]

            dets = []
            for b in r.boxes:
                x, y, w, h = b.xywh[0].tolist()
                name = self.model.names[int(b.cls[0])].lower()
                c = float(b.conf[0])
                if c < (a.conf_marks if name in ("min", "max") else a.conf):
                    continue                     # per-class confidence threshold
                dets.append({"name": name, "conf": c, "x": x, "y": y, "w": w, "h": h})
            best = {}
            for d in dets:                       # highest-confidence detection per class
                if d["name"] not in best or d["conf"] > best[d["name"]]["conf"]:
                    best[d["name"]] = d
            self._apply_mark_memory(best)

            # Needle position as a fraction of the scale (0 at the min mark, 1 at the max mark)
            frac = None
            if all(k in best for k in ("center", "tip", "min", "max")):
                pt = lambda k: (best[k]["x"], best[k]["y"])
                frac = compute_value(pt("center"), pt("tip"), pt("min"), pt("max"), 0.0, 1.0, a.ccw)

            # Scale: manual --vmin/--vmax wins, otherwise numbers read from the dial
            scale = None
            if manual:
                scale = (a.vmin, a.vmax, "manual")
            elif self.scale_reader is not None:
                self.scale_reader.update(frame, best)
                vmin, vmax = self.scale_reader.value("min"), self.scale_reader.value("max")
                if vmin is not None and vmax is not None and vmax > vmin:
                    scale = (vmin, vmax, "auto")

            value = None
            if frac is not None:
                self.history.append(frac)
                f = statistics.median(self.history)
                value = scale[0] + f * (scale[1] - scale[0]) if scale else f * 100.0
                low = a.alarm_low is not None and value <= a.alarm_low
                high = a.alarm_high is not None and value >= a.alarm_high
                if (low or high) and time.time() - self.last_alarm > ALARM_COOLDOWN:
                    self.last_alarm = time.time()
                    threading.Thread(target=_beep, daemon=True).start()
            else:
                self.history.clear()

            latency = time.time() - t0
            with self.lock:
                self.result = {"dets": dets, "best": best, "value": value, "scale": scale,
                               "ocr": self.scale_reader is not None,
                               "latency": latency, "time": time.time()}

    def _apply_mark_memory(self, best):
        """The min/max marks never move relative to the gauge, so remember where they are
        (as an offset from the needle pivot, in gauge widths) and reuse that position
        in frames where the model misses one of them."""
        now = time.time()
        if "center" in best:
            self.last_center_t = now
        elif now - self.last_center_t > 1.5:
            self.marks.clear()          # gauge left the view: forget its marks (next one may differ)
            self.last_gw = None
            if self.scale_reader is not None:
                self.scale_reader.reset()
        if "gauge" in best:
            self.last_gw = best["gauge"]["w"]
        gw = self.last_gw
        if "center" not in best or not gw:
            return
        cx, cy = best["center"]["x"], best["center"]["y"]
        for k in ("min", "max"):
            if k in best:
                rx, ry = (best[k]["x"] - cx) / gw, (best[k]["y"] - cy) / gw
                old = self.marks.get(k)
                self.marks[k] = (rx, ry) if old is None else (0.7 * old[0] + 0.3 * rx,
                                                              0.7 * old[1] + 0.3 * ry)
            elif k in self.marks:
                rx, ry = self.marks[k]
                best[k] = {"name": k, "conf": 0.0, "x": cx + rx * gw, "y": cy + ry * gw,
                           "w": 0, "h": 0, "remembered": True}

    def get(self):
        with self.lock:
            return self.result

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)


def draw(frame, res, args, hud):
    if res is None:
        cv2.putText(frame, "starting...", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        return
    for d in res["dets"]:
        x1, y1 = int(d["x"] - d["w"] / 2), int(d["y"] - d["h"] / 2)
        x2, y2 = int(d["x"] + d["w"] / 2), int(d["y"] + d["h"] / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
        cv2.putText(frame, f"{d['name']} {d['conf']:.2f}", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    for k in ("min", "max"):           # marks reused from memory (not detected this frame)
        m = res["best"].get(k)
        if m and m.get("remembered"):
            cv2.circle(frame, (int(m["x"]), int(m["y"])), 8, (0, 255, 255), 2)
            cv2.putText(frame, f"{k} (remembered)", (int(m["x"]) + 10, int(m["y"])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    b = res["best"]
    if "center" in b and "tip" in b:   # needle line
        cv2.line(frame, (int(b["center"]["x"]), int(b["center"]["y"])),
                 (int(b["tip"]["x"]), int(b["tip"]["y"])), (0, 255, 0), 2)

    h = frame.shape[0]
    scale = res["scale"]
    stale = time.time() - res["time"] > 2.0
    v = res["value"]
    if v is not None and not stale:
        alarm = (args.alarm_low is not None and v <= args.alarm_low) or \
                (args.alarm_high is not None and v >= args.alarm_high)
        color = (0, 0, 255) if alarm else (0, 255, 0)
        unit = args.unit if scale else "% of scale"
        cv2.putText(frame, f"{fmt(v, scale)} {unit}".strip(), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        if alarm:
            cv2.putText(frame, "OUT OF RANGE", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        missing = [k for k in ("center", "tip", "min", "max") if k not in b]
        msg = "missing: " + ", ".join(missing) if missing else "bad min/max geometry"
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    if hud["msg"] and time.time() - hud["msg_t"] < 2.5:
        cv2.putText(frame, hud["msg"], (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    if scale:
        cap = f"scale {scale[0]:g} to {scale[1]:g} ({scale[2]})"
    elif res["ocr"]:
        cap = "scale unknown yet - reading min/max numbers..."
    else:
        cap = "scale unknown - pass --vmin/--vmax for real units"
    cv2.putText(frame, "[l] save  [n] next round  [d/a] device  [q] quit", (10, h - 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)
    cv2.putText(frame, hud["label"], (10, h - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, cap, (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"inference {res['latency'] * 1000:.0f} ms", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default="0", help="camera index or stream URL")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="path to trained best.pt")
    ap.add_argument("--vmin", type=float, default=None,
                    help="manual: value at the gauge's min mark (default: read from the dial)")
    ap.add_argument("--vmax", type=float, default=None,
                    help="manual: value at the gauge's max mark (default: read from the dial)")
    ap.add_argument("--unit", default="", help="unit label to show/save, e.g. psi, bar")
    ap.add_argument("--no-ocr", action="store_true", help="don't read the scale from the dial")
    ap.add_argument("--debug-ocr", action="store_true",
                    help="print what OCR reads and save the crops it sees to ./ocr_debug")
    ap.add_argument("--ccw", action="store_true", help="scale runs counter-clockwise from min to max")
    ap.add_argument("--conf", type=float, default=0.4, help="min confidence for center/tip/gauge")
    ap.add_argument("--conf-marks", type=float, default=0.2,
                    help="min confidence for the min/max marks (lower because they are harder to detect)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device-id", default=None, help="inference device, e.g. 0 for GPU, cpu")
    ap.add_argument("--alarm-low", type=float, default=None)
    ap.add_argument("--alarm-high", type=float, default=None)
    ap.add_argument("--record", help="optional output .mp4 path")
    ap.add_argument("--log", default=DEFAULT_LOG, help="CSV file readings are appended to")
    ap.add_argument("--device-name", default="Device", help="label used in the CSV, e.g. 'Device'")
    ap.add_argument("--device", type=int, default=1, help="current device number")
    ap.add_argument("--round", type=int, default=1, help="current inspection round number")
    ap.add_argument("--auto-log", type=float, default=0.0,
                    help="also save a reading every N seconds (0 = only when you press l)")
    args = ap.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}\nTrain first (train_gauge.py) or pass --model path\\to\\best.pt")
    model = YOLO(args.model)

    scale_reader = None
    manual = args.vmin is not None and args.vmax is not None
    if not manual and not args.no_ocr:
        try:
            print("Loading OCR (first run downloads its model, needs internet)...")
            scale_reader = ScaleReader(debug=args.debug_ocr)
        except Exception as e:
            print(f"OCR not available ({e}). Install it with:  python -m pip install easyocr\n"
                  "Continuing without it: pass --vmin/--vmax for real units, otherwise % of scale is shown.")

    logger = ReadingLogger(args.log)
    state = {"round": args.round, "device": args.device}
    hud = {"label": "", "msg": "", "msg_t": 0.0}
    last_auto = time.time()

    def is_alarm(v):
        return (args.alarm_low is not None and v <= args.alarm_low) or \
               (args.alarm_high is not None and v >= args.alarm_high)

    def log_now(res):
        if res is None or res["value"] is None or time.time() - res["time"] > 2.0:
            hud.update(msg="no valid reading to save", msg_t=time.time())
            return False
        v, scale = res["value"], res["scale"]
        unit = args.unit if scale else "% of scale"
        device = f"{args.device_name} {state['device']}"
        try:
            logger.log(device, state["round"], v, scale, unit, "ALARM" if is_alarm(v) else "OK")
        except OSError as e:           # e.g. the CSV is open in Excel
            hud.update(msg=f"cannot write CSV ({e.strerror}) - close it in Excel", msg_t=time.time())
            return False
        text = f"saved: {device}, round {state['round']} = {fmt(v, scale)} {unit}".strip()
        print(text)
        hud.update(msg=text, msg_t=time.time())
        return True

    print(f"Saving readings to: {args.log}")
    stream = CameraStream(source)
    inferencer = Inferencer(stream, model, args, scale_reader)
    writer = None
    try:
        while True:
            frame, _ = stream.read()
            if frame is None:
                time.sleep(0.01)
                continue
            res = inferencer.get()
            hud["label"] = f"{args.device_name} {state['device']}  |  inspection round {state['round']}"
            draw(frame, res, args, hud)

            if args.auto_log > 0 and time.time() - last_auto >= args.auto_log:
                log_now(res)
                last_auto = time.time()

            if args.record:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                writer.write(frame)
            cv2.imshow("Gauge Live", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("l"):
                log_now(res)
            elif key == ord("n"):
                state["round"] += 1
                hud.update(msg=f"started inspection round {state['round']}", msg_t=time.time())
            elif key == ord("d"):
                state["device"] += 1
            elif key == ord("a"):
                state["device"] = max(1, state["device"] - 1)
    finally:
        inferencer.stop()
        if scale_reader is not None:
            scale_reader.stop()
        stream.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()