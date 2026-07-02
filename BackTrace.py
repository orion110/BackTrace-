#!/usr/bin/env python3

import os
import re
import json
import subprocess
import platform
import shutil
import random
import readline
import sys
import time
import threading
from datetime import datetime

import requests

# -------------------------------
# HISTORY FILE
# -------------------------------
HISTORY_FILE = os.path.expanduser("~/.backtrace_history")
readline.set_history_length(1000)
readline.parse_and_bind("tab: complete")
readline.parse_and_bind('"\\e[A": previous-history')
readline.parse_and_bind('"\\e[B": next-history')
readline.parse_and_bind('"\\e[C": forward-char')
readline.parse_and_bind('"\\e[D": backward-char')
try:
    readline.read_history_file(HISTORY_FILE)
except Exception:
    pass

def save_history():
    try:
        readline.write_history_file(HISTORY_FILE)
    except Exception:
        pass

# -------------------------------
# THEMES
# -------------------------------
THEMES = {
    "white":       "\033[1;37m",
    "orange":      "\033[1;38;5;214m",
    "light_orange": "\033[1;38;5;220m",
    "green":       "\033[1;38;5;82m",
    "cyan":        "\033[1;38;5;51m",
}
RESET = "\033[0m"

# -------------------------------
# SETTINGS
# -------------------------------
SETTINGS_DIR = os.path.expanduser("~/BackTrace")
SETTINGS_FILE = os.path.join(SETTINGS_DIR, "settings.json")
DEFAULT_SETTINGS = {
    "theme": "white",
    "model": "qwen2:0.5b",
    "voice": True,
    "mic": False,
}
os.makedirs(SETTINGS_DIR, exist_ok=True)

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                return {**DEFAULT_SETTINGS, **json.load(f)}
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()

def save_settings():
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(SETTINGS, f)
    except Exception:
        pass

SETTINGS = load_settings()
ACCENT = THEMES.get(SETTINGS["theme"], THEMES["cyan"])

# -------------------------------
# VOICE (Kokoro TTS)
# -------------------------------
KOKORO_AVAILABLE = False
_kokoro = None
_kokoro_lock = threading.Lock()
_speaking = False

try:
    from kokoro_onnx import Kokoro
    import sounddevice as sd
    import numpy as np
    KOKORO_AVAILABLE = True
except ImportError:
    pass

def _get_kokoro():
    global _kokoro
    with _kokoro_lock:
        if _kokoro is None:
            _kokoro = Kokoro(
                os.path.join(SETTINGS_DIR, "kokoro-v1.0.onnx"),
                os.path.join(SETTINGS_DIR, "voices-v1.0.bin")
            )
        return _kokoro

def _preload_kokoro():
    try:
        k = _get_kokoro()
        for phrase in ("hi", "ready", "okay"):
            k.create(phrase, voice="af_heart", speed=0.9, lang="en-us")
    except Exception:
        pass

if KOKORO_AVAILABLE and SETTINGS.get("voice"):
    threading.Thread(target=_preload_kokoro, daemon=True).start()

def clean_text(text):
    clean = text
    for code in [ACCENT, RESET]:
        clean = clean.replace(code, "")
    clean = re.sub(r'\033\[[0-9;]*m', '', clean)
    clean = re.sub(r'[•\*#`]', '', clean)
    return clean.strip()

def speak(text):
    if not SETTINGS.get("voice") or not KOKORO_AVAILABLE:
        return
    clean = clean_text(text)
    if not clean:
        return
    try:
        k = _get_kokoro()
        chunk = clean[:200].rsplit(' ', 1)[0] + "..." if len(clean) > 200 else clean
        samples, sr = k.create(chunk, voice="af_heart", speed=0.9, lang="en-us")
        sd.play(samples, sr)
        sd.wait()
    except Exception as e:
        print(f"speak error: {e}")

def wait_speaking():
    while _speaking:
        time.sleep(0.05)

def toggle_voice():
    if not KOKORO_AVAILABLE:
        vprint("Kokoro not found. Install with:")
        vprint("  pip install kokoro-onnx sounddevice numpy --break-system-packages")
        vprint(f"  Download kokoro-v1.0.onnx and voices-v1.0.bin to {SETTINGS_DIR}/")
        return
    SETTINGS["voice"] = not SETTINGS.get("voice", False)
    state = "ON" if SETTINGS["voice"] else "OFF"
    vprint(f"Voice {state}")
    save_settings()
    if SETTINGS["voice"]:
        threading.Thread(target=_preload_kokoro, daemon=True).start()
        speak("Voice mode enabled.")

# -------------------------------
# MIC (Vosk STT)
# -------------------------------
def _ensure_pulseaudio():
    try:
        subprocess.run(["pulseaudio", "--check"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        subprocess.Popen(["pulseaudio", "--start"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)

VOSK_MODEL_PATH = os.path.join(SETTINGS_DIR, "vosk-model")
VOSK_AVAILABLE = False
_VOSK_MODEL = None

try:
    import vosk
    import sounddevice as _sd_vosk
    import queue as queue_mod
    if os.path.isdir(VOSK_MODEL_PATH):
        VOSK_AVAILABLE = True
except ImportError:
    pass

def _suppress_stderr():
    devnull = open(os.devnull, 'w')
    old_fd = os.dup(2)
    os.dup2(devnull.fileno(), 2)
    return old_fd, devnull

def _restore_stderr(old_fd, devnull):
    os.dup2(old_fd, 2)
    os.close(old_fd)
    devnull.close()

def listen_mic():
    global _VOSK_MODEL
    if not VOSK_AVAILABLE:
        vprint("Vosk not available.")
        return None
    _ensure_pulseaudio()
    try:
        import array as array_mod
        if _VOSK_MODEL is None:
            old_fd, devnull = _suppress_stderr()
            try:
                _VOSK_MODEL = vosk.Model(VOSK_MODEL_PATH)
            finally:
                _restore_stderr(old_fd, devnull)

        model = _VOSK_MODEL
        q = queue_mod.Queue()
        capture_rate = 48000
        target_rate = 16000
        downsample = capture_rate // target_rate
        blocksize = 8000

        def callback(indata, frames, time_info, status):
            q.put(bytes(indata))

        rec = vosk.KaldiRecognizer(model, target_rate)
        result_text = ""
        silence_count = 0
        max_chunks = 80
        max_initial_silence = 15

        with _sd_vosk.RawInputStream(samplerate=capture_rate, blocksize=blocksize,
                                     dtype="int16", channels=1, callback=callback,
                                     device=None):
            for _ in range(max_chunks):
                data = q.get()
                a = array_mod.array('h', data)
                downsampled = bytes(array_mod.array('h', a[::downsample]))
                if rec.AcceptWaveform(downsampled):
                    res = json.loads(rec.Result())
                    text = res.get("text", "").strip()
                    if text:
                        result_text += " " + text
                        silence_count = 0
                    else:
                        silence_count += 1
                        if silence_count >= 4 and result_text.strip():
                            break
                else:
                    partial = json.loads(rec.PartialResult())
                    p = partial.get("partial", "").strip()
                    if p:
                        silence_count = 0
                        sys.stdout.write(f"\r{ACCENT}> {p}...{RESET}    ")
                        sys.stdout.flush()
                    elif not result_text:
                        silence_count += 1
                        if silence_count >= max_initial_silence:
                            break

        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()
        return result_text.strip() if result_text.strip() else None

    except Exception as e:
        vprint(f"Mic error: {e}")
        return None

def toggle_mic():
    if not VOSK_AVAILABLE:
        vprint("Vosk not available. Run:")
        vprint("  pip install vosk sounddevice --break-system-packages")
        vprint(f"  Download model to: {VOSK_MODEL_PATH}")
        vprint("  wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
        vprint(f"  unzip vosk-model-small-en-us-0.15.zip -d {VOSK_MODEL_PATH}")
        return
    SETTINGS["mic"] = not SETTINGS.get("mic", False)
    state = "ON" if SETTINGS["mic"] else "OFF"
    vprint(f"Mic {state}")
    save_settings()

def listen_loop():
    if not VOSK_AVAILABLE:
        vprint("Vosk not available.")
        return
    vprint("Voice mode locked. Say 'stop' or Ctrl+C to exit.")
    print()
    silent_rounds = 0
    while True:
        try:
            wait_speaking()
            vprint("Listening...")
            spoken = listen_mic()
            if not spoken:
                silent_rounds += 1
                if silent_rounds >= 5:
                    vprint("No input detected. Exiting voice mode.")
                    break
                continue
            silent_rounds = 0
            if spoken.lower().strip() in ("stop", "exit", "quit", "bye"):
                vprint("Exiting voice mode.")
                break
            process(spoken)
        except KeyboardInterrupt:
            print()
            vprint("Exiting voice mode.")
            break

# -------------------------------
# UI
# -------------------------------
def vprint(t):
    print(ACCENT + t + RESET)

def refresh():
    os.system("clear")
    t = datetime.now().strftime("%H:%M:%S")
    vprint(f"BackTrace >_ | {t} | {SETTINGS['model']}")
    print()
    vprint("help | listen | clear | exit")
    print()

# -------------------------------
# STREAM
# -------------------------------
def stream(text, delay=0.001):
    sys.stdout.write(ACCENT)
    for c in text:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(RESET)
    print()

# -------------------------------
# PONDERING
# -------------------------------
PONDER = [
    "thinking...",
    "pondering...",
    "let me think...",
    "processing...",
    "one moment...",
    "considering...",
    "working on it...",
    "hmm...",
]

# -------------------------------
# CONVERSATION HISTORY
# -------------------------------
HISTORY = []
MAX_HISTORY_PAIRS = 10

# -------------------------------
# OFFLINE INTELLIGENCE
# -------------------------------
GREETINGS = ["hey", "hi", "hello", "yo", "sup", "what's up", "whats up", "howdy", "hiya"]
GREETING_REPLIES = [
    "Hey. What do you need?",
    "Hi. Ready when you are.",
    "Yo. What's up?",
    "Hello. How can I help?",
    "Hey, what's going on?",
    "Howdy. What can I do for you?",
    "Hi there. What's on your mind?",
    "Hey! Good to hear from you.",
    "Sup. Need something?",
    "Hello! What can I assist you with?",
]

HOW_ARE_YOU = [
    "how are you", "how r u", "you good", "you ok",
    "hows it going", "how's it going", "how you doing", "how you doin"
]
HOW_REPLIES = [
    "Running fine. You?",
    "All systems go.",
    "Good. What do you need?",
    "Operational. Ask me something.",
    "Doing well, thanks for asking. What's up?",
    "Pretty good. Ready to help.",
    "All good here. What about you?",
    "Solid. What can I do for you?",
]

WHO_ARE_YOU = ["who are you", "what are you", "what is veo-x", "who made you", "what are you?"]
THANKS = ["thanks", "thank you", "thx", "ty", "appreciate it", "cheers"]
THANKS_REPLIES = ["No problem.", "Anytime.", "Sure thing.", "Glad I could help.", "You got it."]
GOODBYE = ["bye", "goodbye", "later", "peace", "cya", "see ya", "good night", "gn"]
GOODBYE_REPLIES = ["Later.", "Take care.", "Peace.", "See you.", "Catch you later."]

def offline_reply(prompt):
    p = prompt.lower().strip().rstrip("?!.,")
    if p in GREETINGS:
        return random.choice(GREETING_REPLIES)
    if p in HOW_ARE_YOU:
        return random.choice(HOW_REPLIES)
    if p in WHO_ARE_YOU:
        return "I am BackTrace, a local terminal AI assistant."
    if p in THANKS:
        return random.choice(THANKS_REPLIES)
    if p in GOODBYE:
        return random.choice(GOODBYE_REPLIES)
    return None

# -------------------------------
# SHELL STATE
# -------------------------------
shell_cwd = os.path.expanduser("~")

def shell_exec(cmd):
    global shell_cwd
    stripped = cmd.strip()
    if stripped.startswith("cd ") or stripped == "cd":
        target = stripped[3:].strip() if stripped != "cd" else os.path.expanduser("~")
        target = os.path.expandvars(target)
        if target.startswith("~"):
            target = os.path.expanduser(target)
        new_dir = os.path.normpath(os.path.join(shell_cwd, target))
        if os.path.isdir(new_dir):
            shell_cwd = new_dir
        else:
            print(f"cd: {target}: No such file or directory")
        return

    first = stripped.split()[0] if stripped.split() else ""
    if first in ("sudo", "top", "nano", "vim", "nvim", "htop", "less", "more", "man"):
        os.system(f"cd '{shell_cwd}' && {cmd}")
        return

    try:
        result = subprocess.run(
            cmd, shell=True, cwd=shell_cwd,
            capture_output=True, text=True, timeout=15,
        )
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip())
    except subprocess.TimeoutExpired:
        print("(command timed out)")
    except Exception as e:
        print(f"shell error: {e}")

# -------------------------------
# FETCH
# -------------------------------
def fetch_info():
    lines = []
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    os_name = line.split("=", 1)[1].strip().strip('"')
                    lines.append(f"  OS       {os_name}")
                    break
    except Exception:
        lines.append("  OS       Unknown")
    try:
        kernel = subprocess.check_output("uname -r", shell=True, text=True).strip()
        lines.append(f"  Kernel   {kernel}")
    except Exception:
        pass
    try:
        uptime = subprocess.check_output("uptime -p", shell=True, text=True).strip()
        lines.append(f"  Uptime   {uptime}")
    except Exception:
        pass
    lines.append(f"  Shell    {os.environ.get('SHELL', 'unknown')}")
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    lines.append(f"  CPU      {line.split(':', 1)[1].strip()}")
                    break
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                k, v = line.split(":", 1)
                mem[k.strip()] = v.strip()
        total = int(mem["MemTotal"].split()[0]) // 1024
        avail = int(mem["MemAvailable"].split()[0]) // 1024
        lines.append(f"  RAM      {total - avail}M / {total}M")
    except Exception:
        pass
    lines.append(f"  Model    {SETTINGS['model']}")
    lines.append(f"  Voice    {'ON (kokoro)' if SETTINGS.get('voice') else 'OFF'}")
    lines.append(f"  Mic      {'ON (vosk)' if SETTINGS.get('mic') else 'OFF'}")
    lines.append(f"  Time     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    for l in lines:
        vprint(l)
    print()

# -------------------------------
# WIKI
# -------------------------------
def wiki_search(query):
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            data = r.json()
            if data.get("type") != "disambiguation":
                extract = data.get("extract", "")
                if extract:
                    return extract
        s = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": query, "limit": 1, "format": "json"},
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        results = s.json()
        if results[1]:
            title = results[1][0]
            r2 = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}",
                timeout=5,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if r2.status_code == 200:
                return r2.json().get("extract", "")
    except Exception:
        pass
    return None

# -------------------------------
# DDG
# -------------------------------
def ddg_search(query):
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": 1, "no_html": 1, "skip_disambig": 1, "t": "veox"},
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
            timeout=6,
        )
        data = r.json()
        results = []
        abstract = data.get("AbstractText", "").strip()
        if abstract:
            results.append(abstract)
        answer = data.get("Answer", "").strip()
        if answer and answer not in abstract:
            results.append(answer)
        for result in data.get("Results", [])[:3]:
            text = result.get("Text", "").strip()
            if text and len(text) > 20:
                results.append(text)
        if not results:
            for topic in data.get("RelatedTopics", [])[:3]:
                text = topic.get("Text", "").strip()
                if text and len(text) > 20:
                    results.append(text)
        return results if results else None
    except Exception:
        return None

# -------------------------------
# SEARCH
# -------------------------------
def search(query):
    if not query:
        return "Search what?"
    wiki = wiki_search(query)
    ddg = ddg_search(query)
    output = f"{query.title()}\n"
    if wiki:
        sentences = wiki.split(". ")
        summary = ". ".join(sentences[:3]).strip()
        if not summary.endswith("."):
            summary += "."
        output += f"\n{summary}\n"
    if ddg:
        output += "\nWeb Results:\n"
        for item in ddg[:5]:
            output += f"• {item}\n"
    if not wiki and not ddg:
        return None
    return output

# -------------------------------
# OLLAMA
# -------------------------------
def ask_ollama(prompt):
    global HISTORY
    try:
        t0 = time.time()
        msg = random.choice(PONDER)
        sys.stdout.write(ACCENT + msg + RESET)
        sys.stdout.flush()

        HISTORY.append({"role": "user", "content": prompt})
        if len(HISTORY) > MAX_HISTORY_PAIRS * 2:
            HISTORY = HISTORY[-(MAX_HISTORY_PAIRS * 2):]

        messages = [
            {"role": "system", "content": (
                "You are BackTrace, a sharp and helpful terminal AI assistant. "
                "Be concise and direct. Never repeat yourself or restate what was already said. "
                "Never start responses with filler like 'Sure', 'Of course', 'Certainly', or 'Great'. "
                "Get straight to the point."
            )}
        ] + HISTORY

        r = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": SETTINGS["model"],
                "stream": False,
                "messages": messages,
            },
            timeout=60,
        )

        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

        elapsed = time.time() - t0
        data = r.json()
        response = data.get("message", {}).get("content", "").strip()
        if response:
            HISTORY.append({"role": "assistant", "content": response})
            stream(response)
            speak(response)
            vprint(f"[{elapsed:.2f}s]")
        else:
            vprint(f"(empty response: {data})")
    except requests.exceptions.ConnectionError:
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
        vprint("Ollama not running. Start with: ollama serve")
    except Exception as e:
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
        vprint(f"Ollama error: {type(e).__name__}: {e}")

# -------------------------------
# PROCESS
# -------------------------------
def process(cmd):
    c = cmd.strip()
    cl = c.lower()

    if cl == "clear":
        refresh()
        return

    if cl in ("exit", "quit"):
        save_history()
        vprint("BackTrace shutting down.")
        sys.exit(0)

    if cl == "help":
        voice_status = "ON" if SETTINGS.get("voice") else "OFF"
        mic_status = "ON" if SETTINGS.get("mic") else "OFF"
        vprint(f"""
Commands:
  search <query>   wiki + web search, falls back to AI
  fetch            system info
  voice            toggle voice on/off
  mic              toggle mic on/off
  listen           lock into continuous voice mode (say 'stop' to exit)
  help             this menu
  info             BackTrace info
  clear            clear screen
  exit             quit

Status: voice:{voice_status} | mic:{mic_status}
""")
        return

    if cl == "info" or cl in WHO_ARE_YOU:
        vprint(f"BackTrace — Local terminal AI assistant")
        vprint(f"OS: {platform.system()} | Kernel: {platform.release()}")
        vprint(f"Model: {SETTINGS['model']}")
        vprint(f"Voice: {'ON (kokoro)' if SETTINGS.get('voice') else 'OFF'}")
        vprint(f"Mic:   {'ON (vosk)' if SETTINGS.get('mic') else 'OFF'}\n")
        speak(f"BackTrace. Local terminal AI assistant. Running on {platform.system()} with kernel {platform.release()}. Model is {SETTINGS['model']}.")
        return

    if cl == "fetch":
        fetch_info()
        return

    if cl == "voice":
        toggle_voice()
        return

    if cl == "mic":
        toggle_mic()
        return

    if cl == "listen":
        if not VOSK_AVAILABLE:
            vprint("Vosk not available. Type 'help' for setup instructions.")
            return
        listen_loop()
        return

    if cl in ("time", "whats the time", "what's the time", "what time is it", "current time"):
        t = datetime.now().strftime("Time: %H:%M:%S | Date: %A, %B %d %Y")
        vprint(t)
        speak(t)
        print()
        return

    if cl.startswith("search "):
        q = cmd[7:].strip()
        vprint("\nBackTrace Search Results:\n")
        t0 = time.time()
        result = search(q)
        elapsed = time.time() - t0
        if result:
            stream(result)
            speak(result)
            vprint(f"[{elapsed:.2f}s]")
        else:
            vprint("No results found. Asking AI...\n")
            ask_ollama(q)
        print()
        return

    # OFFLINE LAYER
    offline = offline_reply(cmd)
    if offline:
        if random.random() < 0.5:
            vprint(offline)
            speak(offline)
        else:
            ask_ollama(cmd)
        print()
        return

    # SHELL PASSTHROUGH
    first_word = cl.split()[0] if cl.split() else ""
    SHELL_PREFIXES = (
        "ls", "ll", "cat", "pwd", "echo", "mkdir", "rm", "mv", "cp",
        "touch", "grep", "find", "df", "du", "ps", "kill", "top",
        "ping", "curl", "wget", "git", "python", "python3", "pip",
        "nano", "vim", "nvim", "cd", "chmod", "chown", "tar", "unzip",
        "systemctl", "journalctl", "apt", "dnf", "pacman", "sudo",
        "neofetch", "htop", "ssh", "scp", "rsync", "zip", "which",
        "whoami", "uname", "lsblk", "lscpu", "free", "uptime", "date",
    )
    if first_word in SHELL_PREFIXES or c.startswith("./") or c.startswith("/"):
        shell_exec(c)
        return

    QUESTION_WORDS = {"who", "what", "when", "where", "why", "how", "is", "are", "does", "do", "can", "will", "should", "would", "could"}
    if first_word and first_word not in QUESTION_WORDS and shutil.which(first_word):
        shell_exec(c)
        return

    ask_ollama(cmd)
    print()

# -------------------------------
# MAIN LOOP
# -------------------------------
refresh()

while True:
    try:
        user = input(ACCENT + f"[{shell_cwd}] ~ " + RESET)
        if user.strip().lower() in ("exit", "quit"):
            save_history()
            vprint("BackTrace >_ shutting down.")
            sys.exit(0)
        if user.strip():
            process(user)
    except KeyboardInterrupt:
        print()
        continue
    finally:
        save_history()
