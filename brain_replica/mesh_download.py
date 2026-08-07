import pyvista as pv
import random
import time
import numpy as np
import threading
import os
import requests
import json
import pyttsx3

# Initialize TTS engine
tts = pyttsx3.init()
tts.setProperty("rate", 175)
tts.setProperty("volume", 1.0)

def speak(text):
    """Speak out text using TTS engine."""
    print(f"[Assistant] Speaking: {text}")
    tts.say(text)
    tts.runAndWait()



TOGETHER_API_KEY = "f833dc3e1adb92eb7248479d9d19001e991aafbd649a3873246e13175ccdce5e"
TOGETHER_API_URL = os.environ.get("TOGETHER_API_URL", "https://api.together.xyz/v1/chat/completions")
TOGETHER_MODEL = os.environ.get("TOGETHER_MODEL", "lgai/exaone-3-5-32b-instruct")

# Initialize TTS engine once
_tts_engine = None

def get_tts_engine():
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = pyttsx3.init()
    return _tts_engine

def call_together_model(prompt, temperature=0.2, max_tokens=1024):
    """Call Together.ai REST-compatible chat completions endpoint.
    This function expects TOGETHER_API_KEY to be present in the environment.
    The endpoint and model are configurable via TOGETHER_API_URL and TOGETHER_MODEL env vars.
    """
    if not TOGETHER_API_KEY:
        raise RuntimeError("TOGETHER_API_KEY not set in environment. Export your key as TOGETHER_API_KEY and retry.")

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": TOGETHER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(TOGETHER_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print("Together API call failed:", e)
        return None

    # Try common response shapes documented by Together.ai
    # 1) choices[0].message.content
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        pass
    # 2) choices[0].text
    try:
        return data["choices"][0].get("text")
    except Exception:
        pass
    # 3) output key
    if "output" in data:
        return data["output"]

    # Fallback: return entire JSON as string
    return json.dumps(data)

def show_status_text(text):
    global status_text_actor
    if status_text_actor is not None:
        try:
            plotter.remove_actor(status_text_actor)
        except Exception:
            pass
    if text:
        status_text_actor = plotter.add_text(text, font_size=20, color='black')
    else:
        status_text_actor = None
    plotter.render()

def llm_conversation(prompt):
    """High-level flow that animates listening/thinking/speaking and uses Together.ai + TTS.
    Use by queuing messages like: "ask: What is a neuron?" in the existing input loop.
    """
    global current_animation_timer_id

    show_status_text("Listening...")
    # 1) Listening animation while we start the request
    stop_current_animation()
    print("LLM: starting (listening animation)")
    current_animation_timer_id = animate_pathway(listening_pathway, glow_color=[1,1,0], glow_duration=2)

    def background_task():
        # Call the model in the background thread (no VTK calls here)
        model_output = call_together_model(prompt)

        # Tell the main thread to stop the listening animation and show thinking text
        schedule_ui_update(stop_current_animation)
        schedule_ui_update(show_status_text, "Thinking...")

        # Start thinking animation on the main thread - continuous loop
        schedule_ui_update(show_status_text, "Thinking...")
        schedule_ui_update(animate_thinking_loop, thinking_pathway, [0,1,1])

        # Prepare speaking
        schedule_ui_update(show_status_text, "Speaking...")
        schedule_ui_update(start_animation_set_timer, speaking_pathway, [1,0.5,0], None)  # indefinite glow

        output_text = model_output or "(no response)"
        print("LLM Output:", output_text)

        # Run TTS in the background thread (no VTK calls here)
        engine = get_tts_engine()
        engine.say(output_text)
        engine.runAndWait()

        # Clear speaking animation and status text on the main thread
        schedule_ui_update(stop_current_animation)
        schedule_ui_update(show_status_text, "")

    threading.Thread(target=background_task, daemon=True).start()



import queue

listening_pathway = [
    "Brain Stem",
    "Left Temporal Lobe",
    "Right Temporal Lobe",
    "Left Parietal Lobe",
    "Right Parietal Lobe"
]

speaking_pathway = [
    "Left Frontal Lobe",
    "Right Frontal Lobe",
    "Left Temporal Lobe",
    "Brain Stem"
]

thinking_pathway = [
    "Left Frontal Lobe",
    "Right Frontal Lobe",
    "Left Parietal Lobe",
    "Right Parietal Lobe",
    "Left Temporal Lobe"
]

# Read the full GLTF/GLB model
full_scene = pv.read("static/brain.gltf")

# Convert to MultiBlock if needed so each GLTF node/mesh is a separate part
if not isinstance(full_scene, pv.MultiBlock):
    full_scene = pv.MultiBlock([full_scene])

def flatten_multiblock(mb):
    parts = []
    for item in mb:
        if isinstance(item, pv.MultiBlock):
            parts.extend(flatten_multiblock(item))
        else:
            parts.append(item)
    return parts

flat_parts = flatten_multiblock(full_scene)

plotter = pv.Plotter()

status_text_actor = None

# UI update queue to ensure all PyVista/VTK calls run on the main thread
ui_update_queue = queue.Queue()

def schedule_ui_update(fn, *args, **kwargs):
    """Schedule a callable to run on the main/UI thread."""
    ui_update_queue.put((fn, args, kwargs))

def process_ui_updates():
    """Execute any pending UI updates. MUST be called from the main thread (e.g. in check_queue)."""
    while not ui_update_queue.empty():
        fn, args, kwargs = ui_update_queue.get()
        try:
            fn(*args, **kwargs)
        except Exception as e:
            print("UI update failed:", e)

def animate_thinking_loop(pathway, glow_color=[0,1,1]):
    """Animate the thinking pathway continuously in a loop until stopped."""
    global current_animation_timer_id
    stop_current_animation()  # Stop any existing animation first
    print(f"Starting looping thinking animation")
    current_animation_timer_id = animate_pathway(pathway, glow_color=glow_color, glow_duration=1)

def start_animation_set_timer(pathway, glow_color=[0,0,0], glow_duration=None):
    """Helper to start an animation and store the returned timer id on the main thread.
    If glow_duration is None, run indefinitely until stopped.
    """
    global current_animation_timer_id

    indices = [region_to_node[name] for name in pathway if name in region_to_node]
    n = len(indices)
    if n == 0:
        print("No valid regions in pathway.")
        return None

    start_time = time.time()
    current_index = [0]  # use list for mutable integer in closure
    last_cycle = [0]

    def timer_callback(obj, event):
        with animation_lock:
            elapsed_total = time.time() - start_time
            if glow_duration is None:
                # Run indefinitely cycling at 2 seconds per full cycle
                cycle_duration = 2
                cycle_position = elapsed_total % cycle_duration
                intensity = (np.sin(np.pi * cycle_position / (cycle_duration / 2))) ** 2
                if int(elapsed_total / cycle_duration) != last_cycle[0]:
                    current_index[0] = (current_index[0] + 1) % n
                    last_cycle[0] = int(elapsed_total / cycle_duration)
            else:
                cycle_duration = glow_duration * 2
                cycle_position = elapsed_total % cycle_duration
                intensity = (np.sin(np.pi * cycle_position / glow_duration)) ** 2
                cycle_number = int(elapsed_total / cycle_duration)
                if cycle_number != last_cycle[0]:
                    current_index[0] = (current_index[0] + 1) % n
                    last_cycle[0] = cycle_number

            # Reset all to original colors and surface representation first
            for actor in actors:
                actor.GetProperty().SetColor(original_colors.get(actor, [1,1,1]))
                actor.GetProperty().SetRepresentationToSurface()
            # Glow current actor
            idx = indices[current_index[0]]
            actor = actors[idx]
            orig_color = original_colors.get(actor, [1,1,1])
            mixed_color = [
                glow_color[j]*intensity + orig_color[j]*(1 - intensity) for j in range(3)
            ]
            actor.GetProperty().SetColor(mixed_color)
            actor.GetProperty().SetRepresentationToWireframe()
            actor.GetProperty().SetLineWidth(3)
            actor.GetProperty().SetEdgeColor(0, 0, 0)
            plotter.render()

    timer_id = plotter.iren.add_observer("TimerEvent", timer_callback)
    try:
        if glow_duration is None:
            # Repeat every 1000 ms for indefinite glow
            plotter.iren.CreateRepeatingTimer(1000)
        else:
            plotter.iren.CreateRepeatingTimer(int(glow_duration * 1000))
    except AttributeError:
        if glow_duration is None:
            plotter.iren.create_timer(1000, repeating=True)
        else:
            plotter.iren.create_timer(int(glow_duration * 1000), repeating=True)
    return timer_id

actors = []
original_colors = {}

# final hardcoded mappings for model_1 
region_to_node = {
    "Cerebellum": 5,
    "Left Occipital Lobe": 23,
    "Right Occipital Lobe": 22,
    "Left Temporal Lobe": 4,
    "Right Temporal Lobe": 13,
    "Left Parietal Lobe": 28,
    "Right Parietal Lobe": 25,
    "Left Frontal Lobe": 14,
    "Right Frontal Lobe": 9,
    "Brain Stem": 18,
    "Pituitary Gland": 24,
}

# Hardcode colors - map to regions 
region_pastel_colors = {
    "Cerebellum":        [0.8, 0.7, 1.0],   # pastel purple
    "Left Occipital Lobe":  [0.7, 0.9, 1.0], # pastel blue
    "Right Occipital Lobe": [0.7, 1.0, 0.8], # pastel teal
    "Left Temporal Lobe":   [1.0, 0.6, 0.5], # pastel coral/orange (updated)
    "Right Temporal Lobe":  [1.0, 0.7, 0.8], # pastel pink
    "Left Parietal Lobe":   [0.8, 1.0, 0.7], # pastel green
    "Right Parietal Lobe":  [1.0, 1.0, 0.7], # pastel yellow
    "Left Frontal Lobe":    [0.7, 0.8, 1.0], # pastel periwinkle
    "Right Frontal Lobe":   [1.0, 0.85, 0.7],# pastel peach
    "Brain Stem":           [0.9, 0.7, 1.0], # pastel violet
    "Pituitary Gland":      [1.0, 0.7, 0.95],# pastel magenta
}

# Add all parts to plotter and keep track of actors and original colors, using pastel colors for regions
for i, part in enumerate(flat_parts):
    # See if this part corresponds to a region by node index
    region_name = None
    for name, idx in region_to_node.items():
        if idx == i:
            region_name = name
            break
    pastel_color = region_pastel_colors.get(region_name, [0.85, 0.85, 0.85])  # fallback light gray

    # Always override color with pastel_color, even if scalars exist
    actor = plotter.add_mesh(part, color=pastel_color, show_edges=False, opacity=1.0)
    original_colors[actor] = pastel_color
    actors.append(actor)

def highlight_region_by_name(name, duration=2, steps=50):
    if name not in region_to_node:
        print(f"Region '{name}' not found in mapping.")
        return
    node_idx = region_to_node[name]
    actor = actors[node_idx]
    orig_color = original_colors.get(actor, [1, 1, 1]) or [1, 1, 1]

    for i in range(steps):
        intensity = (np.sin(np.pi * i / steps))**2
        glow_color = [1, 1, 0]
        mixed_color = [
            glow_color[j] * intensity + orig_color[j] * (1 - intensity) for j in range(3)
        ]
        actor.GetProperty().SetColor(mixed_color)
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetLineWidth(3)
        actor.GetProperty().SetEdgeColor(0, 0, 0)
        plotter.render()
        time.sleep(duration / steps)

    actor.GetProperty().SetColor(orig_color)
    actor.GetProperty().SetRepresentationToSurface()
    plotter.render()

current_animation_timer_id = None
animation_lock = threading.Lock()

def animate_pathway(pathway, glow_color=[0,0,0], glow_duration=4, steps=20):
    """
    Animate the provided pathway with a smooth looping glow. The glow intensity is a sine wave,
    and actors fade back to their original color smoothly, rather than resetting abruptly.
    Only the glowing actor is wireframe; others blend to surface.
    """
    indices = [region_to_node[name] for name in pathway if name in region_to_node]
    n = len(indices)
    if n == 0:
        print("No valid regions in pathway.")
        return None

    start_time = time.time()
    current_index = [0]  # use list for mutable integer in closure
    last_cycle = [0]

    # Store per-actor "current color" for smooth fading
    actor_current_colors = {actor: list(original_colors.get(actor, [1, 1, 1])) for actor in actors}
    fade_speed = 0.15  # how quickly non-glowing actors fade back to original (0=slow, 1=instant)

    def timer_callback(obj, event):
        with animation_lock:
            elapsed_total = time.time() - start_time
            cycle_duration = glow_duration * 2
            cycle_position = elapsed_total % cycle_duration
            # Sine-squared for smooth up/down glow
            intensity = (np.sin(np.pi * cycle_position / glow_duration)) ** 2
            cycle_number = int(elapsed_total / cycle_duration)
            if cycle_number != last_cycle[0]:
                current_index[0] = (current_index[0] + 1) % n
                last_cycle[0] = cycle_number

            glowing_idx = indices[current_index[0]]
            for i, actor in enumerate(actors):
                orig_color = original_colors.get(actor, [1, 1, 1])
                # If this is the glowing actor, set glow color
                if i == glowing_idx:
                    mixed_color = [
                        glow_color[j] * intensity + orig_color[j] * (1 - intensity)
                        for j in range(3)
                    ]
                    actor_current_colors[actor] = mixed_color
                    actor.GetProperty().SetColor(mixed_color)
                    actor.GetProperty().SetRepresentationToWireframe()
                    actor.GetProperty().SetLineWidth(3)
                    actor.GetProperty().SetEdgeColor(0, 0, 0)
                else:
                    # Fade non-glowing actors smoothly back to original color
                    prev_color = actor_current_colors[actor]
                    new_color = [
                        prev_color[j] + fade_speed * (orig_color[j] - prev_color[j])
                        for j in range(3)
                    ]
                    actor_current_colors[actor] = new_color
                    actor.GetProperty().SetColor(new_color)
                    # Smoothly blend back to surface representation
                    actor.GetProperty().SetRepresentationToSurface()
            plotter.render()

    timer_id = plotter.iren.add_observer("TimerEvent", timer_callback)
    try:
        plotter.iren.CreateRepeatingTimer(int(glow_duration * 1000 // 10))  # 10 FPS for smoothness
    except AttributeError:
        plotter.iren.create_timer(int(glow_duration * 1000 // 10), repeating=True)
    return timer_id

def stop_current_animation():
    global current_animation_timer_id
    with animation_lock:
        if current_animation_timer_id is not None:
            if hasattr(plotter.iren, "RemoveObserver"):
                plotter.iren.RemoveObserver(current_animation_timer_id)
            elif hasattr(plotter.iren, "remove_observer"):
                plotter.iren.remove_observer(current_animation_timer_id)
            current_animation_timer_id = None
        # Reset all actors to original colors
        for actor in actors:
            actor.GetProperty().SetColor(original_colors.get(actor, [1,1,1]))
        plotter.render()

region_queue = queue.Queue()

def input_thread():
    while True:
        print("\nCommands:")
        print("- Type a brain region name to highlight it")
        print("- Type 'listening', 'speaking', or 'thinking' for pathway animations")
        print("- Type 'ask: your question' to use LLM")
        print("- Type 'stop' to stop current animation")
        print("- Type 'exit' to quit")
        
        region_name = input("\nEnter command: ").strip()
        region_queue.put(region_name)
        if region_name.lower() == "exit":
            break

threading.Thread(target=input_thread, daemon=True).start()

def check_queue(obj, event):
    process_ui_updates()
    global current_animation_timer_id
    while not region_queue.empty():
        region_name = region_queue.get()
        if region_name.lower() == "exit":
            print("Exiting...")
            stop_current_animation()
            plotter.close()
            return
        elif region_name.lower() == "stop":
            stop_current_animation()
            print("Stopped any running animation.")
        elif region_name.lower() in ["listening", "speaking", "thinking"]:
            stop_current_animation()
            pathway_map = {
                "listening": listening_pathway,
                "speaking": speaking_pathway,
                "thinking": thinking_pathway
            }
            pathway = pathway_map[region_name.lower()]
            print(f"Starting animation for pathway: {region_name}")
            current_animation_timer_id = animate_pathway(pathway)
        elif region_name.lower().startswith("ask:"):
            # Format: ask: your question here
            prompt = region_name[len("ask:"):].strip()
            if prompt:
                try:
                    llm_conversation(prompt)
                except Exception as e:
                    print("LLM conversation failed:", e)
            else:
                print("Usage: ask: <your question>")
        elif region_name in region_to_node:
            stop_current_animation()
            print(f"Highlighting {region_name}...")
            highlight_region_by_name(region_name)
        else:
            print(f"Region '{region_name}' not found. Available regions:")
            for r in region_to_node.keys():
                print(" -", r)

try:
    # Newer VTK (>=9.2)
    plotter.iren.CreateRepeatingTimer(100)
except AttributeError:
    # Older VTK
    plotter.iren.create_timer(100, repeating=True)  # use repeating keyword argument
plotter.iren.add_observer("TimerEvent", check_queue)

print("Brain visualization started!")
plotter.show()

