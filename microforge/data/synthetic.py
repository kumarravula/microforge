# Copyright 2024 MicroForge Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Synthetic data generator for MicroForge.
Generates realistic tool-calling queries, negative queries (out-of-scope),
multi-tool queries, and structured extraction datasets.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class SyntheticDataGenerator:
    """Generate rich synthetic tool-calling datasets from tool schemas."""

    def __init__(self, schemas_path: Optional[str] = None, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.tools: List[Dict[str, Any]] = []
        if schemas_path:
            with open(schemas_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.tools = data.get("tools", data if isinstance(data, list) else [])

        self.cities = ["New York", "London", "Tokyo", "Paris", "Berlin", "Lagos", "San Francisco", "Mumbai", "Sydney"]
        self.rooms = ["living room", "kitchen", "bedroom", "office", "hallway", "bathroom", "garage"]
        self.artists = ["Taylor Swift", "The Beatles", "Daft Punk", "Miles Davis", "Coldplay", "Drake"]
        self.durations = [1, 2, 5, 10, 15, 20, 25, 30, 45, 60]
        self.contacts = ["Alice", "Bob", "Charlie", "David", "Emma", "Sarah", "Mom", "Dad", "Boss"]
        self.messages = [
            "I'm running 5 minutes late",
            "Are we still on for lunch?",
            "Can you send the report?",
            "See you tomorrow!",
            "Call me when you're free",
        ]
        self.negative_queries = [
            "What is the meaning of life?",
            "Can you write a poem about autumn leaves?",
            "Explain the theory of general relativity",
            "Who was the fourth president of the United States?",
            "Tell me a joke about computers",
            "How do airplanes fly in strong winds?",
            "Write a python function to invert a binary tree",
            "What is the capital of Mongolia?",
            "How does quantum superposition work?",
            "Recommend a good book on philosophy",
        ]

    def _sample_for_tool(self, tool_name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        if tool_name == "get_weather":
            city = self.rng.choice(self.cities)
            templates = [
                f"What's the weather in {city}?",
                f"How is the weather looking in {city} today?",
                f"Is it raining in {city} right now?",
                f"Check temperature for {city}",
                f"Weather report for {city} please",
            ]
            return self.rng.choice(templates), {"city": city, "units": self.rng.choice(["celsius", "fahrenheit"])}

        elif tool_name == "get_forecast":
            city = self.rng.choice(self.cities)
            days = self.rng.choice([1, 3, 5, 7])
            templates = [
                f"What's the {days}-day forecast for {city}?",
                f"Will it rain in {city} over the next {days} days?",
                f"Show me the weather forecast for {city} for {days} days",
            ]
            return self.rng.choice(templates), {"city": city, "days": days}

        elif tool_name == "set_lights":
            room = self.rng.choice(self.rooms)
            state = self.rng.choice(["on", "off"])
            brightness = self.rng.randint(10, 100)
            templates = [
                f"Turn {state} the {room} lights",
                f"Please switch {state} the lights in the {room}",
                f"Set {room} lights to {brightness}%",
                f"Dim the {room} lights to {brightness} percent",
            ]
            return self.rng.choice(templates), {"room": room, "state": state, "brightness": brightness}

        elif tool_name == "set_thermostat":
            temp = self.rng.randint(18, 25)
            mode = self.rng.choice(["heat", "cool", "auto"])
            templates = [
                f"Set the temperature to {temp} degrees",
                f"Adjust thermostat to {temp} in {mode} mode",
                f"Make it {temp} degrees in here",
            ]
            return self.rng.choice(templates), {"temperature": temp, "mode": mode}

        elif tool_name == "play_music":
            artist = self.rng.choice(self.artists)
            genre = self.rng.choice(["pop", "rock", "jazz", "electronic", "classical"])
            templates = [
                f"Play some music by {artist}",
                f"Put on some {genre} music",
                f"Play {artist} songs on shuffle",
                f"Start playing {artist}",
            ]
            return self.rng.choice(templates), {"artist": artist, "genre": genre}

        elif tool_name == "pause_music":
            templates = ["Pause music", "Pause playback", "Stop playing for a moment", "Hold the track"]
            return self.rng.choice(templates), {}

        elif tool_name == "set_timer":
            mins = self.rng.choice(self.durations)
            label = self.rng.choice(["pasta", "laundry", "study session", "tea", "workout"])
            templates = [
                f"Set a timer for {mins} minutes",
                f"Start a {mins} minute timer for {label}",
                f"Timer {mins} mins",
                f"Remind me in {mins} minutes for {label}",
            ]
            return self.rng.choice(templates), {"duration_seconds": mins * 60, "label": label}

        elif tool_name == "set_alarm":
            hour = self.rng.randint(6, 10)
            minute = self.rng.choice([0, 15, 30, 45])
            time_str = f"{hour:02d}:{minute:02d}"
            templates = [
                f"Set an alarm for {hour}:{minute:02d} AM",
                f"Wake me up at {time_str}",
                f"Alarm at {time_str} tomorrow",
            ]
            return self.rng.choice(templates), {"time": time_str, "label": "Morning Alarm"}

        elif tool_name == "send_message":
            contact = self.rng.choice(self.contacts)
            msg = self.rng.choice(self.messages)
            templates = [
                f"Send a message to {contact} saying '{msg}'",
                f"Text {contact}: {msg}",
                f"Tell {contact} that {msg}",
            ]
            return self.rng.choice(templates), {"recipient": contact, "message": msg}

        elif tool_name == "lock_door":
            door = self.rng.choice(["front door", "back door", "garage door"])
            state = self.rng.choice(["lock", "unlock"])
            templates = [
                f"{state.capitalize()} the {door}",
                f"Can you {state} the {door} please?",
            ]
            return self.rng.choice(templates), {"door": door, "action": state}

        # Fallback generic generator
        templates = [f"Execute {tool_name}", f"Please run {tool_name}"]
        return self.rng.choice(templates), {}

    def generate_dataset(
        self,
        n_samples: int = 2000,
        negative_ratio: float = 0.15,
        multi_tool_ratio: float = 0.10,
    ) -> List[Dict[str, Any]]:
        """Generate full synthetic dataset with single, multi, and negative queries."""
        dataset: List[Dict[str, Any]] = []
        tool_names = [t["name"] for t in self.tools] if self.tools else [
            "get_weather", "get_forecast", "set_lights", "set_thermostat",
            "play_music", "pause_music", "set_timer", "set_alarm", "send_message", "lock_door"
        ]

        for _ in range(n_samples):
            roll = self.rng.random()
            if roll < negative_ratio:
                # Negative sample
                query = self.rng.choice(self.negative_queries)
                dataset.append({
                    "query": query,
                    "calls": [],
                    "confidence": 0.0,
                })
            elif roll < negative_ratio + multi_tool_ratio:
                # Multi-tool sample
                t1, t2 = self.rng.sample(tool_names, 2)
                res1 = self._sample_for_tool(t1)
                res2 = self._sample_for_tool(t2)
                if res1 and res2:
                    q1, a1 = res1
                    q2, a2 = res2
                    connector = self.rng.choice([" and ", ", also ", " then "])
                    combined_query = q1.rstrip(".?") + connector + q2[:1].lower() + q2[1:]
                    dataset.append({
                        "query": combined_query,
                        "calls": [
                            {"name": t1, "arguments": a1},
                            {"name": t2, "arguments": a2},
                        ],
                        "confidence": 0.95,
                    })
            else:
                # Single tool sample
                t = self.rng.choice(tool_names)
                res = self._sample_for_tool(t)
                if res:
                    q, a = res
                    dataset.append({
                        "query": q,
                        "calls": [{"name": t, "arguments": a}],
                        "confidence": round(self.rng.uniform(0.92, 0.99), 2),
                    })

        self.rng.shuffle(dataset)
        return dataset

    def export_splits(
        self,
        out_dir: Union[str, Path],
        total_samples: int = 3000,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> Dict[str, str]:
        """Export train, val, and test splits to JSONL."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        samples = self.generate_dataset(n_samples=total_samples)
        n_train = int(total_samples * train_ratio)
        n_val = int(total_samples * val_ratio)

        splits = {
            "train": samples[:n_train],
            "val": samples[n_train:n_train + n_val],
            "test": samples[n_train + n_val:],
        }

        paths = {}
        for split_name, split_data in splits.items():
            file_path = out_path / f"{split_name}.jsonl"
            with open(file_path, "w", encoding="utf-8") as f:
                for row in split_data:
                    f.write(json.dumps(row) + "\n")
            paths[split_name] = str(file_path)

        return paths
