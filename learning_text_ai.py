"""
Learning Text AI
================

A bare-bones chat system that gets better at talking the more you talk to it.
No ML libraries — just a word-level n-gram model (like a smarter autocomplete)
that updates itself after every message and saves to disk, so it keeps
learning across sessions too.

How it learns:
  - Every message you type is broken into words.
  - For each word, it remembers what words tend to follow it (and what
    follows *pairs* of words, for better context) along with how often.
  - To reply, it chains words together by picking probable next-words,
    weighted by how often it has seen that transition.
  - The more you chat, the richer its transition tables get, so replies
    stop being random and start sounding like "you".

This is a statistical parrot, not true understanding — but it visibly
improves with more data, which is the point.
"""

import json
import os
import random
import re

MEMORY_FILE = "text_ai_memory.json"


class LearningTextAI:
    def __init__(self, memory_path=MEMORY_FILE):
        self.memory_path = memory_path
        # bigram: word -> {next_word: count}
        self.bigram = {}
        # trigram: (word1, word2) -> {next_word: count}
        self.trigram = {}
        self.vocab = set()
        self.messages_seen = 0
        self._load()

    # ---------- persistence ----------

    def _load(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.bigram = data.get("bigram", {})
            # json can't have tuple keys, so trigram keys are stored as "w1|w2"
            self.trigram = data.get("trigram", {})
            self.vocab = set(data.get("vocab", []))
            self.messages_seen = data.get("messages_seen", 0)

    def save(self):
        data = {
            "bigram": self.bigram,
            "trigram": self.trigram,
            "vocab": list(self.vocab),
            "messages_seen": self.messages_seen,
        }
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    # ---------- learning ----------

    @staticmethod
    def _tokenize(text):
        text = text.strip().lower()
        return re.findall(r"[a-z0-9']+|[.,!?;]", text)

    def learn(self, text):
        words = self._tokenize(text)
        if not words:
            return
        self.messages_seen += 1
        self.vocab.update(words)

        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            self.bigram.setdefault(w1, {})
            self.bigram[w1][w2] = self.bigram[w1].get(w2, 0) + 1

        for i in range(len(words) - 2):
            key = f"{words[i]}|{words[i+1]}"
            nxt = words[i + 2]
            self.trigram.setdefault(key, {})
            self.trigram[key][nxt] = self.trigram[key].get(nxt, 0) + 1

    # ---------- generation ----------

    def _weighted_choice(self, options):
        words = list(options.keys())
        weights = list(options.values())
        return random.choices(words, weights=weights, k=1)[0]

    def generate_reply(self, seed_text="", max_words=20):
        if not self.vocab:
            return "(I don't know any words yet — say something to teach me.)"

        words = self._tokenize(seed_text)
        # start from the last word or two of the seed if we know them, else random known word
        if len(words) >= 2 and f"{words[-2]}|{words[-1]}" in self.trigram:
            history = [words[-2], words[-1]]
        elif words and words[-1] in self.bigram:
            history = [words[-1]]
        else:
            history = [random.choice(list(self.vocab))]

        output = list(history)
        for _ in range(max_words):
            key = f"{history[-2]}|{history[-1]}" if len(history) >= 2 else None
            if key and key in self.trigram:
                nxt = self._weighted_choice(self.trigram[key])
            elif history[-1] in self.bigram:
                nxt = self._weighted_choice(self.bigram[history[-1]])
            else:
                break
            output.append(nxt)
            history.append(nxt)
            if len(history) > 2:
                history.pop(0)
            if nxt in ".!?":
                break

        # tidy up spacing around punctuation
        text = " ".join(output)
        text = re.sub(r"\s+([.,!?;])", r"\1", text)
        return text.capitalize()

    def stats(self):
        return {
            "messages_seen": self.messages_seen,
            "vocab_size": len(self.vocab),
            "bigram_pairs": len(self.bigram),
            "trigram_pairs": len(self.trigram),
        }


if __name__ == "__main__":
    ai = LearningTextAI()
    print("Learning Text AI — talk to it and it gets better over time.")
    print("Commands: 'stats' to see what it has learned, 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.strip().lower() == "quit":
            break
        if user_input.strip().lower() == "stats":
            print("AI stats:", ai.stats())
            continue

        reply = ai.generate_reply(seed_text=user_input)
        print("AI:", reply)

        ai.learn(user_input)
        ai.save()

    print("\nSaved progress to", ai.memory_path)
