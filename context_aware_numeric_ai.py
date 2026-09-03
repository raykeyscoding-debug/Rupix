"""
Context-Aware Numeric + Trigonometric Learning Text AI
======================================================

Experimental features:
- Letters -> numbers (a=1 ... z=26)
- Trigonometric scoring
- Bigram + trigram language memory
- Recent conversation context
- Topic/context keywords
- Candidate response comparison
- Anti-echo / anti-copy penalties
- N-gram repetition detection
- Avoids simply rearranging the user's words

Important:
This is still a statistical experiment, not a transformer/LLM.
It cannot truly understand meaning like ChatGPT, but these systems give it
a much better approximation of context and stop obvious parroting.
"""

import json
import math
import os
import random
import re
from collections import Counter

MEMORY_FILE = "context_numeric_ai_memory.json"

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "i", "you", "he",
    "she", "it", "we", "they", "to", "of", "and", "or", "in", "on",
    "at", "for", "with", "my", "your", "this", "that", "what", "how",
    "do", "does", "did", "be", "am", "have", "has", "had"
}

PUNCTUATION = {".", ",", "!", "?", ";"}


class LearningTextAI:
    def __init__(self, memory_path=MEMORY_FILE):
        self.memory_path = memory_path

        # Basic language memory.
        self.bigram = {}
        self.trigram = {}

        # Context memory.
        self.context_memory = {}

        # Recent conversation turns.
        self.recent_inputs = []
        self.recent_outputs = []

        self.vocab = set()
        self.messages_seen = 0

        self._load()

    # ============================================================
    # PERSISTENCE
    # ============================================================

    def _load(self):
        if not os.path.exists(self.memory_path):
            return

        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.bigram = data.get("bigram", {})
            self.trigram = data.get("trigram", {})
            self.context_memory = data.get("context_memory", {})
            self.vocab = set(data.get("vocab", []))
            self.messages_seen = data.get("messages_seen", 0)

        except (json.JSONDecodeError, OSError):
            print("Warning: Could not load memory.")

    def save(self):
        data = {
            "bigram": self.bigram,
            "trigram": self.trigram,
            "context_memory": self.context_memory,
            "vocab": list(self.vocab),
            "messages_seen": self.messages_seen
        }

        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ============================================================
    # TOKENIZATION
    # ============================================================

    @staticmethod
    def _tokenize(text):
        text = text.strip().lower()
        return re.findall(r"[a-z0-9']+|[.,!?;]", text)

    @staticmethod
    def _content_words(words):
        """Remove common grammar words when detecting a topic."""
        return [
            word for word in words
            if word not in STOP_WORDS
            and word not in PUNCTUATION
            and len(word) > 1
        ]

    # ============================================================
    # LETTER -> NUMBER
    # ============================================================

    @staticmethod
    def letters_to_numbers(text):
        return [
            ord(char) - ord("a") + 1
            for char in text.lower()
            if "a" <= char <= "z"
        ]

    @staticmethod
    def word_number(word):
        values = LearningTextAI.letters_to_numbers(word)

        return sum(
            value * (index + 1)
            for index, value in enumerate(values)
        )

    def text_number(self, text):
        numbers = self.letters_to_numbers(text)

        return sum(
            number * (index + 1)
            for index, number in enumerate(numbers)
        )

    # ============================================================
    # TRIGONOMETRIC DECISION SYSTEM
    # ============================================================

    def trig_score(self, text, context_number=0):
        x = self.text_number(text) + context_number

        wave1 = math.sin(x / 7.0)
        wave2 = math.cos(x / 11.0)
        wave3 = math.sin((x + context_number) / 17.0)
        wave4 = math.cos((x * 0.5 + context_number) / 23.0)

        raw = (
            wave1 * 0.30
            + wave2 * 0.25
            + wave3 * 0.25
            + wave4 * 0.20
        )

        return (raw + 1.0) / 2.0

    # ============================================================
    # CONTEXT DETECTION
    # ============================================================

    def get_context_keywords(self, text, limit=5):
        """
        Finds the most meaningful words in the current message.

        Example:
            "Can you help me make a Python game?"

        Might become:
            ["help", "make", "python", "game"]
        """
        words = self._tokenize(text)
        content = self._content_words(words)

        counts = Counter(content)

        return [
            word
            for word, count in counts.most_common(limit)
        ]

    def context_signature(self, text):
        """
        Creates a stable context label.

        Words are sorted so:
            "python game help"
        and:
            "help with a python game"

        can share some context.
        """
        keywords = self.get_context_keywords(text)

        if not keywords:
            return "_general"

        return "|".join(sorted(keywords))

    def context_similarity(self, text_a, text_b):
        """
        Simple topic similarity using content-word overlap.

        0 = unrelated
        1 = nearly identical topic
        """
        a = set(self._content_words(self._tokenize(text_a)))
        b = set(self._content_words(self._tokenize(text_b)))

        if not a or not b:
            return 0.0

        intersection = len(a & b)
        union = len(a | b)

        return intersection / union if union else 0.0

    def recent_context_score(self, candidate):
        """
        Rewards candidates connected to recent conversation topics.
        """
        if not self.recent_inputs:
            return 0.5

        scores = [
            self.context_similarity(
                candidate,
                previous_input
            )
            for previous_input in self.recent_inputs[-4:]
        ]

        return sum(scores) / len(scores)

    # ============================================================
    # ANTI-ECHO / ANTI-PARROT SYSTEM
    # ============================================================

    def word_overlap(self, input_words, output_words):
        """
        Measures how many meaningful words are shared.

        This catches direct copying even when word order changes.
        """
        input_set = set(self._content_words(input_words))
        output_set = set(self._content_words(output_words))

        if not output_set:
            return 0.0

        return len(input_set & output_set) / len(output_set)

    @staticmethod
    def ngrams(words, size=2):
        return {
            tuple(words[i:i + size])
            for i in range(len(words) - size + 1)
        }

    def reordered_copy_score(self, input_words, output_words):
        """
        Detects a response that is basically the user's sentence
        rearranged into a different order.
        """
        a = set(
            word for word in input_words
            if word not in STOP_WORDS
            and word not in PUNCTUATION
        )

        b = set(
            word for word in output_words
            if word not in STOP_WORDS
            and word not in PUNCTUATION
        )

        if not a or not b:
            return 0.0

        # Jaccard similarity ignores word order.
        return len(a & b) / len(a | b)

    def sequence_copy_score(self, input_words, output_words):
        """
        Detects copied word sequences such as:

        User: "the dog ran through the park"
        AI:   "the dog ran through the park today"
        """
        input_bigrams = self.ngrams(input_words, 2)
        output_bigrams = self.ngrams(output_words, 2)

        if not output_bigrams:
            return 0.0

        overlap = len(input_bigrams & output_bigrams)

        return overlap / len(output_bigrams)

    def repetition_score(self, candidate_words):
        """
        Penalizes repeated words and repeated phrases inside
        the AI's own response.
        """
        if not candidate_words:
            return 1.0

        content = [
            word for word in candidate_words
            if word not in STOP_WORDS
            and word not in PUNCTUATION
        ]

        if not content:
            return 0.0

        counts = Counter(content)

        repeated_words = sum(
            count - 1
            for count in counts.values()
            if count > 1
        )

        word_penalty = repeated_words / max(len(content), 1)

        bigrams = [
            tuple(content[i:i + 2])
            for i in range(len(content) - 1)
        ]

        duplicate_bigrams = len(bigrams) - len(set(bigrams))
        phrase_penalty = duplicate_bigrams / max(len(bigrams), 1)

        return min(
            word_penalty * 0.6
            + phrase_penalty * 0.4,
            1.0
        )

    def is_too_similar(self, candidate, user_words):
        """
        Hard rejection for obvious parroting.
        """
        candidate_words = self._tokenize(
            " ".join(candidate)
        )

        if not candidate_words:
            return True

        # Exact sentence.
        if candidate_words == user_words:
            return True

        reorder = self.reordered_copy_score(
            user_words,
            candidate_words
        )

        sequence = self.sequence_copy_score(
            user_words,
            candidate_words
        )

        overlap = self.word_overlap(
            user_words,
            candidate_words
        )

        # Reject if most meaningful words came from the user.
        if reorder > 0.72:
            return True

        if sequence > 0.65:
            return True

        if overlap > 0.80 and len(candidate_words) <= len(user_words) + 3:
            return True

        return False

    # ============================================================
    # LEARNING
    # ============================================================

    def learn(self, text):
        words = self._tokenize(text)

        if not words:
            return

        self.messages_seen += 1
        self.vocab.update(words)

        # Bigram learning.
        for i in range(len(words) - 1):
            w1 = words[i]
            w2 = words[i + 1]

            self.bigram.setdefault(w1, {})
            self.bigram[w1][w2] = (
                self.bigram[w1].get(w2, 0) + 1
            )

        # Trigram learning.
        for i in range(len(words) - 2):
            key = f"{words[i]}|{words[i + 1]}"
            nxt = words[i + 2]

            self.trigram.setdefault(key, {})
            self.trigram[key][nxt] = (
                self.trigram[key].get(nxt, 0) + 1
            )

        # Store vocabulary under the message's context.
        signature = self.context_signature(text)

        self.context_memory.setdefault(
            signature,
            {}
        )

        for word in self._content_words(words):
            self.context_memory[signature][word] = (
                self.context_memory[signature].get(word, 0) + 1
            )

    # ============================================================
    # MEMORY OPTIONS
    # ============================================================

    def _get_memory_options(self, history):
        if len(history) >= 2:
            key = f"{history[-2]}|{history[-1]}"

            if key in self.trigram:
                return self.trigram[key]

        if history and history[-1] in self.bigram:
            return self.bigram[history[-1]]

        return {}

    def _context_related_words(self, seed_text):
        """
        Finds words from memory that are associated with similar contexts.

        This gives the generator a topic-based escape route instead of
        forcing it to continue the user's exact sentence.
        """
        seed_words = set(
            self._content_words(
                self._tokenize(seed_text)
            )
        )

        related = Counter()

        for signature, words in self.context_memory.items():
            signature_words = set(signature.split("|"))

            if signature == "_general":
                continue

            similarity = (
                len(seed_words & signature_words)
                / max(len(seed_words | signature_words), 1)
            )

            if similarity > 0:
                for word, count in words.items():
                    related[word] += count * similarity

        return related

    def _choose_from_memory(
        self,
        history,
        context_number,
        forbidden_words,
        context_words
    ):
        """
        Selects a next word from several memory sources.

        The forbidden-word system is important:
        words heavily used by the current user message receive a penalty,
        preventing the AI from simply rebuilding the same sentence.
        """
        options = Counter()

        # Normal language memory.
        direct_options = self._get_memory_options(history)

        for word, count in direct_options.items():
            options[word] += count

        # Context memory can introduce topic-related alternatives.
        for word, count in context_words.items():
            options[word] += count * 0.35

        if not options:
            return None

        max_count = max(options.values())
        candidates = []

        for word, count in options.items():
            memory_strength = count / max_count
            trig_strength = self.trig_score(
                word,
                context_number
            )

            # Penalize words copied directly from the user's message.
            copy_penalty = (
                0.55 if word in forbidden_words
                and word not in PUNCTUATION
                else 0.0
            )

            # Penalize words already repeated too often.
            repeated = history.count(word)
            repeat_penalty = min(repeated * 0.25, 0.60)

            score = (
                memory_strength * 0.55
                + trig_strength * 0.30
                + (1.0 - copy_penalty) * 0.15
                - repeat_penalty * 0.20
            )

            if score > 0:
                candidates.append((word, score))

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[1],
            reverse=True
        )

        shortlist = candidates[:min(6, len(candidates))]

        words = [item[0] for item in shortlist]
        weights = [max(item[1], 0.01) for item in shortlist]

        return random.choices(
            words,
            weights=weights,
            k=1
        )[0]

    # ============================================================
    # CANDIDATE GENERATION
    # ============================================================

    def _starting_points(self, seed_words, seed_text):
        """
        Context-aware starting points.

        Instead of ALWAYS starting the response with the user's last words,
        the AI can start from topic-related memory.

        This is one of the biggest fixes for parroting.
        """
        starts = []

        # Old behavior as one possible option.
        if len(seed_words) >= 2:
            starts.append(seed_words[-2:])
        elif seed_words:
            starts.append([seed_words[-1]])

        # Context-related alternative starts.
        related = self._context_related_words(seed_text)

        for word, count in related.most_common(8):
            if word not in seed_words:
                starts.append([word])

        # Random learned starts.
        vocab_choices = list(self.vocab)

        if vocab_choices:
            for _ in range(min(5, len(vocab_choices))):
                starts.append([
                    random.choice(vocab_choices)
                ])

        return starts or [[random.choice(list(self.vocab))]]

    def _make_candidate(
        self,
        seed_words,
        seed_text,
        max_words=20
    ):
        starts = self._starting_points(
            seed_words,
            seed_text
        )

        history = list(random.choice(starts))
        output = list(history)

        forbidden_words = set(
            self._content_words(seed_words)
        )

        context_words = self._context_related_words(
            seed_text
        )

        context_number = self.text_number(
            seed_text
        )

        for _ in range(max_words):
            nxt = self._choose_from_memory(
                history,
                context_number,
                forbidden_words,
                context_words
            )

            if nxt is None:
                break

            output.append(nxt)
            history.append(nxt)

            if len(history) > 2:
                history.pop(0)

            context_number += self.word_number(nxt)

            if nxt in ".!?":
                break

        return output

    # ============================================================
    # CANDIDATE SCORING
    # ============================================================

    def _score_candidate(
        self,
        candidate,
        seed_text
    ):
        text = " ".join(candidate)
        candidate_words = self._tokenize(text)
        seed_words = self._tokenize(seed_text)

        trig_part = self.trig_score(
            text,
            self.text_number(seed_text)
        )

        # Memory confidence.
        transition_scores = []

        for i in range(len(candidate) - 1):
            current_word = candidate[i]
            next_word = candidate[i + 1]

            if current_word in self.bigram:
                options = self.bigram[current_word]
                total = sum(options.values())

                if next_word in options and total:
                    transition_scores.append(
                        options[next_word] / total
                    )

        memory_part = (
            sum(transition_scores)
            / len(transition_scores)
            if transition_scores else 0.0
        )

        # Context relevance.
        context_part = self.recent_context_score(text)

        # Anti-copy measurements.
        overlap = self.word_overlap(
            seed_words,
            candidate_words
        )

        reorder_copy = self.reordered_copy_score(
            seed_words,
            candidate_words
        )

        sequence_copy = self.sequence_copy_score(
            seed_words,
            candidate_words
        )

        repetition = self.repetition_score(
            candidate_words
        )

        # Prefer useful lengths.
        length_part = min(
            len(candidate_words) / 10.0,
            1.0
        )

        # Positive signals minus copying/repetition.
        score = (
            trig_part * 0.20
            + memory_part * 0.30
            + context_part * 0.25
            + length_part * 0.10
            - overlap * 0.25
            - reorder_copy * 0.45
            - sequence_copy * 0.50
            - repetition * 0.35
        )

        return score

    # ============================================================
    # RESPONSE COMBINATION
    # ============================================================

    def _combine_candidates(
        self,
        selected,
        user_words,
        max_words=24
    ):
        """
        Combines candidates while checking each added word against:
        - repetition
        - direct copying
        - user's current vocabulary
        """
        if not selected:
            return []

        combined = []
        forbidden = set(
            self._content_words(user_words)
        )

        for candidate in selected:
            for word in candidate:
                if len(combined) >= max_words:
                    break

                # Don't repeat a word too many times.
                if (
                    word not in PUNCTUATION
                    and combined.count(word) >= 2
                ):
                    continue

                # Avoid filling the response with copied user words.
                copied_count = sum(
                    1 for existing in combined
                    if existing in forbidden
                )

                if (
                    word in forbidden
                    and copied_count >= max(1, len(combined) // 4)
                ):
                    continue

                # Avoid immediate duplicates.
                if combined and word == combined[-1]:
                    continue

                combined.append(word)

            if len(combined) >= max_words:
                break

        return combined[:max_words]

    # ============================================================
    # FINAL RESPONSE
    # ============================================================

    def generate_reply(
        self,
        seed_text="",
        max_words=24,
        candidate_count=20,
        select_count=3
    ):
        if not self.vocab:
            return (
                "(I don't know enough yet. "
                "Teach me more sentences and conversations.)"
            )

        seed_words = self._tokenize(seed_text)

        candidates = []

        # Generate many possibilities.
        for _ in range(candidate_count):
            candidate = self._make_candidate(
                seed_words,
                seed_text,
                max_words
            )

            if not candidate:
                continue

            # Reject obvious copies before scoring.
            if self.is_too_similar(
                candidate,
                seed_words
            ):
                continue

            score = self._score_candidate(
                candidate,
                seed_text
            )

            candidates.append(
                (candidate, score)
            )

        # If all candidates were rejected, try a random
        # context-related starting point.
        if not candidates:
            related = self._context_related_words(
                seed_text
            )

            possible = [
                word for word in related
                if word not in seed_words
            ]

            if possible:
                return random.choice(possible).capitalize() + "."

            return "I need more varied examples to learn this context."

        candidates.sort(
            key=lambda item: item[1],
            reverse=True
        )

        # Select the best DIFFERENT candidates.
        selected = []

        for candidate, score in candidates:
            candidate_text = " ".join(candidate)

            too_similar_to_selected = any(
                self.context_similarity(
                    candidate_text,
                    " ".join(existing)
                ) > 0.85
                for existing in selected
            )

            if not too_similar_to_selected:
                selected.append(candidate)

            if len(selected) >= select_count:
                break

        output = self._combine_candidates(
            selected,
            seed_words,
            max_words
        )

        # Final safety check.
        if (
            not output
            or self.is_too_similar(
                output,
                seed_words
            )
        ):
            output = selected[0] if selected else []

        text = " ".join(output)
        text = re.sub(
            r"\s+([.,!?;])",
            r"\1",
            text
        )

        # Save conversation context AFTER generating.
        self.recent_inputs.append(seed_text)
        self.recent_outputs.append(text)

        self.recent_inputs = self.recent_inputs[-6:]
        self.recent_outputs = self.recent_outputs[-6:]

        return text.capitalize() if text else (
            "I need more varied training examples."
        )

    # ============================================================
    # STATISTICS
    # ============================================================

    def stats(self):
        return {
            "messages_seen": self.messages_seen,
            "vocab_size": len(self.vocab),
            "bigram_pairs": len(self.bigram),
            "trigram_pairs": len(self.trigram),
            "context_groups": len(self.context_memory),
            "recent_context_turns": len(self.recent_inputs)
        }


if __name__ == "__main__":
    ai = LearningTextAI()

    print("Context-Aware Numeric + Trigonometric AI")
    print()
    print("Features:")
    print("- remembers word transitions")
    print("- tracks conversation topics")
    print("- generates multiple candidate responses")
    print("- rejects copied or rearranged input")
    print("- uses trig + memory + context scoring")
    print()
    print("Commands:")
    print("  stats")
    print("  numbers <text>")
    print("  context <text>")
    print("  quit")
    print()

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break

        command = user_input.strip()

        if command.lower() == "quit":
            break

        if command.lower() == "stats":
            print("AI stats:", ai.stats())
            continue

        if command.lower().startswith("numbers "):
            text = command[8:]

            print(
                "Numbers:",
                ai.letters_to_numbers(text)
            )
            print(
                "Numeric fingerprint:",
                ai.text_number(text)
            )
            print(
                "Trig score:",
                round(ai.trig_score(text), 4)
            )
            continue

        if command.lower().startswith("context "):
            text = command[8:]

            print(
                "Keywords:",
                ai.get_context_keywords(text)
            )
            print(
                "Context signature:",
                ai.context_signature(text)
            )
            continue

        # Generate using old memory first.
        reply = ai.generate_reply(
            seed_text=user_input
        )

        print("AI:", reply)

        # Learn the user's input afterward.
        ai.learn(user_input)
        ai.save()

    print("\nSaved progress to", ai.memory_path)
