import re

def process_operator(lemma, operator):
    if not operator or operator.strip() == "":
        return lemma

    # Check for Reduplication block <E1>...<E2>
    if "<E1>" in operator and "<E2>" in operator:
        # Split by hyphen or direct placement
        # Handle <E1>-<E2> or <E1>-[<E2><DL1>]
        parts = operator.split("-")
        res_parts = []
        for p in parts:
            p_clean = p.replace("[", "").replace("]", "")
            if p_clean == "<E1>":
                res_parts.append(lemma)
            elif p_clean.startswith("<E2>"):
                ops = p_clean[4:]
                res_parts.append(apply_single_ops(lemma, ops))
            else:
                res_parts.append(apply_single_ops(lemma, p_clean))
        return "-".join(res_parts)
    else:
        return apply_single_ops(lemma, operator)

def apply_single_ops(lemma, ops_str):
    cursor = len(lemma)
    buf = list(lemma)

    # Parse operators sequentially
    # Operators can be <L>, <R>, <Nk>, <NBk>, <Dk>, <DNk>, <DLk> or literal text
    tokens = re.split(r'(<[^>]+>)', ops_str)
    
    for token in tokens:
        if not token:
            continue
        if token == "<L>":
            cursor = 0
        elif token == "<R>":
            cursor = len(buf)
        elif token.startswith("<N") and token[2:-1].isdigit():
            k = int(token[2:-1])
            cursor = min(len(buf), cursor + k)
        elif token.startswith("<NB") and token[3:-1].isdigit():
            k = int(token[3:-1])
            cursor = max(0, cursor - k)
        elif token.startswith("<DL") and token[3:-1].isdigit():
            k = int(token[3:-1])
            # Delete k chars to the left of cursor
            start = max(0, cursor - k)
            buf = buf[:start] + buf[cursor:]
            cursor = start
        elif token.startswith("<D") and token[2:-1].isdigit():
            k = int(token[2:-1])
            # Delete k chars to the right of cursor
            buf = buf[:cursor] + buf[cursor + k:]
        else:
            # Literal insertion at cursor
            insert_chars = list(token)
            buf = buf[:cursor] + insert_chars + buf[cursor:]
            cursor += len(insert_chars)

    return "".join(buf)

def generate_full_forms():
    # 1. Load Lemmata
    lemmas = []
    with open(r"C:\Users\priha\Documents\cortex\model\rule-based\id-lemma.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                lemmas.append((parts[0], parts[1])) # (code, lemma)

    # 2. Load Rules
    rules = []
    with open(r"C:\Users\priha\Documents\cortex\model\rule-based\id-word-formation.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip comments
            if "#" in line:
                line = line.split("#")[0].strip()
            parts = line.split(maxsplit=2)
            code = parts[0]
            pos = parts[1]
            op = parts[2] if len(parts) > 2 else ""
            rules.append((code, pos, op))

    # 3. Generate Word Form Entries (Word-form.txt format: WordForm \t POS \t Lemma)
    word_forms = []
    seen = set()

    for code, lemma in lemmas:
        for r_code, pos, op in rules:
            if code == r_code:
                formed_word = process_operator(lemma, op)
                entry = (formed_word, pos, lemma)
                if entry not in seen:
                    seen.add(entry)
                    word_forms.append(entry)

    # Write id-word-form.txt
    with open(r"C:\Users\priha\Documents\cortex\model\rule-based\id-word-form.txt", "w", encoding="utf-8") as f:
        for word, pos, lemma in word_forms:
            f.write(f"{word}\t{pos}\t{lemma}\n")

    # Generate id-lexicon.txt (Combines base lexicon items + synthesized word forms)
    # Include closed-class words & full forms
    closed_class = [
        ("yang", "PRON", "yang"),
        ("dan", "CCONJ", "dan"),
        ("di", "PREP", "di"),
        ("ke", "PREP", "ke"),
        ("dari", "PREP", "dari"),
        ("ini", "DET", "ini"),
        ("itu", "DET", "itu"),
        ("saya", "PRON", "saya"),
        ("kamu", "PRON", "kamu"),
        ("mereka", "PRON", "mereka"),
    ]

    lexicon_entries = list(closed_class)
    for word, pos, lemma in word_forms:
        lexicon_entries.append((word, pos, lemma))

    with open(r"C:\Users\priha\Documents\cortex\model\rule-based\id-lexicon.txt", "w", encoding="utf-8") as f:
        for word, pos, lemma in lexicon_entries:
            f.write(f"{word}\t{pos}\t{lemma}\n")

    print(f"Generated {len(word_forms)} synthesized word forms in id-word-form.txt")
    print(f"Generated {len(lexicon_entries)} total entries in id-lexicon.txt")

if __name__ == "__main__":
    generate_full_forms()
