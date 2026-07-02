import math
import collections
import re
import csv
import io

class CustomDataDrivenTagger:
    def __init__(self, guesser_tag="NN", algorithm="Averaged Perceptron", context_window=2, prob_threshold=0.1):
        self.guesser_tag = guesser_tag
        self.algorithm = algorithm
        self.context_window = context_window
        self.prob_threshold = prob_threshold
        
        self.lexicon = {}  # word -> list of (tag, lemma)
        self.tag_lemmas = {} # (word, tag) -> lemma
        self.known_words = set()
        
        # Perceptron Weights
        self.perceptron_weights = {}
        self.perceptron_classes = set()
        
        # Naive Bayes probabilities
        self.nb_priors = {}
        self.nb_likelihoods = {} # feature -> {tag: count}
        self.nb_tag_counts = {}
        self.nb_feature_vocab_sizes = {}
        
        # HMM probabilities
        self.hmm_unigrams = collections.Counter()
        self.hmm_bigrams = collections.Counter()
        self.hmm_trigrams = collections.Counter()
        self.hmm_emissions = collections.Counter() # (word, tag)
        self.hmm_tag_word_counts = collections.Counter() # tag count
        self.hmm_total_tokens = 0
        self.hmm_lambdas = [0.0, 0.0, 0.0]  # [trigram, bigram, unigram]

    def parse_csv_or_tsv_content(self, content):
        """
        Parses content that could be CSV/comma-separated or whitespace/tab-separated.
        Yields a list of string elements for each line.
        """
        lines = content.splitlines()
        if not lines:
            return
            
        # Detect if it's a CSV by looking at the presence of commas in non-empty lines
        is_csv = False
        comma_lines = 0
        checked_lines = 0
        for line in lines:
            line_strip = line.strip()
            if not line_strip or line_strip.startswith('#'):
                continue
            checked_lines += 1
            if ',' in line_strip:
                comma_lines += 1
            if checked_lines >= 10:
                break
                
        if checked_lines > 0 and (comma_lines / checked_lines) > 0.5:
            is_csv = True
            
        if is_csv:
            reader = csv.reader(io.StringIO(content))
            for row in reader:
                if not row:
                    yield []
                    continue
                if row[0].strip().startswith('#'):
                    continue
                yield [part.strip() for part in row]
        else:
            for line in lines:
                line_strip = line.strip()
                if not line_strip:
                    yield []
                    continue
                if line_strip.startswith('#'):
                    continue
                parts = line_strip.split()
                yield parts

    def parse_lexicon(self, lexicon_file_content):
        """
        Parses the optional pre-annotated lexicon.
        Supports both comma and tab/space separated rows.
        """
        for parts in self.parse_csv_or_tsv_content(lexicon_file_content):
            if not parts:
                continue
            if len(parts) >= 2:
                word = parts[0]
                tag = parts[1]
                lemma = parts[2] if len(parts) >= 3 else word
                
                if word not in self.lexicon:
                    self.lexicon[word] = []
                if (tag, lemma) not in self.lexicon[word]:
                    self.lexicon[word].append((tag, lemma))
                self.tag_lemmas[(word, tag)] = lemma
                self.known_words.add(word)

    def parse_corpus(self, corpus_file_content):
        """
        Parses the mandatory pre-annotated corpus.
        Format: one token per line: word <tab/space/comma> TAG [<tab/space/comma> lemma]
        Sentences are separated by empty lines.
        """
        sentences = []
        current_sentence = []
        
        for parts in self.parse_csv_or_tsv_content(corpus_file_content):
            if not parts:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
                
            if len(parts) >= 2:
                word = parts[0]
                tag = parts[1]
                lemma = parts[2] if len(parts) >= 3 else word
                current_sentence.append((word, tag, lemma))
                self.known_words.add(word)
                self.tag_lemmas[(word, tag)] = lemma
                
        if current_sentence:
            sentences.append(current_sentence)
            
        return sentences

    def train(self, corpus_content, lexicon_content=None):
        """
        Trains the tagger on the provided pre-annotated corpus and lexicon.
        """
        # 1. Parse Lexicon if provided
        if lexicon_content:
            self.parse_lexicon(lexicon_content)
            
        # 2. Parse Corpus
        sentences = self.parse_corpus(corpus_content)
        if not sentences:
            raise ValueError("No valid training sentences found in the corpus.")
            
        # 3. If no lexicon was uploaded, build one from the corpus data
        if not lexicon_content:
            for sent in sentences:
                for word, tag, lemma in sent:
                    if word not in self.lexicon:
                        self.lexicon[word] = []
                    # Avoid duplicates
                    if (tag, lemma) not in self.lexicon[word]:
                        self.lexicon[word].append((tag, lemma))

        # 4. Train the specific algorithm
        if self.algorithm == "Averaged Perceptron":
            self._train_perceptron(sentences)
        elif self.algorithm == "Naive Bayes":
            self._train_naive_bayes(sentences)
        elif self.algorithm == "Hidden Markov Model (TnT Style)":
            self._train_hmm(sentences)

    # =========================================================================
    # AVERAGED PERCEPTRON TAGGER IMPLEMENTATION
    # =========================================================================
    
    def _get_perceptron_features(self, i, words, tags):
        """
        Extract features for the word at index i in sentence `words` with preceding tags `tags`.
        """
        features = {}
        
        def add_feat(name, val):
            features[f"{name}={val}"] = 1
            
        w_len = len(words)
        word = words[i]
        
        add_feat("bias", "")
        add_feat("word", word)
        
        # Surrounding words context
        # Prefix/Suffix features of current word
        for length in [1, 2, 3]:
            if len(word) >= length:
                add_feat(f"pref{length}", word[:length])
                add_feat(f"suff{length}", word[-length:])
                
        # Left context (tokens)
        if i >= 1:
            add_feat("w-1", words[i-1])
        else:
            add_feat("w-1", "_START_")
            
        if i >= 2:
            add_feat("w-2", words[i-2])
        else:
            add_feat("w-2", "_START2_")
            
        # Right context (tokens)
        if i < w_len - 1:
            add_feat("w+1", words[i+1])
        else:
            add_feat("w+1", "_END_")
            
        if i < w_len - 2:
            add_feat("w+2", words[i+2])
        else:
            add_feat("w+2", "_END2_")
            
        # Left context (tags)
        if i >= 1:
            add_feat("t-1", tags[i-1])
        else:
            add_feat("t-1", "_START_")
            
        if i >= 2:
            add_feat("t-2", tags[i-2])
        else:
            add_feat("t-2", "_START2_")
            
        # Bigram features
        if i >= 1:
            add_feat("w-1_w", f"{words[i-1]}_{word}")
            add_feat("t-1_w", f"{tags[i-1]}_{word}")
            
        return features

    def _train_perceptron(self, sentences, epochs=5):
        """
        Trains the Averaged Perceptron tagger.
        """
        # Collect all unique tags as classes
        classes = set()
        for sent in sentences:
            for _, tag, _ in sent:
                classes.add(tag)
        self.perceptron_classes = classes
        
        weights = {}  # feature -> {tag: weight_val}
        accumulated = {} # feature -> {tag: accumulated_weight_val}
        last_updated = {} # feature -> {tag: last_updated_step}
        
        total_steps = 0
        
        for epoch in range(epochs):
            for sent in sentences:
                words = [w for w, _, _ in sent]
                true_tags = [t for _, t, _ in sent]
                pred_tags = []
                
                for i in range(len(words)):
                    total_steps += 1
                    feats = self._get_perceptron_features(i, words, pred_tags)
                    
                    # Predict tag
                    best_tag = self.guesser_tag
                    best_score = -float('inf')
                    
                    # Score candidates
                    for tag in classes:
                        score = 0
                        for feat in feats:
                            if feat in weights and tag in weights[feat]:
                                score += weights[feat][tag]
                        if score > best_score:
                            best_score = score
                            best_tag = tag
                            
                    pred_tags.append(best_tag)
                    true_tag = true_tags[i]
                    
                    # Update weights if prediction is wrong
                    if best_tag != true_tag:
                        for feat in feats:
                            # Initialize feature dicts if necessary
                            if feat not in weights:
                                weights[feat] = {}
                                accumulated[feat] = {}
                                last_updated[feat] = {}
                                
                            for t in [true_tag, best_tag]:
                                if t not in weights[feat]:
                                    weights[feat][t] = 0
                                    accumulated[feat][t] = 0
                                    last_updated[feat][t] = 0
                                    
                            # Accumulate prior weights before change
                            # true_tag
                            steps_since_update = total_steps - last_updated[feat][true_tag]
                            accumulated[feat][true_tag] += weights[feat][true_tag] * steps_since_update
                            weights[feat][true_tag] += 1
                            last_updated[feat][true_tag] = total_steps
                            
                            # best_tag
                            steps_since_update = total_steps - last_updated[feat][best_tag]
                            accumulated[feat][best_tag] += weights[feat][best_tag] * steps_since_update
                            weights[feat][best_tag] -= 1
                            last_updated[feat][best_tag] = total_steps
                            
        # Final accumulation sweep for weights at the end of training
        for feat in weights:
            for tag in weights[feat]:
                steps_since_update = total_steps - last_updated[feat][tag]
                accumulated[feat][tag] += weights[feat][tag] * steps_since_update
                
                # Average weight calculation
                avg_weight = weights[feat][tag] - (accumulated[feat][tag] / total_steps)
                
                if feat not in self.perceptron_weights:
                    self.perceptron_weights[feat] = {}
                self.perceptron_weights[feat][tag] = avg_weight

    def _tag_perceptron(self, sentence_tokens):
        """
        Tags a sentence of tokens using the Averaged Perceptron.
        Returns a list of dicts: {'pos': tag, 'lemma': lemma, 'confidence': float}
        """
        words = sentence_tokens
        pred_tags = []
        tagged_results = []
        
        for i in range(len(words)):
            word = words[i]
            candidates = self.lexicon.get(word, self.lexicon.get(word.lower(), []))
            
            # Case 1: Out of Vocabulary
            if not candidates and word not in self.known_words and word.lower() not in self.known_words:
                pred_tags.append(self.guesser_tag)
                tagged_results.append({
                    'pos': self.guesser_tag,
                    'lemma': 'unknown_lemma',
                    'confidence': 0.0
                })
                continue
                
            # Convert candidate formats
            candidate_tags = [c[0] for c in candidates]
            if not candidate_tags:
                candidate_tags = list(self.perceptron_classes) if self.perceptron_classes else [self.guesser_tag]
                
            # Case 2: Unambiguous
            if len(candidate_tags) == 1:
                tag = candidate_tags[0]
                pred_tags.append(tag)
                # Find matching lemma
                lemma = self.tag_lemmas.get((word, tag), self.tag_lemmas.get((word.lower(), tag), word))
                tagged_results.append({
                    'pos': tag,
                    'lemma': lemma,
                    'confidence': 1.0
                })
                continue
                
            # Case 3: Ambiguous
            feats = self._get_perceptron_features(i, words, pred_tags)
            
            # Score each candidate tag
            tag_scores = {}
            for tag in candidate_tags:
                score = 0
                for feat in feats:
                    if feat in self.perceptron_weights and tag in self.perceptron_weights[feat]:
                        score += self.perceptron_weights[feat][tag]
                tag_scores[tag] = score
                
            # Calculate pseudo-probability via softmax over scores to apply threshold
            max_score = max(tag_scores.values()) if tag_scores else 0
            # Stability trick: subtract max_score from exponents
            exp_scores = {tag: math.exp(min(max(score - max_score, -20), 20)) for tag, score in tag_scores.items()}
            sum_exp = sum(exp_scores.values())
            probs = {tag: exp / sum_exp for tag, exp in exp_scores.items()}
            
            best_tag = max(probs, key=probs.get)
            best_prob = probs[best_tag]
            
            if best_prob < self.prob_threshold:
                pred_tags.append(self.guesser_tag)
                tagged_results.append({
                    'pos': self.guesser_tag,
                    'lemma': 'unknown_lemma',
                    'confidence': best_prob
                })
            else:
                pred_tags.append(best_tag)
                lemma = self.tag_lemmas.get((word, best_tag), self.tag_lemmas.get((word.lower(), best_tag), word))
                tagged_results.append({
                    'pos': best_tag,
                    'lemma': lemma,
                    'confidence': best_prob
                })
                
        return tagged_results

    # =========================================================================
    # NAIVE BAYES TAGGER IMPLEMENTATION
    # =========================================================================
    
    def _get_nb_features(self, i, words, tags):
        """
        Gets the contextual features for Naive Bayes tagger.
        Window size N is configurable (self.context_window).
        """
        features = {}
        w_len = len(words)
        
        # Word features in window
        for offset in range(-self.context_window, self.context_window + 1):
            if offset == 0:
                continue
            idx = i + offset
            if 0 <= idx < w_len:
                features[f"w_offset_{offset}"] = words[idx]
            else:
                features[f"w_offset_{offset}"] = "_START_" if offset < 0 else "_END_"
                
        # Tag features in window (only look left for resolved tags)
        for offset in range(-self.context_window, 0):
            idx = i + offset
            if 0 <= idx < len(tags):
                features[f"t_offset_{offset}"] = tags[idx]
            else:
                features[f"t_offset_{offset}"] = "_START_"
                
        return features

    def _train_naive_bayes(self, sentences):
        """
        Trains the Naive Bayes tagger.
        """
        tag_counts = collections.Counter()
        likelihood_counts = {}  # feature_name -> {feature_val -> {tag: count}}
        feature_vocab = {} # feature_name -> set()
        
        for sent in sentences:
            words = [w for w, _, _ in sent]
            tags = [t for _, t, _ in sent]
            
            for i in range(len(words)):
                tag = tags[i]
                tag_counts[tag] += 1
                
                # Baseline target word likelihood
                feat_name = "target_word"
                feat_val = words[i]
                if feat_name not in likelihood_counts:
                    likelihood_counts[feat_name] = {}
                    feature_vocab[feat_name] = set()
                if feat_val not in likelihood_counts[feat_name]:
                    likelihood_counts[feat_name][feat_val] = collections.Counter()
                likelihood_counts[feat_name][feat_val][tag] += 1
                feature_vocab[feat_name].add(feat_val)
                
                # Contextual features
                nb_feats = self._get_nb_features(i, words, tags[:i])
                for feat_name, feat_val in nb_feats.items():
                    if feat_name not in likelihood_counts:
                        likelihood_counts[feat_name] = {}
                        feature_vocab[feat_name] = set()
                    if feat_val not in likelihood_counts[feat_name]:
                        likelihood_counts[feat_name][feat_val] = collections.Counter()
                    likelihood_counts[feat_name][feat_val][tag] += 1
                    feature_vocab[feat_name].add(feat_val)
                    
        # Compute priors
        total_tokens = sum(tag_counts.values())
        self.nb_priors = {tag: count / total_tokens for tag, count in tag_counts.items()}
        self.nb_tag_counts = tag_counts
        
        # Keep references of vocab sizes for Laplace smoothing
        self.nb_feature_vocab_sizes = {fname: len(vocab) for fname, vocab in feature_vocab.items()}
        
        # Convert likelihood counts to log probabilities
        self.nb_likelihoods = likelihood_counts

    def _tag_naive_bayes(self, sentence_tokens):
        """
        Tags a sentence of tokens using Naive Bayes.
        """
        words = sentence_tokens
        pred_tags = []
        tagged_results = []
        
        for i in range(len(words)):
            word = words[i]
            candidates = self.lexicon.get(word, self.lexicon.get(word.lower(), []))
            
            # Case 1: Out of Vocabulary
            if not candidates and word not in self.known_words and word.lower() not in self.known_words:
                pred_tags.append(self.guesser_tag)
                tagged_results.append({
                    'pos': self.guesser_tag,
                    'lemma': 'unknown_lemma',
                    'confidence': 0.0
                })
                continue
                
            candidate_tags = [c[0] for c in candidates]
            if not candidate_tags:
                candidate_tags = list(self.nb_priors.keys()) if self.nb_priors else [self.guesser_tag]
                
            # Case 2: Unambiguous
            if len(candidate_tags) == 1:
                tag = candidate_tags[0]
                pred_tags.append(tag)
                lemma = self.tag_lemmas.get((word, tag), self.tag_lemmas.get((word.lower(), tag), word))
                tagged_results.append({
                    'pos': tag,
                    'lemma': lemma,
                    'confidence': 1.0
                })
                continue
                
            # Case 3: Ambiguous context lookup
            nb_feats = self._get_nb_features(i, words, pred_tags)
            
            # Log score calculation for each candidate tag
            tag_scores = {}
            for tag in candidate_tags:
                # 1. Prior
                prior = self.nb_priors.get(tag, 1e-6)
                score = math.log(prior)
                
                # 2. Target word likelihood
                target_count = 0
                if "target_word" in self.nb_likelihoods and word in self.nb_likelihoods["target_word"]:
                    target_count = self.nb_likelihoods["target_word"][word].get(tag, 0)
                # Laplace smoothing
                vocab_size = self.nb_feature_vocab_sizes.get("target_word", 1)
                prob_target = (target_count + 1) / (self.nb_tag_counts.get(tag, 0) + vocab_size)
                score += math.log(prob_target)
                
                # 3. Contextual features likelihood
                for fname, fval in nb_feats.items():
                    feat_count = 0
                    if fname in self.nb_likelihoods and fval in self.nb_likelihoods[fname]:
                        feat_count = self.nb_likelihoods[fname][fval].get(tag, 0)
                    vocab_size = self.nb_feature_vocab_sizes.get(fname, 1)
                    prob_feat = (feat_count + 1) / (self.nb_tag_counts.get(tag, 0) + vocab_size)
                    score += math.log(prob_feat)
                    
                tag_scores[tag] = score
                
            # Convert log scores to normalized probabilities via softmax
            max_log = max(tag_scores.values()) if tag_scores else 0
            exp_scores = {tag: math.exp(min(max(score - max_log, -20), 20)) for tag, score in tag_scores.items()}
            sum_exp = sum(exp_scores.values())
            probs = {tag: exp / sum_exp for tag, exp in exp_scores.items()}
            
            best_tag = max(probs, key=probs.get)
            best_prob = probs[best_tag]
            
            if best_prob < self.prob_threshold:
                pred_tags.append(self.guesser_tag)
                tagged_results.append({
                    'pos': self.guesser_tag,
                    'lemma': 'unknown_lemma',
                    'confidence': best_prob
                })
            else:
                pred_tags.append(best_tag)
                lemma = self.tag_lemmas.get((word, best_tag), self.tag_lemmas.get((word.lower(), best_tag), word))
                tagged_results.append({
                    'pos': best_tag,
                    'lemma': lemma,
                    'confidence': best_prob
                })
                
        return tagged_results

    # =========================================================================
    # HIDDEN MARKOV MODEL (TnT STYLE VITERBI DECODER)
    # =========================================================================
    
    def _train_hmm(self, sentences):
        """
        Trains the Trigram HMM model and runs linear interpolation (deleted interpolation).
        """
        for sent in sentences:
            words = [w for w, _, _ in sent]
            tags = [t for _, t, _ in sent]
            
            self.hmm_total_tokens += len(tags)
            
            # Start tag pads
            t_prev2 = "*START2*"
            t_prev1 = "*START1*"
            
            for i in range(len(tags)):
                tag = tags[i]
                word = words[i]
                
                # Trigram transitions
                self.hmm_unigrams[tag] += 1
                self.hmm_bigrams[(t_prev1, tag)] += 1
                self.hmm_trigrams[(t_prev2, t_prev1, tag)] += 1
                
                # Emissions
                self.hmm_emissions[(word, tag)] += 1
                self.hmm_tag_word_counts[tag] += 1
                
                # Slide context
                t_prev2 = t_prev1
                t_prev1 = tag
                
            # Count the final end tag sequences to pad sentence boundaries
            self.hmm_bigrams[(t_prev1, "*END*")] += 1
            self.hmm_trigrams[(t_prev2, t_prev1, "*END*")] += 1
            
        # --- DELETED INTERPOLATION SMOOTHING (Brants, 2000) ---
        # Calculates lambda weights dynamically based on training corpus counts
        l1, l2, l3 = 0, 0, 0
        total_tokens = self.hmm_total_tokens
        
        for (t1, t2, t3), c3 in self.hmm_trigrams.items():
            if c3 > 0:
                c2 = self.hmm_bigrams.get((t1, t2), 0)
                c1 = self.hmm_unigrams.get(t2, 0)
                
                # Calculate division quotients
                # Trigram factor
                q3 = (c3 - 1) / (c2 - 1) if c2 > 1 else 0
                # Bigram factor
                q2 = (self.hmm_bigrams.get((t2, t3), 0) - 1) / (c1 - 1) if c1 > 1 else 0
                # Unigram factor
                q1 = (self.hmm_unigrams.get(t3, 0) - 1) / (total_tokens - 1) if total_tokens > 1 else 0
                
                max_q = max(q3, q2, q1)
                if max_q == q3 and q3 > 0:
                    l1 += c3
                elif max_q == q2 and q2 > 0:
                    l2 += c3
                elif max_q == q1 and q1 > 0:
                    l3 += c3
                    
        sum_l = l1 + l2 + l3
        if sum_l > 0:
            self.hmm_lambdas = [l1 / sum_l, l2 / sum_l, l3 / sum_l]
        else:
            self.hmm_lambdas = [0.33, 0.33, 0.34]  # uniform fallback

    def _get_hmm_transition_prob(self, t3, t2, t1):
        """
        Transition probability P(t3 | t1, t2) using smoothed linear interpolation.
        """
        l1, l2, l3 = self.hmm_lambdas
        
        # 1. Trigram probability
        c2 = self.hmm_bigrams.get((t1, t2), 0)
        p_tri = self.hmm_trigrams.get((t1, t2, t3), 0) / c2 if c2 > 0 else 0
        
        # 2. Bigram probability
        c1 = self.hmm_unigrams.get(t2, 0)
        p_bi = self.hmm_bigrams.get((t2, t3), 0) / c1 if c1 > 0 else 0
        
        # 3. Unigram probability
        p_uni = self.hmm_unigrams.get(t3, 0) / self.hmm_total_tokens if self.hmm_total_tokens > 0 else 0
        if p_uni == 0:
            p_uni = 1e-7  # Fallback for special boundary tags like *END* or OOV tags
            
        prob = l1 * p_tri + l2 * p_bi + l3 * p_uni
        return max(prob, 1e-12)

    def _get_hmm_emission_prob(self, word, tag):
        """
        Emission probability P(word | tag).
        """
        count = self.hmm_emissions.get((word, tag), 0)
        total_tag = self.hmm_tag_word_counts.get(tag, 0)
        if total_tag == 0:
            return 1e-7
        # Laplace/add-one smoothing for emissions
        return (count + 1e-4) / (total_tag + 1e-4 * len(self.known_words))

    def _tag_hmm(self, sentence_tokens):
        """
        Tags a sentence of tokens using the smoothed Trigram HMM and Viterbi decoding.
        """
        words = sentence_tokens
        n = len(words)
        
        # Initialize Viterbi trellis: trellis[i][(t_prev, t_curr)] = log_prob
        # backpointers[i][(t_prev, t_curr)] = t_prev2
        trellis = [{} for _ in range(n)]
        backpointers = [{} for _ in range(n)]
        
        # Fetch candidate tags for each step
        candidates = []
        for word in words:
            word_cand = self.lexicon.get(word, self.lexicon.get(word.lower(), []))
            word_tags = [c[0] for c in word_cand]
            if not word_tags:
                # If word is completely unseen (OOV), fallback to guesser tag and known tags
                if word not in self.known_words and word.lower() not in self.known_words:
                    word_tags = [self.guesser_tag]
                else:
                    word_tags = list(self.hmm_unigrams.keys())
            candidates.append(word_tags)
            
        # Step 0
        w0 = words[0]
        for t0 in candidates[0]:
            trans = self._get_hmm_transition_prob(t0, "*START1*", "*START2*")
            emiss = self._get_hmm_emission_prob(w0, t0)
            score = math.log(trans) + math.log(emiss)
            trellis[0][("*START1*", t0)] = score
            backpointers[0][("*START1*", t0)] = "*START2*"
            
        # Step 1
        if n > 1:
            w1 = words[1]
            for t0 in candidates[0]:
                for t1 in candidates[1]:
                    if ("*START1*", t0) in trellis[0]:
                        trans = self._get_hmm_transition_prob(t1, t0, "*START1*")
                        emiss = self._get_hmm_emission_prob(w1, t1)
                        score = trellis[0][("*START1*", t0)] + math.log(trans) + math.log(emiss)
                        trellis[1][(t0, t1)] = score
                        backpointers[1][(t0, t1)] = "*START1*"
                        
        # Steps 2 to n-1
        for i in range(2, n):
            wi = words[i]
            for t_prev in candidates[i-1]:
                for t_curr in candidates[i]:
                    best_score = -float('inf')
                    best_prev2 = None
                    
                    for t_prev2 in candidates[i-2]:
                        if (t_prev2, t_prev) in trellis[i-1]:
                            trans = self._get_hmm_transition_prob(t_curr, t_prev, t_prev2)
                            emiss = self._get_hmm_emission_prob(wi, t_curr)
                            score = trellis[i-1][(t_prev2, t_prev)] + math.log(trans) + math.log(emiss)
                            if score > best_score:
                                best_score = score
                                best_prev2 = t_prev2
                                
                    if best_prev2 is not None:
                        trellis[i][(t_prev, t_curr)] = best_score
                        backpointers[i][(t_prev, t_curr)] = best_prev2
                        
        # Find best end sequence
        best_score = -float('inf')
        best_end_pair = None
        
        if n > 1:
            for t_prev in candidates[n-2]:
                for t_curr in candidates[n-1]:
                    if (t_prev, t_curr) in trellis[n-1]:
                        # Include final transition to *END* tag
                        trans = self._get_hmm_transition_prob("*END*", t_curr, t_prev)
                        score = trellis[n-1][(t_prev, t_curr)] + math.log(trans)
                        if score > best_score:
                            best_score = score
                            best_end_pair = (t_prev, t_curr)
        else:
            for t0 in candidates[0]:
                if ("*START1*", t0) in trellis[0]:
                    trans = self._get_hmm_transition_prob("*END*", t0, "*START1*")
                    score = trellis[0][("*START1*", t0)] + math.log(trans)
                    if score > best_score:
                        best_score = score
                        best_end_pair = ("*START1*", t0)
                        
        # Backtrack
        pred_tags = []
        if best_end_pair:
            t_prev, t_curr = best_end_pair
            pred_tags.append(t_curr)
            if n > 1:
                pred_tags.append(t_prev)
                
            for i in range(n-1, 1, -1):
                t_prev2 = backpointers[i].get((t_prev, t_curr))
                if t_prev2 is None:
                    break
                pred_tags.append(t_prev2)
                t_curr = t_prev
                t_prev = t_prev2
                
            pred_tags.reverse()
        else:
            # Viterbi failed (e.g. invalid trellis paths), fallback
            pred_tags = [self.guesser_tag] * n
            
        # Map back to final results with confidence checks
        tagged_results = []
        for i in range(n):
            word = words[i]
            assigned_tag = pred_tags[i]
            
            # Simple local emission probability as HMM confidence indicator
            count_tag = self.hmm_tag_word_counts.get(assigned_tag, 0)
            count_word_tag = self.hmm_emissions.get((word, assigned_tag), 0)
            
            # Normalised probability among candidates at that position
            word_candidates = [c[0] for c in self.lexicon.get(word, self.lexicon.get(word.lower(), []))]
            if not word_candidates:
                word_candidates = [self.guesser_tag]
                
            # If word is OOV or fallback happens
            if word not in self.known_words and word.lower() not in self.known_words:
                tagged_results.append({
                    'pos': self.guesser_tag,
                    'lemma': 'unknown_lemma',
                    'confidence': 0.0
                })
                continue
                
            # Softmax/Normalisation over candidates locally
            probs = {}
            for t in word_candidates:
                em = self.hmm_emissions.get((word, t), 0)
                tot_t = self.hmm_tag_word_counts.get(t, 0)
                probs[t] = em / tot_t if tot_t > 0 else 0
                
            sum_prob = sum(probs.values())
            norm_prob = probs.get(assigned_tag, 0) / sum_prob if sum_prob > 0 else 1.0
            
            if norm_prob < self.prob_threshold:
                tagged_results.append({
                    'pos': self.guesser_tag,
                    'lemma': 'unknown_lemma',
                    'confidence': norm_prob
                })
            else:
                lemma = self.tag_lemmas.get((word, assigned_tag), self.tag_lemmas.get((word.lower(), assigned_tag), word))
                tagged_results.append({
                    'pos': assigned_tag,
                    'lemma': lemma,
                    'confidence': norm_prob
                })
                
        return tagged_results

    # =========================================================================
    # PUBLIC TAG METHOD
    # =========================================================================
    
    def tag(self, sentence_tokens):
        """
        Tags a sentence of tokens using the trained algorithm.
        Returns a list of dicts: [{'pos': tag, 'lemma': lemma, 'confidence': float}]
        """
        if not sentence_tokens:
            return []
            
        if self.algorithm == "Averaged Perceptron":
            return self._tag_perceptron(sentence_tokens)
        elif self.algorithm == "Naive Bayes":
            return self._tag_naive_bayes(sentence_tokens)
        elif self.algorithm == "Hidden Markov Model (TnT Style)":
            return self._tag_hmm(sentence_tokens)
        else:
            # Fallback
            return [{
                'pos': self.guesser_tag,
                'lemma': 'unknown_lemma',
                'confidence': 0.0
            } for _ in sentence_tokens]
