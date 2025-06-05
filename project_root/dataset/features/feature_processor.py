from abc import ABC, abstractmethod
import numpy as np # type: ignore

# ============================
# Abstract Base
# ============================
class FeatureProcessor(ABC):
    def __init__(self, raw_data=None, processed_data=None):
        self.raw_data = raw_data
        self.processed_data = processed_data

    @abstractmethod
    def process(self):
        pass

    def reduce_dimensionality(self, method='autoencoder'):
        pass


# ============================
# Processors for Fixed-Length Outputs
# ============================

class FixedVectorProcessor(FeatureProcessor):
    """ Abstract class for processors that return a fixed-length vector per sample """
    pass


class PseAACProcessor(FixedVectorProcessor):
    def __init__(self, sequences, λ=10, ω=0.05):
        super().__init__(sequences)
        self.λ = λ
        self.ω = ω

    def process(self):
        aa = 'ACDEFGHIKLMNPQRSTVWY'
        aa_index = {res: i for i, res in enumerate(aa)}

        def seq_to_vec(seq):
            L = len(seq)
            f = np.zeros(20)
            for res in seq:
                if res in aa_index:
                    f[aa_index[res]] += 1
            f /= L

            θ = np.zeros(self.λ)
            for k in range(1, self.λ + 1):
                if L <= k:
                    continue
                θ[k - 1] = sum([
                    abs(aa_index.get(seq[i], 0) - aa_index.get(seq[i + k], 0)) / 20.0
                    for i in range(L - k)
                ]) / (L - k)

            denom = sum(f) + self.ω * sum(θ)
            return np.concatenate([f, self.ω * θ]) / denom

        self.processed_data = np.array([seq_to_vec(seq) for seq in self.raw_data])


class SVMProt188DProcessor(FixedVectorProcessor):
    def process(self):
        aa_groups = {
            'hydrophobicity': {
                'polar': set('RKEDQN'),
                'neutral': set('GASTPHY'),
                'hydrophobic': set('CVLIMFW')
            },
            'van_der_waals_volume': {
                'small': set(['A', 'G', 'S', 'T', 'P', 'D', 'C']),
                'medium': set(['N', 'V', 'E', 'Q', 'I', 'L']),
                'large': set(['M', 'H', 'K', 'F', 'R', 'Y', 'W'])
            },
            'polarity': {
                'nonpolar': set(['G', 'A', 'V', 'L', 'I', 'M', 'F', 'W', 'P']),
                'polar_uncharged': set(['S', 'T', 'C', 'Y', 'N', 'Q']),
                'polar_charged': set(['D', 'E', 'K', 'R', 'H'])
            },
            'polarizability': {
                'low': set(['G', 'A', 'S', 'T', 'P', 'D']),
                'medium': set(['C', 'E', 'N', 'Q', 'K', 'H']),
                'high': set(['M', 'I', 'L', 'V', 'F', 'Y', 'W', 'R'])
            },
            'charge': {
                'positive': set(['K', 'R', 'H']),
                'negative': set(['D', 'E']),
                'neutral': set(['A', 'N', 'C', 'Q', 'G', 'I', 'L', 'M', 'F', 'P', 'S', 'T', 'V', 'W', 'Y'])
            },
            'secondary_structure': {
                'helix': set(['E', 'A', 'L', 'M', 'Q', 'K', 'R', 'H']),
                'strand': set(['V', 'I', 'Y', 'F', 'W', 'T']),
                'coil': set(['G', 'N', 'P', 'S', 'D', 'C'])
            },
            'solvent_accessibility': {
                'buried': set(['A', 'L', 'F', 'C', 'G', 'I', 'V', 'W']),
                'exposed': set(['R', 'K', 'Q', 'E', 'N', 'D']),
                'intermediate': set(['M', 'S', 'P', 'T', 'H', 'Y'])
            },
            'surface_tension': {
                'low': set(['A', 'G', 'S', 'T', 'P']),
                'medium': set(['D', 'E', 'N', 'Q', 'K']),
                'high': set(['M', 'I', 'L', 'V', 'F', 'Y', 'W', 'R', 'H', 'C'])
            }
        }

        def feature_vector(seq):
            L = len(seq)
            vec = []

            aa = 'ACDEFGHIKLMNPQRSTVWY'
            aa_count = [seq.count(res) / L for res in aa]
            vec.extend(aa_count)

            for prop in aa_groups:
                group = aa_groups[prop]
                group_counts = {k: 0 for k in group}
                for res in seq:
                    for g, chars in group.items():
                        if res in chars:
                            group_counts[g] += 1

                comp = [group_counts[g] / L for g in group]
                vec.extend(comp)

                trans = []
                group_keys = list(group.keys())
                for i in range(len(group_keys)):
                    for j in range(i + 1, len(group_keys)):
                        g1, g2 = group_keys[i], group_keys[j]
                        cnt = 0
                        for k in range(L - 1):
                            a, b = seq[k], seq[k + 1]
                            if (a in group[g1] and b in group[g2]) or (a in group[g2] and b in group[g1]):
                                cnt += 1
                        trans.append(cnt / (L - 1))
                vec.extend(trans)

                for g in group:
                    pos = [i for i, res in enumerate(seq) if res in group[g]]
                    if len(pos) == 0:
                        vec.extend([0]*5)
                    else:
                        vec.extend([
                            pos[0]/L,
                            pos[int(len(pos)*0.25)]/L if len(pos) > 1 else pos[0]/L,
                            pos[int(len(pos)*0.5)]/L if len(pos) > 2 else pos[0]/L,
                            pos[int(len(pos)*0.75)]/L if len(pos) > 3 else pos[0]/L,
                            pos[-1]/L
                        ])
            return vec

        self.processed_data = np.array([feature_vector(seq) for seq in self.raw_data])

class Pse3DiProcessor(FixedVectorProcessor):
    def __init__(self, sequences, λ=10, ω=0.05):
        super().__init__(sequences)
        self.λ = λ
        self.ω = ω

    def process(self):
        tokens = sorted(list({t for seq in self.raw_data for t in seq}))
        token_index = {t: i for i, t in enumerate(tokens)}
        n_tokens = len(token_index)

        def seq_to_vec(seq):
            L = len(seq)
            f = np.zeros(n_tokens)
            for res in seq:
                if res in token_index:
                    f[token_index[res]] += 1
            f /= L

            θ = np.zeros(self.λ)
            for k in range(1, self.λ + 1):
                if L <= k:
                    continue
                θ[k - 1] = sum([
                    abs(token_index.get(seq[i], 0) - token_index.get(seq[i + k], 0)) / n_tokens
                    for i in range(L - k)
                ]) / (L - k)

            denom = sum(f) + self.ω * sum(θ)
            return np.concatenate([f, self.ω * θ]) / denom

        self.processed_data = np.array([seq_to_vec(seq) for seq in self.raw_data])


# ============================
# Tokenized Processors
# ============================
class TokenizedProcessor(FeatureProcessor):
    def __init__(self, raw_data, token_type='sequence', encoding='onehot', vocab=None, max_length=128):
        super().__init__(raw_data)
        self.token_type = token_type
        self.encoding = encoding
        self.max_length = max_length
        self.vocab = vocab or self._build_vocab(raw_data)

    def _build_vocab(self, tokens):
        unique = set(t for sample in tokens for t in sample)
        return {t: i for i, t in enumerate(sorted(unique))}

    def one_hot_encode(self, tokens):
        if self.token_type == 'sequence':
            return self._encode_sequence(tokens)
        else:
            return self._encode_set(tokens)

    def _encode_sequence(self, tokens):
        encoded = []
        for seq in tokens:
            # Ensure seq is a list of tokens (convert string to list of characters if necessary)
            if isinstance(seq, str):
                seq = list(seq)
            
            # Pad or truncate the sequence to the maximum length
            seq = seq[:self.max_length] + ['<PAD>'] * (self.max_length - len(seq))
            
            # Convert tokens to one-hot vectors
            vecs = [self._token_to_onehot(tok) for tok in seq]
            encoded.append(vecs)
        
        return np.array(encoded)

    def _encode_set(self, tokens):
        encoded = []
        for group in tokens:
            vec = np.zeros(len(self.vocab))
            for tok in group:
                idx = self.vocab.get(tok)
                if idx is not None:
                    vec[idx] = 1
            encoded.append(vec)
        return np.array(encoded)

    def _token_to_onehot(self, tok):
        vec = np.zeros(len(self.vocab))
        idx = self.vocab.get(tok, None)
        if idx is not None:
            vec[idx] = 1
        return vec

    def process(self):
        if self.encoding == 'onehot':
            self.processed_data = self.one_hot_encode(self.raw_data)
        elif self.encoding == 'embedding':
            self.processed_data = self.embed(self.raw_data)

    def embed(self, tokens):
        # Placeholder for learned embeddings
        pass


# ============================
# Specific Token Processors
# ============================

class SequenceProcessor(TokenizedProcessor):
    def __init__(self, raw_data, encoding='onehot', max_length=512):
        super().__init__(raw_data, token_type='sequence', encoding=encoding, max_length=max_length)


class ThreeDiProcessor(TokenizedProcessor):
    def __init__(self, raw_data, encoding='onehot', max_length=256):
        super().__init__(raw_data, token_type='sequence', encoding=encoding, max_length=max_length)


class GOProcessor(TokenizedProcessor):
    def __init__(self, raw_data, encoding='onehot'):
        super().__init__(raw_data, token_type='set', encoding=encoding)


# ============================
# Embedding Processor (pLM)
# ============================
class EmbeddingProcessor(FeatureProcessor):
    def __init__(self, model_name, raw_data=None, processed_data=None, pooling_strategy='mean'):
        super().__init__(raw_data, processed_data)
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy

    def process(self):
        if self.processed_data is not None:
            return self.processed_data

        if self.pooling_strategy == 'mean':
            self.processed_data = np.mean(self.raw_data, axis=1)
        # Add other pooling strategies

class pLMProcessor(EmbeddingProcessor):
    pass
