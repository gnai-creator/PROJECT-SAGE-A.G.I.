# SAGE-14: AGI
# Codinome: AGI
# Author: Felipe Maya Muniz

import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, GRUCell, Conv2D, Flatten, Input, Softmax

class ValueSystem(tf.keras.layers.Layer):
    """
    Represents an internal value alignment mechanism for the agent.

    The ValueSystem compares current neural activations with a mutable, non-trainable 'value vector' 
    that symbolizes ethical or goal-aligned behavior. This vector is initialized randomly to simulate 
    an innate or subjective baseline, which is iteratively updated based on projected values derived 
    from the agent's current state. This design reflects the idea of ethical fluidity rather than 
    rigid, data-driven optimization.

    Attributes:
        value_vector: A mutable vector representing internalized ethical ideals or goals, updated 
                      per forward pass but not learned through backpropagation.
        internal_ethics: A projection layer that interprets and proposes value adjustments based on 
                         the agent's current state.
        alignment_gate: Determines how much influence the internal value vector has compared to the 
                        current state representation.
        sensitivity: A trainable parameter that modulates the magnitude of synthetic 'pain' resulting 
                     from deviation between agent behavior and internal values.
    """
    def __init__(self, dim):
        super().__init__()
        self.value_vector = tf.Variable(tf.random.normal([1, dim]), trainable=False)
        self.internal_ethics = Dense(dim, activation='tanh')
        self.alignment_gate = Dense(1, activation='sigmoid')
        self.sensitivity = tf.Variable(tf.ones([1, dim]), trainable=True)

    def call(self, x):
        projection = self.internal_ethics(x)
        gate = self.alignment_gate(x)
        updated_value = 0.9 * self.value_vector + 0.1 * projection
        self.value_vector.assign(updated_value)
        ethical_aligned = x * (1 - gate) + self.value_vector * gate
        pain_signal = self.sensitivity * tf.square(x - self.value_vector)
        return ethical_aligned, gate, pain_signal

class EthicalConflict(tf.keras.layers.Layer):
    """
    Computes and accumulates an ethical conflict score over time.

    This layer assesses divergence between the agent's behavior (action), its internalized value system,
    and the hypothesized output. It simulates an ethical self-monitoring mechanism by incrementally 
    building up a conflict score, representing ongoing misalignment or discomfort.

    Attributes:
        accumulated_pain: A non-trainable running total of conflict values to capture longitudinal
                          deviations, influencing the total conflict score.
    """
    def __init__(self):
        super().__init__()
        self.accumulated_pain = tf.Variable(0.0, trainable=False)

    def call(self, action, value, output):
        conflict = tf.abs(action - value)
        score = tf.reduce_mean(conflict)
        self.accumulated_pain.assign_add(score)
        return score + self.accumulated_pain * 0.01

class ReflectiveMoralAgent(tf.keras.layers.Layer):
    """
    A recurrent ethical processing unit that integrates memory.

    This component maintains an internal hidden state (via a GRU cell) across evaluations to simulate
    ethical consistency and reflection. It processes symbolic representations, compresses them,
    and updates its state based on new observations—thereby enabling dynamic, memory-informed
    reasoning over time.

    Attributes:
        cell: A GRU cell to handle sequential state updates.
        state: A persistent memory state vector updated on each call.
        reflect: A transformation layer to prepare inputs for recurrent reasoning.
    """
    def __init__(self, dim):
        super().__init__()
        self.cell = GRUCell(dim)
        self.state = tf.Variable(tf.zeros([1, dim]), trainable=False)
        self.reflect = Dense(dim, activation='relu')

    def call(self, x):
        x = self.reflect(tf.reduce_mean(x, axis=1))
        out, state = self.cell(x, [self.state])
        self.state.assign(state[0])
        return out

class ARCMetaHypothesis(tf.keras.layers.Layer):
    """
    A simple ensemble layer modeling abstract reasoning patterns.

    This layer proposes and scores multiple transformations over the input. Intended to simulate
    hypothesis generation over symbolic states (e.g., from ARC tasks), it blends three dense transformations
    with a softmax-based attention mechanism to select the most promising one.

    Attributes:
        hypotheses: A list of dense transformations acting as candidate reasoning patterns.
        selector: A softmax layer selecting the most aligned hypothesis.
    """
    def __init__(self, dim):
        super().__init__()
        self.hypotheses = [Dense(dim) for _ in range(3)]
        self.selector = Dense(3, activation='softmax')

    def call(self, x):
        candidates = [h(x) for h in self.hypotheses]
        stacked = tf.stack(candidates, axis=1)
        scores = self.selector(tf.reduce_mean(stacked, axis=-1))
        return tf.reduce_sum(stacked * tf.expand_dims(scores, -1), axis=1)

class VisualPatternAdapter(tf.keras.layers.Layer):
    """
    Encodes 2D visual patterns into dense vectors.

    This adapter is meant to support pixel-based tasks such as those in the ARC dataset. It uses a
    small CNN to convert visual inputs into flat symbolic representations.

    Attributes:
        conv: A convolutional feature extractor.
        flatten: A flattener to prepare for dense transformation.
        adapter: A final dense encoder.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = Conv2D(hidden_dim, (3, 3), activation='relu', padding='same')
        self.flatten = Flatten()
        self.adapter = Dense(hidden_dim, activation='relu')

    def call(self, img):
        feat = self.conv(img)
        flat = self.flatten(feat)
        return self.adapter(flat)

class Sage14AGI(tf.keras.Model):
    """
    The main model architecture simulating symbolic-ethical reasoning for ARC-style tasks.

    It includes encoding, ethical alignment, hypothesis testing, and decision decoding, structured
    to process both symbolic vectors and image inputs with reflection and ethical self-monitoring.

    Author: Felipe Maya Muniz
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = Dense(hidden_dim, activation='relu')
        assert hidden_dim % 8 == 0, "hidden_dim must be divisible by 8 to match num_heads * key_dim"
        self.attn = MultiHeadAttention(num_heads=8, key_dim=hidden_dim // 8)
        self.norm = LayerNormalization()
        self.agent = ReflectiveMoralAgent(hidden_dim)
        self.value_system = ValueSystem(hidden_dim)
        self.ethical_conflict = EthicalConflict()
        self.hypothesis = ARCMetaHypothesis(hidden_dim)
        self.decoder = Dense(output_dim)

    def call(self, x, training=False):
        tf.debugging.assert_rank(x, 2)
        x = self.encoder(x)  # Symbolic state encoding
        x = tf.expand_dims(x, 1)  # Prepare for attention layer
        x = self.attn(x, x, x)  # Contextual attention
        x = self.norm(x)  # Normalize post-attention
        agent_out = self.agent(x)  # Recurrent ethical reflection
        aligned, gate, pain_signal = self.value_system(agent_out)  # Ethical alignment process
        hypothesis = self.hypothesis(agent_out)  # ARC hypothesis formulation
        conflict_score = self.ethical_conflict(agent_out, self.value_system.value_vector, hypothesis)  # Ethical divergence tracking
        output = self.decoder(aligned)  # Final decision transformation from aligned ethical vector to output
        if training:
            return output
        return output, conflict_score, gate, self.value_system.value_vector, pain_signal
