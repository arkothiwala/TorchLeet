# Sinusoidal Positional Embeddings — Implementation Notes & Learnings

## Context

As part of a self-implementation assignment, I implemented **Sinusoidal Positional Embeddings** from scratch (as introduced in the Transformer paper).

During the process, I made a subtle but fundamentally breaking mistake that helped clarify several deep properties of sinusoidal embeddings.

This document records:

* The initial mistake
* Why it seemed reasonable
* Why it was mathematically wrong
* What it breaks
* The conceptual learnings gained

---
# TLDR
- tried to implement myself and made a fundamental mistake as discussed below
- continuously made mistakes b/w timestamps and frequency dimensions repeadly
- created issues related to dimensionality [i.e. creating 1D vectors for timestamp and frequency -> later reshaped to (-1,1) and (1,-1) so that they it could return (time_stamps, model_dim) shape] -> This was due to the kind of choice I had made to create different functions [for scaler input] and then improving them to handle vectorized input -> could have thought from tensor angle from the beginning.


---

# The Clock Analogy

<strike>

- We can apply clock analogy with the sinusoidal position embeddings in following way:
    - clock time representation - 13:04:25
    - 6d vector [i.e. [1,3,0,4,2,5]] essentially represent 3 clocks rotating at three different frequency [hour, minute, second]
    - dials
        - eqn - $10000**(-2*i/n_dim)$
            - here the progression increments the dial speed in arithmetic progression
        - clock's equivalent eq - 60**i [hour, minute, second]
            - here the progression is more like geometrical one
        - i's progression determines relative speed b/w dials. In the eqn the speed changes gradually among the dials. For clock it is massive and direct multiple of 60.
    - base
        - eqn - 10000
        - clock - I don't know but it appears how much does the clock move when hour increments by an hour - I am confused here because when hour moved by 1 -> minute and second will be on the same position anyway -> this is not happening in the sinusoidal positional encoding -> there the increment is subtle
</strike>
---

# The Initial Mistake

My [implementation](https://github.com/arkothiwala/TorchLeet/commit/f50b5ef3a52ed7076472b4d7ffa5d594e7f72c16) computed:

```python
sine_op = self.get_sine_wave(idx) * (1 - np.arange(self.n_dim) % 2)
cos_op  = self.get_cos_wave(idx) * (np.arange(self.n_dim) % 2)
return sine_op + cos_op
```

Where:

```python
sin(idx / 10000^(dim / d_model))
cos(idx / 10000^(dim / d_model))
```

I masked:

* even dimensions → sine
* odd dimensions → cosine

At first glance, this seemed correct.

---

# Why It Looked Correct

The Transformer paper defines:

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$
$$
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

My reasoning:

* Compute sin and cos across all dimensions
* Use masking to keep sine in even dims and cosine in odd dims

Superficially, this matches the even/odd requirement.

---

# The Subtle but Breaking Error

The issue was **not the masking**.

The issue was this:

> **The sine and cosine values for a dimension pair must share the same frequency.**

In the correct formulation:

| Dimension | Function | Frequency |
| --------- | -------- | --------- |
| 0         | sin      | ω₀        |
| 1         | cos      | ω₀        |
| 2         | sin      | ω₁        |
| 3         | cos      | ω₁        |

Each pair `(2i, 2i+1)` shares the same frequency.

But my implementation implicitly did:

| Dimension | Function | Frequency |
| --------- | -------- | --------- |
| 0         | sin      | ω₀        |
| 1         | cos      | ω₁ ❌      |
| 2         | sin      | ω₂        |
| 3         | cos      | ω₃ ❌      |

The cosine frequency was shifted relative to the sine.

This destroys the intended structure.

---

# What This Breaks Mathematically

## 1. Loss of Rotational Structure

Correct embedding for one frequency:

$$
v(pos) =
\begin{bmatrix}
\sin(\omega * pos) \
\cos(\omega * pos)
\end{bmatrix}
$$

Property:

$$
\sin^2 + \cos^2 = 1
$$

This forms a unit circle → a pure rotation as position increases.

In my incorrect version:

$$
u(pos) =
\begin{bmatrix}
\sin(\omega_a * pos) \
\cos(\omega_b * pos)
\end{bmatrix}
$$

Now:

$$
\sin^2(\omega_a * pos) + \cos^2(\omega_b * pos) \neq 1
$$

The vector:

* changes magnitude
* no longer rotates cleanly
* traces distorted Lissajous curves

The geometric interpretation collapses.

---

## 2. Loss of Relative Position Encoding via Dot Products

Correct formulation:

$$v(p) \cdot v(q) $$
$$\sin(\omega p)\sin(\omega q) + \cos(\omega p)\cos(\omega q) $$
$$\cos(\omega (p - q))$$


This is critical.

The dot product depends **only on relative distance (p - q)**.

This is why attention layers can infer distance.

In my incorrect implementation:

$$
\sin(\omega_a p)\sin(\omega_a q)
+
\cos(\omega_b p)\cos(\omega_b q)
$$

There is no identity that reduces this to a function of `(p - q)`.

This breaks:

* relative position encoding
* translation equivariance
* distance-aware attention scoring

---

## 3. No Linear Shift Operator

Correct embeddings satisfy:

$$PE(p + \Delta) = R(\Delta) PE(p)$$

Where $(R(\Delta))$ is a rotation matrix.

This elegant linear structure disappears when sine and cosine frequencies are mismatched.

The model can no longer represent position shifts as simple rotations.

---

# Why This Is Subtle

* The output tensor shape is correct
* No runtime error occurs
* The model still trains
* Loss still decreases

But the fundamental geometric property is broken.

This is a **silent correctness bug**, not a syntactic one.

---

# Conceptual Learnings

## 1. Each Dimension Pair Is a Rotating Clock

The clean mental model:

* One frequency = one clock
* sin & cos = x and y coordinates of that clock hand
* Position = angle of rotation

Never mix clocks inside a coordinate pair.

---

## 2. Geometry Matters in ML

Sinusoidal embeddings are not arbitrary feature tricks.

They encode:

* rotational structure
* constant norm
* orthogonality
* relative distance via cosine identities

Breaking frequency pairing destroys all of this.

---

## 3. Implementation Fidelity Matters

A change that:

* looks algebraically similar
* preserves shapes
* preserves types

…can still fundamentally alter the mathematical invariants the architecture relies on.

Understanding the *why* behind the formula is more important than reproducing it.

---

## 4. Correct Invariant to Remember

For every frequency $( \omega_i )$: $(\sin(\omega_i pos), \cos(\omega_i pos))$ must always be treated as a single 2D rotating vector.

---

# Final Takeaway

The mistake was subtle but structurally destructive.

Matching sine and cosine frequencies per dimension pair is not cosmetic —
it is what gives sinusoidal positional embeddings:

* rotational geometry
* relative distance encoding
* clean dot-product structure
* extrapolation ability

This exercise reinforced that deep learning architectures often rely on hidden mathematical invariants that are easy to violate if not deeply understood.

---
