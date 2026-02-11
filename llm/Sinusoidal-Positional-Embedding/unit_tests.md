## Action Items — Unit Tests to Validate Correctness

Below is a non-exhaustive but **structurally complete** set of unit tests that validate whether the sinusoidal positional embedding implementation preserves the intended mathematical invariants.

These tests are designed to catch exactly the kind of subtle bug encountered (frequency mismatch across dimension pairs).

---

### 1️⃣ Unit Magnitude Check (Per Frequency Pair)

**Purpose:**
Ensure each `(2i, 2i+1)` dimension pair forms a unit circle.

**Invariant:**

For any position `pos` and any pair `i`:

$$PE(pos, 2i)^2 + PE(pos, 2i+1)^2 = 1$$

**Test Strategy:**

* Sample multiple positions (e.g., 0, 1, 10, 100, random large index).
* For every even dimension index:

  * Compute squared magnitude of `(dim, dim+1)`.
  * Assert closeness to 1 within tolerance.

**Why it matters:**
This immediately detects mismatched frequencies (the exact bug [previously](https://github.com/arkothiwala/TorchLeet/commit/f50b5ef3a52ed7076472b4d7ffa5d594e7f72c16) made).

---

### 2️⃣ Dot Product Relative Shift Invariance Test

**Purpose:**
Ensure dot products depend only on relative distance.

**Invariant (per frequency pair):**

$$PE(p) \cdot PE(q) = PE(p + k) \cdot PE(q + k)$$

for any constant shift `k`.

**Test Strategy:**

* Choose positions `p`, `q`, and shift `k`.
* Compute:

  * `dot1 = dot(PE(p), PE(q))`
  * `dot2 = dot(PE(p+k), PE(q+k))`
* Assert `dot1 ≈ dot2`.

**Why it matters:**
This validates correct relative position encoding.
The earlier incorrect implementation would fail this test.

---

### 3️⃣ Pairwise Dot Product Identity Test

**Purpose:**
Validate the cosine identity:

$$PE(p) \cdot PE(q) = \sum_i \cos(\omega_i (p - q))$$

**Test Strategy:**

* Compute embeddings for `p` and `q`.
* Compute dot product.
* Independently compute expected sum using known frequencies.
* Compare.

**Why it matters:**
Ensures correct frequency pairing and formulation.

---

### 4️⃣ Linear Shift Operator Test (Rotation Property)

**Purpose:**
Validate that:

$$PE(p + \Delta) = R(\Delta) PE(p)$$

for each frequency pair.

**Test Strategy (per pair):**

* Extract `(2i, 2i+1)` components.
* Compute expected rotation matrix:

  ```
  [ cos(ωΔ)  sin(ωΔ) ]
  [-sin(ωΔ)  cos(ωΔ) ]
  ```
* Verify applying rotation to `PE(p)` gives `PE(p+Δ)`.

**Why it matters:**
Confirms clean rotational geometry.

---

### 5️⃣ Frequency Pair Equality Test

**Purpose:**
Ensure even and odd dimensions share identical frequencies.

**Test Strategy:**

* Extract denominator terms used for:

  * `dim = 2i`
  * `dim = 2i+1`
* Assert frequency values are equal.

**Why it matters:**
Directly catches the original bug.

---

### 6️⃣ Constant Norm Per Pair Across Positions

**Purpose:**
Ensure magnitude does not oscillate across positions.

**Test Strategy:**

* For fixed pair `i`
* Compute magnitude across a wide range of positions.
* Assert variance is near zero.

**Why it matters:**
Detects distorted Lissajous behavior from mismatched frequencies.

---

### 7️⃣ Orthogonality Check Across Different Frequencies

**Purpose:**
Validate that different frequency pairs are linearly independent.

**Test Strategy:**

* For large random set of positions:

  * Compute embeddings
  * Compute correlation between different frequency pairs
* Assert low cross-correlation.

**Why it matters:**
Ensures basis frequencies are distinct and properly constructed.

---

### 8️⃣ Long-Sequence Stability Test

**Purpose:**
Validate extrapolation consistency.

**Test Strategy:**

* Generate embeddings for:

  * training range (e.g., 0–512)
  * extended range (e.g., 10k+)
* Ensure:

  * no numerical explosion
  * no drift in magnitude
  * consistent relative dot products

**Why it matters:**
Confirms exponential scaling is implemented correctly.

---

### 9️⃣ Batch Consistency Test

**Purpose:**
Ensure vectorized and scalar computation match.

**Test Strategy:**

* Compute `PE(pos)` individually
* Compute batch version `PE([pos])`
* Assert equality

**Why it matters:**
Prevents silent broadcasting mistakes.

---

### 1️⃣0️⃣ Device & dtype Consistency Test (If Using PyTorch)

**Purpose:**
Ensure correctness across CPU/GPU and different dtypes.

**Test Strategy:**

* Generate embeddings on:

  * CPU float32
  * GPU float32
  * (optional) float16
* Assert numerical closeness.

---

# Minimal Critical Tests (If Prioritizing)

If only a few tests are implemented, the must-have ones are:

1. ✅ Unit magnitude per pair
2. ✅ Dot product shift invariance
3. ✅ Frequency equality between dimension pairs

These alone would have caught the original mistake.

---

# Meta-Learning from This Section

These tests enforce **mathematical invariants**, not surface-level shape correctness.

Key lesson:

> If an algorithm is derived from mathematical structure, unit tests must validate the invariants — not just tensor shapes or output ranges.