import numpy as np
import matplotlib.pyplot as plt

"""
Illuminant SPDs L(λ) — realistic, neat, transparent.
Five representative spectra (Tungsten, Daylight, LED, Fluorescent, Metal‑halide)
with handcrafted shapes and minimal overlap for a clean inset figure.
"""

np.random.seed(10)

# Wavelength axis (nm)
lmbd = np.linspace(400, 700, 1200)

# Helpers

def norm01(y):
    y = y - y.min()
    m = y.max() if y.max() > 1e-12 else 1.0
    return y / m

# Physically‑inspired simplified models

def planck_like(T=3000):
    c2 = 1.4388e7  # nm*K
    lam = lmbd
    expo = np.exp(np.clip(c2 / (lam * T), 1e-6, 80.0)) - 1.0
    y = 1.0 / (np.power(lam, 5) * expo)
    y = norm01(y)
    # Slight smoothness adjustment (remove tiny tail bumps)
    y = y ** 0.9
    return y


def daylight_like(c1=455, w1=45, c2=565, w2=85, a1=0.9, a2=1.1):
    y = a1 * np.exp(-0.5 * ((lmbd - c1) / w1) ** 2) + a2 * np.exp(-0.5 * ((lmbd - c2) / w2) ** 2)
    return norm01(y)


def led_white(blue=450, wb=16, ph=560, wp=75, a_b=1.0, a_p=1.25):
    y = a_b * np.exp(-0.5 * ((lmbd - blue) / wb) ** 2) + a_p * np.exp(-0.5 * ((lmbd - ph) / wp) ** 2)
    return norm01(y)


def fluorescent(peaks=(436, 546, 611), widths=(8, 10, 10), amps=(1.0, 0.85, 0.6)):
    y = np.zeros_like(lmbd)
    for c, w, a in zip(peaks, widths, amps):
        y += a * np.exp(-0.5 * ((lmbd - c) / w) ** 2)
    return norm01(y)


def metal_halide(peaks=(420, 550, 580, 610), widths=(12, 18, 14, 12), amps=(0.9, 1.0, 0.85, 0.65)):
    y = np.zeros_like(lmbd)
    for c, w, a in zip(peaks, widths, amps):
        y += a * np.exp(-0.5 * ((lmbd - c) / w) ** 2)
    return norm01(y)

# Build 5 curves with tiny random jitter (to avoid perfect regularity)
curves = []
labels = []

# 1) Tungsten (warm)
curves.append(planck_like(3000))
labels.append('Tungsten')

# 2) Daylight
curves.append(daylight_like(c1=455 + np.random.uniform(-6, 6), c2=565 + np.random.uniform(-8, 8)))
labels.append('Daylight')

# 3) White LED
curves.append(led_white(blue=450 + np.random.uniform(-5, 5), ph=560 + np.random.uniform(-8, 8)))
labels.append('LED')

# 4) Fluorescent (discrete spikes)
curves.append(fluorescent(peaks=(436, 546 + np.random.uniform(-4, 4), 611 + np.random.uniform(-4, 4))))
labels.append('Fluorescent')

# 5) Metal‑halide
curves.append(metal_halide(peaks=(420, 550 + np.random.uniform(-5, 5), 580 + np.random.uniform(-5, 5), 610 + np.random.uniform(-4, 4))))
labels.append('Metal-halide')

# Styling: bright but elegant palette, thinner lines
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
line_w = 1.6
alpha  = 0.95

# Plot — transparent background, minimal clutter
fig, ax = plt.subplots(figsize=(2.3, 2.3), dpi=300)
for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(400, 700)
ax.set_ylim(0, 1.1)

# Slight vertical offsets to reduce crossing clutter (without changing shapes)
offsets = [0.00, 0.02, -0.02, 0.01, -0.01]
scales  = [1.00, 0.95, 0.98, 0.90, 0.92]

for i, y in enumerate(curves):
    y2 = np.clip(y * scales[i] + offsets[i], 0, None)
    ax.plot(lmbd, y2, color=colors[i], linewidth=line_w, alpha=alpha, solid_joinstyle='round', solid_capstyle='round')

# Arrow axes
axis_arrow = dict(arrowstyle='-|>', color='black', linewidth=0.9, mutation_scale=8, shrinkA=0, shrinkB=0)
ax.annotate('', xy=(700, 0), xytext=(400, 0), arrowprops=axis_arrow, clip_on=False)
ax.annotate('', xy=(400, 1.1), xytext=(400, 0), arrowprops=axis_arrow, clip_on=False)

plt.tight_layout()
plt.savefig('illuminant_spds_square.png', dpi=300, transparent=True)
plt.show()