"""
roi_labeler.py

Interactive polygon ROI labeling tool for summary images.
Import from notebook:  from scripts.roi_labeler import ROISelector, label_unlabeled
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backend_bases import MouseButton
import cv2


class ROISelector:
    """Interactive polygon ROI labeling on a summary image."""

    def __init__(self, summary_rgb_path, output_dir=None):
        self.summary_rgb_path = summary_rgb_path
        self.base = os.path.basename(summary_rgb_path).replace("_summary_rgb.npy", "")
        self.output_dir = output_dir or os.path.dirname(summary_rgb_path)

        self.summary_rgb = np.load(summary_rgb_path)
        self.H, self.W = self.summary_rgb.shape[:2]
        
        # Channel selection: 0=mean, 1=std dev, 2=correlation
        self.current_channel = 0
        self.channel_names = ["Mean", "Std Dev", "Correlation"]
        self._update_display_image()

        self.current_vertices = []
        self.instance_masks = []
        self.mask_path = os.path.join(self.output_dir, f"{self.base}_instance_masks.npy")

        if os.path.exists(self.mask_path):
            self.instance_masks = list(np.load(self.mask_path))
            print(f"  Loaded {len(self.instance_masks)} existing ROI(s)")

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.toolbar_visible = True
        self._draw()
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(self.H, 0)
        plt.draw()

    # ── Drawing ──────────────────────────────────────────────────

    def _update_display_image(self):
        """Update the display image based on current channel selection."""
        channel_img = self.summary_rgb[..., self.current_channel]
        self.display_img = (channel_img - channel_img.min()) / (np.ptp(channel_img) + 1e-8)

    def _draw(self):
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear()
        self.ax.imshow(self.display_img, cmap="gray")
        self.ax.set_title(
            f"{self.base} — {self.channel_names[self.current_channel]}\n"
            "Click: vertex | Enter: save ROI | u: undo | d: delete last | q: save & quit | n: skip\n"
            "1: Mean | 2: Std Dev | 3: Correlation | Zoom: scroll/z/x | Pan: arrows | r: reset",
            fontsize=10,
        )
        self.ax.axis("on")

        for i, m in enumerate(self.instance_masks):
            cnts, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                pts = c.squeeze()
                if pts.ndim == 2 and len(pts) >= 3:
                    self.ax.add_patch(Polygon(pts, closed=True, fill=False, edgecolor="lime", linewidth=2))
                    cx, cy = pts.mean(axis=0)
                    self.ax.text(cx, cy, str(i + 1), color="yellow", fontsize=12,
                                 ha="center", va="center",
                                 bbox=dict(facecolor="black", alpha=0.5, pad=2))

        if self.current_vertices:
            xs, ys = zip(*self.current_vertices)
            self.ax.plot(xs, ys, "ro-", markersize=6, linewidth=2)

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        plt.draw()

    # ── Events ───────────────────────────────────────────────────

    def on_click(self, event):
        if event.inaxes != self.ax or event.button != MouseButton.LEFT:
            return
        self.current_vertices.append([event.xdata, event.ydata])
        self._draw()

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        xd, yd = event.xdata, event.ydata
        scale = 0.8 if event.button == "up" else 1.25
        self.ax.set_xlim([xd - (xd - xlim[0]) * scale, xd + (xlim[1] - xd) * scale])
        self.ax.set_ylim([yd - (yd - ylim[0]) * scale, yd + (ylim[1] - yd) * scale])
        plt.draw()

    def on_key(self, event):
        # Debug: print which key was pressed
        print(f"Key pressed: '{event.key}'")
        
        # Channel switching - handle both string and numeric
        if event.key in ("1", 1):
            self.current_channel = 0
            self._update_display_image()
            self._draw()
            print("  Switched to Mean channel")
        elif event.key in ("2", 2):
            self.current_channel = 1
            self._update_display_image()
            self._draw()
            print("  Switched to Std Dev channel")
        elif event.key in ("3", 3):
            self.current_channel = 2
            self._update_display_image()
            self._draw()
            print("  Switched to Correlation channel")
        elif event.key == "enter" and len(self.current_vertices) >= 3:
            mask = np.zeros((self.H, self.W), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(self.current_vertices, dtype=np.int32)], 1)
            self.instance_masks.append(mask)
            self.current_vertices = []
            print(f"  Saved ROI {len(self.instance_masks)}")
            self._draw()
        elif event.key == "z":
            self._zoom(0.7)
        elif event.key == "x":
            self._zoom(1.3)
        elif event.key == "r":
            self.ax.set_xlim(0, self.W)
            self.ax.set_ylim(self.H, 0)
            plt.draw()
        elif event.key in ("left", "right", "up", "down"):
            self._pan(event.key)
        elif event.key == "u" and self.current_vertices:
            self.current_vertices.pop()
            self._draw()
        elif event.key == "d" and self.instance_masks:
            self.instance_masks.pop()
            print(f"  Deleted (now {len(self.instance_masks)} ROIs)")
            self._draw()
        elif event.key == "q":
            if self.instance_masks:
                np.save(self.mask_path, np.stack(self.instance_masks, axis=0))
                print(f"  Saved {len(self.instance_masks)} mask(s) -> {self.mask_path}")
            plt.close(self.fig)
        elif event.key == "n":
            plt.close(self.fig)

    def _zoom(self, factor):
        xl, yl = self.ax.get_xlim(), self.ax.get_ylim()
        xm, ym = (xl[0] + xl[1]) / 2, (yl[0] + yl[1]) / 2
        xr, yr = (xl[1] - xl[0]) * factor, (yl[1] - yl[0]) * factor
        self.ax.set_xlim(xm - xr / 2, xm + xr / 2)
        self.ax.set_ylim(ym - yr / 2, ym + yr / 2)
        plt.draw()

    def _pan(self, direction):
        """Pan the view in the direction of the arrow key.

        Convention: pressing an arrow key moves the *view window* in that
        direction, revealing image content on that side. Pressing Right
        reveals what's to the right; the image content visually scrolls left,
        matching the natural trackpad / page-scroll feel.

        ylim is stored as `(H, 0)` (matplotlib's image-mode inverted axis), so
        the down/up sign rules are inverted vs left/right.

        (Previous version of this function had all four directions inverted.)
        """
        xl, yl = self.ax.get_xlim(), self.ax.get_ylim()
        sx = abs(xl[1] - xl[0]) * 0.1
        sy = abs(yl[1] - yl[0]) * 0.1
        if direction == "right":
            self.ax.set_xlim(xl[0] + sx, xl[1] + sx)
        elif direction == "left":
            self.ax.set_xlim(xl[0] - sx, xl[1] - sx)
        elif direction == "down":
            # Reveal larger-y content (further down in image)
            self.ax.set_ylim(yl[0] + sy, yl[1] + sy)
        elif direction == "up":
            # Reveal smaller-y content (further up in image)
            self.ax.set_ylim(yl[0] - sy, yl[1] - sy)
        plt.draw()

    def run(self):
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        plt.show(block=True)


# ════════════════════════════════════════════════════════════════
#  Convenience: label all unlabeled images in a directory
# ════════════════════════════════════════════════════════════════

def label_unlabeled(training_dir):
    """Open the ROI labeler for every summary image that lacks masks."""
    summary_files = sorted(glob.glob(os.path.join(training_dir, "*_summary_rgb.npy")))
    unlabeled = [
        f for f in summary_files
        if not os.path.exists(f.replace("_summary_rgb.npy", "_instance_masks.npy"))
    ]
    print(f"{len(summary_files)} summary images total, {len(unlabeled)} unlabeled")
    for i, f in enumerate(unlabeled, 1):
        print(f"\n[{i}/{len(unlabeled)}] {os.path.basename(f)}")
        sel = ROISelector(f)
        sel.run()
    print("\nLabeling complete.")
