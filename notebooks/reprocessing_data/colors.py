# Extracted 8-color palette (hex) from the paper's plots
PALETTE8 = [
    "#135A56",  # deep teal
    "#C66664",  # muted red
    "#81BACE",  # light blue
    "#471F32",  # deep maroon
    "#008CBE",  # cyan-blue
    "#855B8D",  # purple
    "#2F455F",  # dark blue
    "#4D8881",  # teal-gray
]

def show_palette(colors=PALETTE8, title="Paper plot palette (8 colors)"):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 1.6))
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)

    for i, c in enumerate(colors):
        ax.add_patch(Rectangle((i, 0), 1, 1, facecolor=c, edgecolor="black", linewidth=0.8))
        ax.text(i + 0.5, -0.12, c, ha="center", va="top", fontsize=9)

    ax.set_title(title, pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

def set_mpl_color_cycle(colors=PALETTE8):
    # Optional: make matplotlib default line colors match the palette
    import matplotlib as mpl
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colors)

if __name__ == "__main__":
    show_palette()
    # set_mpl_color_cycle()