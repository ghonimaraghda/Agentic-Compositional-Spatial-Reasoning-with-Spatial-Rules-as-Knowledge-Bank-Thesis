# file to generate the sparp_relation_level_f1.png figure in the paper,
# using the per-label F1 scores computed from the JSON result files

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Original SpaRP-style axis names and order
ORDER = [
    "DC", "below", "above", "right", "left", "behind", "front", "far",
    "near", "NTPPI", "TPPI", "NTPP", "TPP", "EQ", "PO", "EC"
]

# Map original-style labels to your dataset labels
LABEL_MAP = {
    "DC": "outside",
    "EC": "outside and touching",
    "PO": "partially overlapping",
    "EQ": None,  # not available in your dataset
    "TPP": "inside and touching",
    "NTPP": "inside",
    "TPPI": "contains and touches",
    "NTPPI": "contains",
    "near": "near",
    "far": "far",
    "front": "in front",
    "behind": "behind",
    "left": "left",
    "right": "right",
    "above": "above",
    "below": "below",
}

# Updated result file names
FILES = {
    "Qwen3.5-27B": {
        "P1": "results/spartun_P1_Qwen3.5-27B.json",
        "P2": "results/spartun_P2_Qwen3.5-27B.json",
    },
    "Llama-3-70B": {
        "P1": "results/spartun_P1_Llama-3-70B-Instruct.json",
        "P2": "results/spartun_P2_Meta-Llama-3-70B-Instruct.json",
    },
    "Qwen2.5-72B": {
        "P1": "results/spartun_P1_Qwen2.5-72B-Instruct.json",
        "P2": "results/spartun_P2_Qwen2.5-72B-Instruct.json",
    },
}

ALL_DATA_LABELS = [
    "left", "right", "above", "below", "behind", "in front",
    "near", "far", "outside", "outside and touching",
    "partially overlapping", "inside and touching",
    "inside", "contains and touches", "contains", "overlapping"
]


def compute_per_label_f1(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores = {}

    for label in ALL_DATA_LABELS:
        tp = fp = fn = 0

        for ex in data["results"]:
            gold = set(ex.get("gold", []))
            pred = set(ex.get("predicted", []))

            if label in gold and label in pred:
                tp += 1
            elif label not in gold and label in pred:
                fp += 1
            elif label in gold and label not in pred:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        scores[label] = f1 * 100

    return scores


def map_scores_to_original_axes(scores):
    mapped = []

    for axis_name in ORDER:
        mapped_label = LABEL_MAP[axis_name]

        if axis_name == "PO":
            # Merge both overlap-style labels by taking the stronger score
            po_score = scores.get("partially overlapping", 0.0)
            overlap_score = scores.get("overlapping", 0.0)
            mapped.append(max(po_score, overlap_score))
        elif mapped_label is None:
            # EQ is not available in the dataset
            mapped.append(0.0)
        else:
            mapped.append(scores.get(mapped_label, 0.0))

    return mapped


def main():
    # Check that all result files exist before plotting
    for model_name, model_files in FILES.items():
        for prompt_name, file_path in model_files.items():
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Missing file for {model_name} {prompt_name}: {file_path}"
                )

    angles = np.linspace(0, 2 * np.pi, len(ORDER), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 5),
        subplot_kw=dict(polar=True)
    )

    for ax, (model_name, model_files) in zip(axes, FILES.items()):
        for prompt_name, file_path in model_files.items():
            raw_scores = compute_per_label_f1(file_path)
            values = map_scores_to_original_axes(raw_scores)
            values += values[:1]

            ax.plot(angles, values, linewidth=2, label=prompt_name)
            ax.fill(angles, values, alpha=0.08)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(ORDER, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80])
        ax.set_yticklabels(["20", "40", "60", "80"], fontsize=7)
        ax.set_title(model_name, fontsize=12, pad=18)
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10)

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    os.makedirs("Images", exist_ok=True)
    output_path = "Images/sparp_relation_level_f1.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()