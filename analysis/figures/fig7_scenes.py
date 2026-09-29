"""Figure 7: ten multi-asset scenes, four recorded frames of one successful episode each.

Left column: scenes in which the trained round-2 policy completed held-out episodes (frames from its first passed held-out
episode). Right column: scenes it never completed on held-out states (frames from the first passed scripted-teacher episode of
collect20 worker 0). Frames are the policy's third-person observation (workspace crop of the 1280x960 render, 320x240), taken at
recorded event ticks: start, first sub-goal reached (transfer scenes: object lift apex), object lift apex (transfer scenes:
gripper release), final state. Counts are the round-2 held-out evaluation (10 episodes per scene). No frame is edited.
Layout: the four frames of a scene abut (no gap); rows are separated only by the 10 pt label strip; a 4 pt gutter keeps
the two scene columns apart. The per-scene counts are not printed in the label strips (the main text reports no
trained-policy success numbers); they remain in the JSON receipt and in the appendix.
Inputs: scenes/fig7_export/manifest.json and the recorded frames scenes/fig7_export/<scene>/obs_<tick>.png (not included
in analysis/results; exported from the scene study's episode directories, see README.md)."""

import json
from pathlib import Path
import argparse

_AP = argparse.ArgumentParser(description="Figure 7 (composed scenes)")
_AP.add_argument(
    "--inputs",
    default=str(Path(__file__).resolve().parents[1] / "results"),
    help="recorded inputs (layout of analysis/results)",
)
_AP.add_argument(
    "--out",
    default=str(Path(__file__).resolve().parents[1] / "out"),
    help="output directory (tables/, figures/, facts/, sections/)",
)
_A = _AP.parse_args()
IN = Path(_A.inputs)
OUT = Path(_A.out)
import style

style.OUT = OUT
from style import Canvas, INK, TEAL, GRAY, LIGHT, PALE, RED, sha

EXP = IN / "scenes/fig7_export"
M = json.load(open(EXP / "manifest.json"))
LEFT = ["cabinet1_box", "drawer_cup", "drawer_to_cabinet", "microwave_box", "drawer_to_fridge"]
RIGHT = ["cabinet_cup", "fridge_cup", "drawer_two", "laptop_cup", "trashcan_box"]
LABEL = {
    "cabinet1_box": "Cabinet (1 door) + box",
    "drawer_cup": "Drawer + cup",
    "drawer_to_cabinet": "Drawer → cabinet, close",
    "microwave_box": "Microwave + box",
    "drawer_to_fridge": "Drawer → fridge, close",
    "cabinet_cup": "Cabinet (2 doors) + cup",
    "fridge_cup": "Fridge + cup",
    "drawer_two": "Drawer + two cups",
    "laptop_cup": "Close laptop, cup on top",
    "trashcan_box": "Trash can + box",
}


def counts(sc):
    c = M["counts"][sc]
    task = c["heldout_trained_vla_passed"]
    s1 = c["heldout_trained_vla_stage1"]
    return task, s1


def main(name="f07_scenes_dense"):
    gutter = 4
    colw = (396 - gutter) / 2
    gap = 0
    fw = colw / 4
    fh = fw * 3 / 4
    head = 10
    pitch = fh + head
    c = Canvas(round(5 * pitch + 0.5))
    facts = []
    for col, scenes in enumerate((LEFT, RIGHT)):
        x0 = col * (colw + gutter)
        for row, sc in enumerate(scenes):
            rec = M["scenes"][sc]
            top = c.h - row * pitch
            y = top - head - fh
            c.text(x0 + 1, top - head / 2 - 0.3, LABEL[sc], 8.6, INK, "bold")
            task, s1 = counts(sc)  # no success counts in the figure; they stay in the receipt and the appendix
            for j, t in enumerate(rec["event_frames"]):
                p = EXP / sc / f"obs_{t:05d}.png"
                c.image(p, x0 + j * (fw + gap), y, fw, fh)
            facts.append(
                dict(
                    scene=sc,
                    label=LABEL[sc],
                    source=rec["source"],
                    episode_dir=rec["episode_dir"],
                    episode_index=rec["episode_index"],
                    seed=rec["seed"],
                    ticks=rec["ticks"],
                    status=rec["status"],
                    goals_ok=rec["goals_ok"],
                    event_frames=rec["event_frames"],
                    event_kinds=rec["event_kinds"],
                    heldout_task=task,
                    heldout_stage1=s1,
                    trace_sha256=rec["trace_sha256"],
                    episode_json_sha256=rec["episode_json_sha256"],
                )
            )
    meta = c.save(
        name,
        "figures",
        dict(
            rows=facts,
            frame_source="policy third-person observation (workspace crop, 320x240)",
            frames_per_scene=4,
            scenes=10,
            all_shown_episodes_passed=all(f["status"] == "passed" for f in facts),
            policy_rows=sum(f["source"].startswith("policy") for f in facts),
            teacher_rows=sum(f["source"].startswith("teacher") for f in facts),
            counts_source="scenes/round2/eval_summary.json (held-out seed 2600, trained_vla, 10 episodes per scene)",
            heldout_totals=dict(task="11/100", stage1="66/100"),
            frames_edited=False,
            text_moved_to_caption=["policy / teacher tags"],
            counts_printed=False,
            frame_gap_pt=gap,
            column_gutter_pt=gutter,
            label_strip_pt=head,
        ),
    )
    print(json.dumps({k: meta[k] for k in ("name", "width_pt", "height_pt", "minimum_text_pt", "out_of_canvas_text")}))


if __name__ == "__main__":
    main()
