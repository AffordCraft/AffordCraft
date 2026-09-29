"""Scene specifications for the AffordCraft multi-asset manipulation suite (pure python, no Isaac).

Each scene composes >=2 AffordCraft-produced assets (library entries selected and adapted by the
method's campaign) whose task requires physical interaction between them (containment, support,
articulation). Positions are in the table frame: table top z=0.9 m, Franka base at (-0.66, 0, 1.007).
Articulated assets are placed with their front facing the robot (-x) unless stated. `plan` is the
scripted teacher; `goals` the independent evaluator's sequential sub-goals.

Paths: the scene run root is $AFFORDCRAFT_RUNS/scenes_vla (baked assets in its assets/ folder). The
runs resolve candidate ids in the campaign asset inventory first and in the asset pool second; both ship
in data/ with their USD paths as ${AFFORDCRAFT_RUNS} placeholders."""

import json, os

_RUN = os.path.join(os.environ.get("AFFORDCRAFT_RUNS", "runs"), "scenes_vla")
POOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "asset_pool_013.json")
INVENTORY = os.environ.get(
    "AFFORDCRAFT_SCENE_INVENTORY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "asset_inventory_013.json")
)
BAKED_DIR = os.path.join(_RUN, "assets")
TABLE_Z = 0.9
BASE = (-0.66, 0.0, 1.007)
# rigid graspables (campaign-built, verified physical_pass); widths <= 0.065 m across the grasp axis
RIGID = {
    "bottle_slim": dict(cid="objaverse_59fec10a91a247118ded6e2fdb906946", label="bottle"),
    "bottle_mustard": dict(cid="ycb_mustard_bottle", label="bottle"),
    "bottle_wide": dict(cid="objaverse_d78c0bfaf7e2448cbd98a485ab079ad2", label="bottle"),
    "cup_a": dict(cid="objaverse_885ae84e85ef401ba0f7a07a81adbb06", label="cup"),
    "cup_b": dict(cid="objaverse_fae08c03cc75437c811f50426a81ac30", label="cup"),
    "cup_small": dict(cid="objaverse_29119fd73cf44d70a4f038849a25dfb5", label="cup"),
    "vase": dict(cid="objaverse_7370199a9ce64332ad9f76a0e14273ca", label="vase"),
    "pen": dict(cid="objaverse_42e02b8cb8884ce09380943e8dd90372", label="pen"),
    "mouse": dict(cid="objaverse_43de4d030ae94667bd8b8a478dd371ac", label="mouse"),
    "book": dict(cid="objaverse_c61227cac7224b86b43c53ac2a2b6ec7", label="book"),
    "bucket": dict(cid="100446", label="bucket"),
    "box_bar": dict(cid="objaverse_160d40d975f94edfa5eb6e6f6a4201ee", label="box"),
}
DISTRACTOR_KEYS = ["cup_small", "vase", "pen", "mouse", "book", "box_bar", "bottle_wide"]
# articulated furniture/appliances with the baked scene scale (PhysX rejects Xform scale over articulations)
ARTICULATED = {
    "cabinet_2door": dict(cid="46277", scale=0.6, friction=0.3, facing="handle"),
    "cabinet_1door": dict(cid="45623", scale=1.0, friction=0.3, facing="handle"),
    "drawer_unit": dict(cid="46130", scale=0.35, friction=0.3, facing="handle"),
    "microwave": dict(cid="7310", scale=1.0, friction=0.3, facing="handle", platform=0.10),
    "fridge": dict(cid="10849", scale=0.8, friction=0.3, facing="handle", clamp_deg=(0.0, 110.0)),
    "laptop": dict(
        cid="11586", scale=0.8, friction=1.0, damping=0.1, facing="hinge_front", clamp_deg=(-85.0, 0.0), press=False
    ),
    "oven": dict(cid="7290", scale=0.8, friction=15.0, facing="handle", platform=0.22),
    "dishwasher": dict(cid="12605", scale=0.4, friction=15.0, facing="handle", platform=0.12),
    "trashcan": dict(cid="11124", scale=1.0, friction=1.0, facing="hinge_back", mu=0.8, clamp_deg=(0.0, 100.0)),
}
# handle bar geometry from the PartNet part meshes (measured offline by a handle_geom.py helper that is not part of this
# release): g = bar centre to door panel, thick = bar thickness along the door normal, measured at `at_scale`
HANDLE_GEOM = {
    "46277": dict(g=0.0217, thick=0.006, at_scale=0.6),
    "7310": dict(g=0.0517, thick=0.0293, at_scale=0.8),
    "45623": dict(g=0.0272, thick=0.0131, at_scale=0.5),
    "7290": dict(g=0.0723, thick=0.0333, at_scale=0.8),
    "12605": dict(g=0.0251, thick=0.0146, at_scale=0.4),
    "47632": dict(g=0.0241, thick=0.0108, at_scale=0.6),
    "45384": dict(g=0.0172, thick=0.008, at_scale=0.5),
}


def R(x0, x1, y0, y1):
    return dict(x=(x0, x1), y=(y0, y1))


SCENES = {
    # diagnostic: robot only
    "diag_empty": dict(
        instruction="diagnostic scene without assets", articulated=[], rigid=[], distractors=0, plan=[], goals=[]
    ),
    # 1. two-door cabinet, far door ajar: open it by its free edge (it swings away from the arm), put the cup on the lower shelf
    "cabinet_cup": dict(
        instruction="open the cabinet door fully and put the cup inside the cabinet",
        articulated=[
            dict(
                key="cabinet",
                asset="cabinet_2door",
                front_x=-0.16,
                y=0.12,
                yaw=180,
                jitter=0.015,
                initial={"joint_1": 30},
            )
        ],
        rigid=[dict(key="cup", asset="cup_b", region=R(-0.36, -0.26, -0.44, -0.30), align="y")],
        distractors=1,
        plan=[
            ("open_edge", "cabinet", "joint_1", 85),
            ("pick", "cup", "top", "y"),
            ("place_inside", "cabinet", "joint_1"),
            ("retreat",),
        ],
        goals=[
            dict(kind="joint_min", asset="cabinet", joint="joint_1", deg=60),
            dict(kind="inside", obj="cup", asset="cabinet", link="body", released=True, stable=True),
        ],
    ),
    # 2. single-door cabinet (ajar 20 deg, hinge on the far side): open, put the box on the shelf
    "cabinet1_box": dict(
        instruction="open the cabinet door fully and put the box inside the cabinet",
        articulated=[
            dict(
                key="cabinet",
                asset="cabinet_1door",
                front_x=-0.16,
                y=0.34,
                yaw=180,
                jitter=0.015,
                initial={"joint_0": 20},
            )
        ],
        rigid=[dict(key="box", asset="box_bar", region=R(-0.40, -0.28, -0.30, -0.16), align="y")],
        distractors=1,
        plan=[
            ("open_edge", "cabinet", "joint_0", 80),
            ("pick", "box", "top", "y"),
            ("place_inside", "cabinet", "joint_0"),
            ("retreat",),
        ],
        goals=[
            dict(kind="joint_min", asset="cabinet", joint="joint_0", deg=60),
            dict(kind="inside", obj="box", asset="cabinet", link="body", released=True, stable=True),
        ],
    ),
    # 3. mini refrigerator (ajar): open the door to 90 deg, put the cup inside
    "fridge_cup": dict(
        instruction="open the refrigerator door fully and put the cup inside the refrigerator",
        articulated=[
            dict(key="fridge", asset="fridge", front_x=-0.18, y=-0.30, yaw=180, jitter=0.015, initial={"joint_0": 25})
        ],
        rigid=[dict(key="cup", asset="cup_b", region=R(-0.36, -0.26, 0.22, 0.36), align="y")],
        distractors=1,
        plan=[
            ("open_edge", "fridge", "joint_0", 90),
            ("pick", "cup", "top", "y"),
            ("place_inside", "fridge", "joint_0"),
            ("retreat",),
        ],
        goals=[
            dict(kind="joint_min", asset="fridge", joint="joint_0", deg=70),
            dict(kind="inside", obj="cup", asset="fridge", link="body", released=True, stable=True),
        ],
    ),
    # 4. microwave on a stand (ajar): open the door, put the box inside
    "microwave_box": dict(
        instruction="open the microwave door fully and put the box inside the microwave",
        articulated=[
            dict(
                key="microwave",
                asset="microwave",
                front_x=-0.16,
                y=0.20,
                yaw=180,
                jitter=0.015,
                initial={"joint_0": 30},
            )
        ],
        rigid=[dict(key="box", asset="box_bar", region=R(-0.40, -0.28, -0.38, -0.22), align="y")],
        distractors=1,
        plan=[
            ("open_edge", "microwave", "joint_0", 85),
            ("pick", "box", "top", "y"),
            ("place_inside", "microwave", "joint_0"),
            ("retreat",),
        ],
        goals=[
            dict(kind="joint_min", asset="microwave", joint="joint_0", deg=70),
            dict(kind="inside", obj="box", asset="microwave", link="body", released=True, stable=True),
        ],
    ),
    # 5. drawer unit: pull the top drawer by its front edge, put the cup in
    "drawer_cup": dict(
        instruction="open the top drawer and put the cup inside the drawer",
        articulated=[dict(key="drawers", asset="drawer_unit", front_x=-0.14, y=0.10, yaw=180, jitter=0.02)],
        rigid=[dict(key="cup", asset="cup_b", region=R(-0.36, -0.26, -0.36, -0.22), align="y")],
        distractors=1,
        plan=[
            ("open_prismatic", "drawers", "joint_2", 0.11),
            ("pick", "cup", "top", "y"),
            ("place_inside", "drawers", "joint_2"),
            ("retreat",),
        ],
        goals=[
            dict(kind="joint_min", asset="drawers", joint="joint_2", m=0.07),
            dict(kind="inside", obj="cup", asset="drawers", link="joint_2", released=True, stable=True),
        ],
    ),
    # 6. drawer unit, two cups side by side
    "drawer_two": dict(
        instruction="open the top drawer and put both cups inside the drawer",
        articulated=[dict(key="drawers", asset="drawer_unit", front_x=-0.14, y=0.10, yaw=180, jitter=0.02)],
        rigid=[
            dict(key="cup", asset="cup_b", region=R(-0.36, -0.26, -0.44, -0.36), align="y"),
            dict(key="cup2", asset="cup_b", region=R(-0.42, -0.32, -0.28, -0.18), align="y"),
        ],
        distractors=1,
        plan=[
            ("open_prismatic", "drawers", "joint_2", 0.11),
            ("pick", "cup", "top", "y"),
            ("place_inside", "drawers", "joint_2", (0, -0.065)),
            ("retreat",),
            ("pick", "cup2", "top", "y"),
            ("place_inside", "drawers", "joint_2", (0, 0.065)),
            ("retreat",),
        ],
        goals=[
            dict(kind="joint_min", asset="drawers", joint="joint_2", m=0.07),
            dict(kind="inside", obj="cup", asset="drawers", link="joint_2", released=True, stable=True),
            dict(kind="inside", obj="cup2", asset="drawers", link="joint_2", released=True, stable=True),
        ],
    ),
    # 7. transfer: cup out of the open drawer into the open cabinet, then close the (empty) drawer
    "drawer_to_cabinet": dict(
        instruction="take the cup out of the drawer, put it into the cabinet and close the drawer",
        articulated=[
            dict(
                key="cabinet",
                asset="cabinet_2door",
                front_x=-0.16,
                y=0.20,
                yaw=180,
                jitter=0.015,
                initial={"joint_1": 85},
            ),
            dict(
                key="drawers",
                asset="drawer_unit",
                front_x=-0.14,
                y=-0.34,
                yaw=180,
                jitter=0.015,
                initial={"joint_2": 0.11},
            ),
        ],
        rigid=[dict(key="cup", asset="cup_b", inside=("drawers", "joint_2"), align="y")],
        distractors=0,
        plan=[
            ("pick", "cup", "top", "y"),
            ("place_inside", "cabinet", "joint_1"),
            ("push_close_prismatic", "drawers", "joint_2"),
            ("retreat",),
        ],
        goals=[
            dict(kind="inside", obj="cup", asset="cabinet", link="body", released=True),
            dict(kind="joint_max", asset="drawers", joint="joint_2", m=0.03, keep="inside"),
        ],
    ),
    # 8. transfer: cup out of the open drawer into the open refrigerator, then close the drawer
    "drawer_to_fridge": dict(
        instruction="take the cup out of the drawer, put it into the refrigerator and close the drawer",
        articulated=[
            dict(key="fridge", asset="fridge", front_x=-0.18, y=-0.32, yaw=180, jitter=0.015, initial={"joint_0": 90}),
            dict(
                key="drawers",
                asset="drawer_unit",
                front_x=-0.14,
                y=0.30,
                yaw=180,
                jitter=0.015,
                initial={"joint_2": 0.11},
            ),
        ],
        rigid=[dict(key="cup", asset="cup_b", inside=("drawers", "joint_2"), align="y")],
        distractors=0,
        plan=[
            ("pick", "cup", "top", "y"),
            ("place_inside", "fridge", "joint_0"),
            ("push_close_prismatic", "drawers", "joint_2"),
            ("retreat",),
        ],
        goals=[
            dict(kind="inside", obj="cup", asset="fridge", link="body", released=True),
            dict(kind="joint_max", asset="drawers", joint="joint_2", m=0.03, keep="inside"),
        ],
    ),
    # 9. laptop: close the open screen by pushing it, then put the cup on the closed laptop
    "laptop_cup": dict(
        instruction="close the laptop and put the cup on top of the laptop",
        articulated=[dict(key="laptop", asset="laptop", front_x=-0.40, y=0.16, yaw=0, jitter=0.02)],
        rigid=[dict(key="cup", asset="cup_b", region=R(-0.34, -0.24, -0.32, -0.18), align="y")],
        distractors=1,
        plan=[
            ("push_rotate_closed", "laptop", "joint_1"),
            ("pick", "cup", "top", "y"),
            ("place_on_top", "laptop", "joint_1"),
            ("retreat",),
        ],
        goals=[
            dict(kind="joint_le", asset="laptop", joint="joint_1", deg=-75),
            dict(kind="on_top", obj="cup", asset="laptop", link="any", released=True, stable=True),
        ],
    ),
    # 10. trash can: hook the lid's front overhang and lift it over the top, drop the box inside
    "trashcan_box": dict(
        instruction="open the trash can lid and put the box into the trash can",
        articulated=[dict(key="trashcan", asset="trashcan", front_x=-0.24, y=0.16, yaw=0, jitter=0.02)],
        rigid=[dict(key="box", asset="box_bar", region=R(-0.40, -0.28, -0.36, -0.20), align="x")],
        distractors=1,
        plan=[
            ("lift_lid", "trashcan", "joint_0", 95),
            ("pick", "box", "top", "x"),
            ("drop_inside", "trashcan", "joint_0"),
            ("retreat",),
        ],
        goals=[
            dict(kind="joint_min", asset="trashcan", joint="joint_0", deg=70),
            dict(kind="inside", obj="box", asset="trashcan", link="body", released=True, stable=True),
        ],
    ),
}


def asset_usd(cid):
    inv = json.load(open(INVENTORY))
    if cid in inv:
        return os.path.expandvars(inv[cid]["usd"])
    pool = json.load(open(POOL))
    return os.path.expandvars(pool[cid]["usd"])
