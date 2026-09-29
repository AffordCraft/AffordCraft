#!/usr/bin/env python
"""Re-create $AA_WORK/src from the official Articulate-Anything tree (excluding .git/__pycache__) and apply
the AffordCraft compatibility patches. Every patch is an exact-string replacement that must match exactly once; the
resulting unified diff is written to $AA_WORK/patches/official_tree.diff. Idempotent (always restarts
from the pristine official tree). The official tree itself is never modified.
Environment: ARTICULATE_ANYTHING_HOME (official checkout), ARTICULATE_ANYTHING_CKPT (partnet-mobility-v0 library),
AA_WORK (working directory)."""
import os
import shutil
import subprocess

OFFICIAL = os.path.abspath(os.environ.get("ARTICULATE_ANYTHING_HOME", "articulate-anything"))
DST = os.path.join(os.path.abspath(os.environ.get("AA_WORK", "aa_run")), "src")
DIFF = os.path.join(os.path.abspath(os.environ.get("AA_WORK", "aa_run")), "patches", "official_tree.diff")
DATASET = os.path.abspath(os.environ.get("ARTICULATE_ANYTHING_CKPT", "partnet-mobility-v0"))

PATCHES = {}


def patch(path, old, new, count=1):
    PATCHES.setdefault(path, []).append((old, new, count))


# ---- 1. VLM client: route every agent through an OpenAI-compatible endpoint (EVAL_API_BASE / EVAL_API_KEY) when AA_OPENAI_COMPAT=1 -------------
patch(
    "articulate_anything/utils/prompt_utils.py",
    "import google.generativeai as genai\n",
    "try:  # AffordCraft compat: google-generativeai/anthropic/openai SDKs are optional here (API client used instead)\n"
    "    import google.generativeai as genai\n"
    "except Exception:  # pragma: no cover\n"
    "    genai = None\n",
)
patch(
    "articulate_anything/utils/prompt_utils.py",
    "import anthropic\nfrom openai import OpenAI\n",
    "try:\n    import anthropic\nexcept Exception:  # pragma: no cover\n    anthropic = None\n"
    "try:\n    from openai import OpenAI\nexcept Exception:  # pragma: no cover\n    OpenAI = None\n",
)
patch(
    "articulate_anything/utils/prompt_utils.py",
    "def setup_vlm_model(model_name, system_instruction=None, api_key=None):\n" '    if "gpt" in model_name:\n',
    "def setup_vlm_model(model_name, system_instruction=None, api_key=None, **api_kwargs):\n"
    '    if os.environ.get("AA_OPENAI_COMPAT") == "1":\n'
    "        # AffordCraft compat: OpenAI-compatible endpoint (EVAL_API_BASE/EVAL_API_KEY from the environment) in place of\n"
    "        # google.generativeai; same prompt-part layout as GPTWrapper below. See code/api_client.py.\n"
    "        from api_client import ApiWrapper\n"
    "        return ApiWrapper(model_name=model_name, system_instruction=system_instruction, **api_kwargs)\n"
    '    if "gpt" in model_name:\n',
)
patch(
    "articulate_anything/agent/agent.py",
    "        self.model = setup_vlm_model(\n"
    "            model_name=cfg.model_name, system_instruction=self.system_instruction, api_key=cfg.api_key\n"
    "        )\n",
    "        self.model = setup_vlm_model(\n"
    "            model_name=cfg.model_name, system_instruction=self.system_instruction, api_key=cfg.api_key,\n"
    "            agent_name=self.__class__.__name__, out_dir=cfg.out_dir,  # AffordCraft compat: transcript routing only\n"
    "        )\n",
)
patch(
    "articulate_anything/agent/agent.py",
    '        # logging.info(f"Usage: {response.usage_metadata}")\n'
    "\n"
    "        self.parse_response(response, **kwargs)\n",
    '        # logging.info(f"Usage: {response.usage_metadata}")\n'
    '        if os.environ.get("AA_OPENAI_COMPAT") == "1":\n'
    "            # AffordCraft compat: gemini-2.5-flash does not always follow the JSON reply format requested by the official\n"
    "            # prompts (prose + \\\\boxed{N}, trailing commas, bare index); rewrite such replies into the canonical form the\n"
    "            # unchanged official parse_response() expects. Code replies are never touched. See code/aa_response_normalizer.py.\n"
    "            from aa_response_normalizer import normalize\n"
    "            response.text = normalize(response.text, self.__class__.__name__, out_dir=self.cfg.out_dir)\n"
    "\n"
    "        self.parse_response(response, **kwargs)\n",
)

# ---- 2. render subprocess: current interpreter instead of `conda run -n articulate-anything`; SAPIEN-3 port of the ----
# ---- official sapien_simulate.py (SAPIEN 2.2.2 segfaults in svulkan2 on driver 595/RTX 5090); per-case hydra dir ----
patch(
    "articulate_anything/utils/utils.py",
    'def make_cmd(script_path: str, conda_env: str = "articulate-anything",\n'
    "             cmd_args=[]):\n"
    "    # Construct the command\n",
    'def make_cmd(script_path: str, conda_env: str = "articulate-anything",\n'
    "             cmd_args=[]):\n"
    '    if os.environ.get("AA_OPENAI_COMPAT") == "1":\n'
    "        # AffordCraft compat: no conda env named `articulate-anything` exists on the run node; run the script with the\n"
    "        # current interpreter, substitute the SAPIEN-3 port of sapien_simulate.py (AA_RENDER_SCRIPT) and keep hydra's\n"
    "        # per-run output directory inside the case directory (AA_HYDRA_RUN_DIR).\n"
    '        if script_path.endswith("sapien_simulate.py") and os.environ.get("AA_RENDER_SCRIPT"):\n'
    '            script_path = os.environ["AA_RENDER_SCRIPT"]\n'
    "        command = [sys.executable, script_path] + list(cmd_args)\n"
    '        run_dir = os.environ.get("AA_HYDRA_RUN_DIR")\n'
    "        if run_dir:\n"
    '            command += ["hydra.run.dir=" + run_dir + "/${now:%Y%m%d-%H%M%S-%f}", "hydra.output_subdir=null"]\n'
    "        return command\n"
    "    # Construct the command\n",
)


def apply():
    os.makedirs(os.path.dirname(DIFF), exist_ok=True)
    # fresh copy of the official tree without .git, __pycache__ and datasets/partnet-mobility-v0 (the executed version
    # used an equivalent archive-mode mirror with --delete; the library symlink is re-created below)
    if os.path.isdir(DST):
        shutil.rmtree(DST)
    shutil.copytree(
        OFFICIAL,
        DST,
        symlinks=True,
        ignore=lambda d, names: [
            n
            for n in names
            if n in (".git", "__pycache__") or (os.path.basename(d) == "datasets" and n == "partnet-mobility-v0")
        ],
    )
    for rel, plist in PATCHES.items():
        p = os.path.join(DST, rel)
        s = open(p, encoding="utf-8").read()
        for old, new, count in plist:
            n = s.count(old)
            if n != count:
                raise SystemExit("patch anchor found %d times (expected %d) in %s:\n%s" % (n, count, rel, old))
            s = s.replace(old, new)
        with open(p, "w", encoding="utf-8") as f:
            f.write(s)
        print("patched", rel, len(plist), "hunk(s)")
    link = os.path.join(DST, "datasets", "partnet-mobility-v0")
    if not os.path.islink(link):
        os.symlink(DATASET, link)
    r = subprocess.run(
        ["diff", "-ruN", "--exclude=.git", "--exclude=__pycache__", "--exclude=partnet-mobility-v0", OFFICIAL, DST],
        capture_output=True,
        text=True,
    )
    with open(DIFF, "w", encoding="utf-8") as f:
        f.write(r.stdout)
    print("diff written", DIFF, "bytes", len(r.stdout))


if __name__ == "__main__":
    apply()
