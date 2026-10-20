"""Downloads the open-access papers this project compares against into this folder.
Run on a machine with internet access:   python download_papers.py
Files that already exist are skipped. Nothing here was downloaded or checked by the assistant (its sandbox has no access to arXiv);
if a link has moved, the arXiv id in the table is the stable handle."""
import sys, time, urllib.request
from pathlib import Path
PAPERS = [
 ("2104.11846", "Boyaci_GNN_FDIA_detection_localization",   "Boyaci et al., joint detection and localization of stealth FDIA with graph neural networks (IEEE TSG)"),
 ("2102.09057", "Li_adversarial_resilient_DNN_FDIA",         "Li et al., adversarial-resilient DNNs for FDIA detection"),
 ("2301.12487", "Riva_random_padding_FDIA",                  "Riva et al., mitigating adversarial effects of FDIA"),
 ("2207.10805", "Yin_PowerFDNet",                            "Yin et al., PowerFDNet (uses SimBench grids)"),
 ("2003.02229", "Wang_autoencoder_FDIA",                     "Wang et al., FDIA detection with an autoencoder"),
 ("2005.07158", "Wang_AE_training_strategies",               "Wang et al., training strategies for autoencoder-based FDIA detection"),
 ("1808.01094", "Niu_dynamic_FDIA_deep_learning",            "Niu et al., dynamic FDIA detection with deep learning"),
 ("1503.06468", "Ozay_ML_attack_detection_smart_grid",       "Ozay et al., machine learning methods for attack detection in the smart grid"),
 ("2606.15005", "Abdulin_Transformer_MoE_FDIA",              "Abdulin and Narimani, transformer mixture-of-experts for FDIA"),
 ("2008.06926", "Sayghe_survey_ML_FDIA",                     "Sayghe et al., survey of ML methods for FDIA detection"),
 ("1607.06015", "Tang_FDIA_colored_noise_GLRT",              "Tang, Yan, Kay, He, FDIA detection under colored Gaussian noise (GLRT)"),
 ("1708.07810", "Sun_information_theoretic_attacks",         "Sun et al., information-theoretic attacks in the smart grid"),
 ("1811.05423", "Zhang_Wang_quickest_detection_FDIA",        "Zhang and Wang, quickest detection of time-varying FDIA"),
 ("1604.00118", "Sanjab_Saad_game_theory_FDIA",              "Sanjab and Saad, data injection attacks with multiple adversaries"),
 ("1310.7616",  "Kim_Tong_Thomas_data_framing",              "Kim, Tong, Thomas, data framing attack on state estimation"),
 ("1502.04254", "Ozay_sparse_attack_construction",           "Ozay et al., sparse attack construction and state estimation"),
 ("2602.10162", "Xiao_Weng_manifold_aligned_FDIA",           "Xiao and Weng, manifold-aligned attacks against residual-based detection"),
]
EXTRA = [("https://techrep.csc.ncsu.edu/2009/TR-2009-5.pdf", "Liu_Ning_Reiter_FDIA_TR2009", "Liu, Ning, Reiter, false data injection attacks against state estimation (NCSU tech report; journal: ACM TISSEC 14(1), 2011)")]
here = Path(__file__).resolve().parent; ok = bad = 0
jobs = [(f"https://arxiv.org/pdf/{i}", n, d) for i, n, d in PAPERS] + EXTRA
for url, name, desc in jobs:
    out = here / f"{name}.pdf"
    if out.exists() and out.stat().st_size > 10000: print("skip   ", name); continue
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research download script)"})
        data = urllib.request.urlopen(req, timeout=60).read()
        if not data.startswith(b"%PDF"): raise ValueError("not a PDF")
        out.write_bytes(data); ok += 1; print("saved  ", name, f"{len(data)/1e6:.1f} MB")
    except Exception as ex:
        bad += 1; print("FAILED ", name, "-", type(ex).__name__, str(ex)[:80], "| open", url, "in a browser")
    time.sleep(3)          # be polite to arXiv
(here / "INDEX.md").write_text("# Papers\n\n" + "\n".join(f"- `{n}.pdf`: {d} ({u})" for u, n, d in jobs) + "\n")
print(f"done: {ok} saved, {bad} failed; index written to INDEX.md")
