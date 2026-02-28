#!/usr/bin/env python3
"""
Open taslab_uav.usdc in Isaac Sim.

Run:
    conda activate NavRL
    cd /home/mint/rl_dev/NavRL/isaac-training
    python training/unit_test/open_taslab_uav.py
    python training/unit_test/open_taslab_uav.py --headless
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without UI")
    args = parser.parse_args()

    from omni.isaac.kit import SimulationApp
    sim_app = SimulationApp({
        "headless": args.headless,
        "width": 1280,
        "height": 720,
        "anti_aliasing": 1,
    })

    import omni.usd

    usd_path = "/home/mint/rl_dev/NavRL/isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/taslab_uav.usd"

    ctx = omni.usd.get_context()
    ctx.open_stage(usd_path)

    print(f"[INFO] Opened USD: {usd_path}")
    print("[INFO] Close the window or press Ctrl+C to exit.")

    try:
        while sim_app.is_running():
            sim_app.update()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    sim_app.close()


if __name__ == "__main__":
    main()
