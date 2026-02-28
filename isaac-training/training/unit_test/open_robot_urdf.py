#!/usr/bin/env python3
"""
Open Isaac Sim window.

Run:
    conda activate NavRL
    cd /home/mint/rl_dev/NavRL/isaac-training
    python training/unit_test/open_robot_urdf.py
    python training/unit_test/open_robot_urdf.py --headless
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Open Isaac Sim window")
    parser.add_argument("--headless", action="store_true", help="Run without UI")
    args = parser.parse_args()

    from omni.isaac.kit import SimulationApp

    sim_app = SimulationApp(
        {
            "headless": args.headless,
            "width": 1280,
            "height": 720,
            "anti_aliasing": 1,
        }
    )

    import omni.usd

    # Create a new empty stage
    ctx = omni.usd.get_context()
    ctx.new_stage()

    print("[INFO] Isaac Sim window opened successfully.")
    print("[INFO] Close the window or press Ctrl+C to exit.")

    try:
        while sim_app.is_running():
            sim_app.update()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    sim_app.close()
    print("[INFO] Isaac Sim closed.")


if __name__ == "__main__":
    main()
