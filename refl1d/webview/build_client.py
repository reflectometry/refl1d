import os
from pathlib import Path
import shutil

from bumps import webview as bumps_webview

def build_client(
    install_dependencies=False,
    sourcemap=False,
    cleanup=False,
):
    """Build the refl1d webview client."""

    def cleanup_bumps_packages():
        for bumps_package_file in client_dir.glob("bumps-webview-client*.tgz"):
            bumps_package_file.unlink()

    if shutil.which("bun"):
        tool = "bun"
    elif shutil.which("npm"):
        tool = "npm"
    else:
        raise RuntimeError("npm/bun is not installed. Please install either npm or bun.")

    client_dir = (Path(__file__).parent / "client").resolve()
    node_modules = client_dir / "node_modules"
    os.chdir(client_dir)

    if install_dependencies or not node_modules.exists():
        print("Installing node modules...")
        os.system(f"{tool} install")

    print("Reinstalling bumps...")
    cleanup_bumps_packages()

    # pack it up for install...
    bumps_dir = Path(bumps_webview.__file__).parent / "client"
    if tool == "bun":
        os.chdir(bumps_dir)
        os.system(f"bun pm pack {bumps_dir} --destination {client_dir}")
        os.chdir(client_dir)
    else:
        os.system(f"npm pack {bumps_dir} --quiet")

    # install packed library
    bumps_package_file = next(client_dir.glob("bumps-webview-client*.tgz"))
    os.system(f"{tool} install {bumps_package_file} --no-save")

    # build the client
    print("Building the webview client...")
    cmd = f"{tool} run build"
    if sourcemap:
        cmd += " -- --sourcemap"
    os.system(cmd)

    if cleanup:
        print("Cleaning up...")
        shutil.rmtree(node_modules)
        cleanup_bumps_packages()
        print("Removed node_modules folders.")

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the webview client.")
    parser.add_argument("--install-dependencies", action="store_true", help="Install dependencies.")
    parser.add_argument("--sourcemap", action="store_true", help="Generate sourcemaps.")
    parser.add_argument("--cleanup", action="store_true", help="Remove the node_modules directory.")
    args = parser.parse_args()
    build_client(
        install_dependencies=args.install_dependencies,
        sourcemap=args.sourcemap,
        cleanup=args.cleanup,
    )
