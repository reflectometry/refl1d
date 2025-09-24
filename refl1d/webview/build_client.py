import os
from pathlib import Path
import shutil


def build_client(
    install_dependencies=False,
    sourcemap=False,
    reinstall_bumps=False,
    cleanup=False,
):
    """Build the refl1d webview client."""
    if shutil.which("bun"):
        tool = "bun"
    elif shutil.which("npm"):
        tool = "npm"
    else:
        raise RuntimeError("npm/bun is not installed. Please install either npm or bun.")

    client_folder = (Path(__file__).parent / "client").resolve()
    node_modules = client_folder / "node_modules"
    os.chdir(client_folder)

    if install_dependencies or not node_modules.exists():
        print("Installing node modules...")
        os.system(f"{tool} install")

    # install to the local version of bumps:
    def cleanup_bumps_packages():
        for bumps_package_file in client_folder.glob("bumps-webview-client*.tgz"):
            bumps_package_file.unlink()

    if reinstall_bumps:
        import bumps.webview

        print("Reinstalling bumps...")

        cleanup_bumps_packages()

        bumps_path = Path(bumps.webview.__file__).parent / "client"
        os.system(f"{tool} install --prefix {bumps_path}")
        shutil.copytree(bumps_path / "src", node_modules / "bumps-webview-client" / "src", dirs_exist_ok=True)
        for file in bumps_path.iterdir():
            if file.is_file():
                shutil.copy(file, node_modules / "bumps-webview-client" / file.name)

        # # pack it up for install...
        # os.system(f"npm pack {bumps_path} --quiet")
        # # get the package filename:
        # bumps_package_file = next(client_folder.glob("bumps-webview-client*.tgz"))
        # os.system(f"npm install {bumps_package_file} --no-save")

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
        print("node_modules folders removed.")

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the webview client.")
    parser.add_argument("--install-dependencies", action="store_true", help="Install dependencies.")
    parser.add_argument("--sourcemap", action="store_true", help="Generate sourcemaps.")
    parser.add_argument("--reinstall-bumps", action="store_true", help="Re-install the local version of bumps.")
    parser.add_argument("--cleanup", action="store_true", help="Remove the node_modules directory.")
    args = parser.parse_args()
    build_client(
        install_dependencies=args.install_dependencies,
        sourcemap=args.sourcemap,
        reinstall_bumps=args.reinstall_bumps,
        cleanup=args.cleanup,
    )
