import os
from pathlib import Path
import shutil


def build_client(
    install_dependencies=False,
    sourcemap=False,
    reinstall_bumps=True,
    cleanup=False,
):
    """Build the refl1d webview client."""

    # check if npm is installed
    if not shutil.which("npm"):
        raise RuntimeError("npm is not installed. Please install npm.")
    client_folder = (Path(__file__).parent / "client").resolve()
    # check if the node_modules directory exists
    node_modules = client_folder / "node_modules"
    os.chdir(client_folder)
    if install_dependencies or not node_modules.exists():
        print("Installing node modules...")
        os.system("npm install")

    # install to the local version of bumps:
    def cleanup_bumps_packages():
        for bumps_package_file in client_folder.glob("bumps-webview-client*.tgz"):
            bumps_package_file.unlink()

    if reinstall_bumps:
        import bumps.webview

        # remove any old packages in client folder
        cleanup_bumps_packages()
        bumps_path = Path(bumps.webview.__file__).parent / "client"
        # pack it up for install...
        os.system(f"npm pack {bumps_path} --quiet")
        # get the package filename:
        bumps_package_file = next(client_folder.glob("bumps-webview-client*.tgz"))
        os.system(f"npm install {bumps_package_file} --no-save")

    # build the client
    print("Building the webview client...")
    cmd = f"npm run build"
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
    parser.add_argument("--reinstall-bumps", action="store_false", help="install the local version of bumps.")
    parser.add_argument("--cleanup", action="store_true", help="Remove the node_modules directory.")
    args = parser.parse_args()
    build_client(
        install_dependencies=args.install_dependencies,
        sourcemap=args.sourcemap,
        reinstall_bumps=args.reinstall_bumps,
        cleanup=args.cleanup,
    )
