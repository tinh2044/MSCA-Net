import subprocess
import sys


def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✓ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package_name}: {e}")
        return False


def main():
    print("Installing packages required for FLOPs calculation...")
    print("=" * 50)

    packages = [
        "thop",  # For FLOPs calculation
        "torchprofile",  # Alternative FLOPs calculation tool
        "ptflops",  # Another alternative
    ]

    success_count = 0
    for package in packages:
        print(f"Installing {package}...")
        if install_package(package):
            success_count += 1
        print()

    print("=" * 50)
    print(
        f"Installation completed: {success_count}/{len(packages)} packages installed successfully"
    )

    if success_count > 0:
        print("\nYou can now run FLOPs calculation using:")
        print("python calculate_flops.py --cfg_path configs/phoenix-2014.yaml")

    print("\nNote: 'thop' is the primary package used for FLOPs calculation.")
    print("The other packages are alternatives in case thop doesn't work.")


if __name__ == "__main__":
    main()
