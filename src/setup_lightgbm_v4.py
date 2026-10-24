"""Install an explicitly authorized, checksum-verified venv-local OpenMP runtime.

Python wheels are installed separately with the project's .venv pip. This helper
does not run Homebrew or modify system paths/Python. The pinned official Sonoma
x86_64 bottle is compatible with this project's Intel macOS host; current upstream
libomp bottles no longer supply Intel macOS builds.
"""

import hashlib
import io
import json
import platform
import argparse
import subprocess
import sys
import sysconfig
import tarfile
import urllib.request
from pathlib import Path


VERSION = "20.1.3"
FORMULA_COMMIT = "43450a2c584a253cba3c160b74b76600dd8da43b"
BOTTLE_SHA256 = "a2eff1984d704ae5b07ac1620268cf81f98221cf5f93ec71c5e7efabe9e42f36"
BOTTLE_URL = f"https://ghcr.io/v2/homebrew/core/libomp/blobs/sha256:{BOTTLE_SHA256}"
FORMULA_URL = f"https://raw.githubusercontent.com/Homebrew/homebrew-core/{FORMULA_COMMIT}/Formula/lib/libomp.rb"


def get(url, headers=None):
    request = urllib.request.Request(url, headers={"User-Agent": "sweettv-project-venv-runtime", **(headers or {})})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configure-loader", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    venv = root / ".venv"
    if Path(sys.prefix).resolve() != venv.resolve() or platform.system() != "Darwin" or platform.machine() != "x86_64":
        raise RuntimeError("This pinned installer requires the project's Intel macOS virtual environment.")
    manifest_path = venv / "libomp_install_v4.json"
    library = venv / "lib/libomp.dylib"
    if args.configure_loader:
        manifest = json.loads(manifest_path.read_text())
        if hashlib.sha256(library.read_bytes()).hexdigest() != manifest["library_sha256"]:
            raise RuntimeError("The authorized local runtime changed.")
        native = Path(sysconfig.get_path("purelib")) / "lightgbm/lib/lib_lightgbm.dylib"
        if venv.resolve() not in native.resolve().parents:
            raise RuntimeError("Refusing to configure a library outside .venv.")
        backup = venv / "lib/lightgbm-v4-original.dylib"
        if backup.exists():
            raise FileExistsError("Preserving an existing loader configuration/backup.")
        original = native.read_bytes()
        backup.write_bytes(original)
        subprocess.run(["/usr/bin/install_name_tool", "-add_rpath", str(venv / "lib"), str(native)], check=True)
        subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", str(native)], check=True)
        manifest["venv_local_loader_configuration"] = {
            "library": str(native.relative_to(root)), "added_rpath": str(venv / "lib"),
            "original_library_sha256": hashlib.sha256(original).hexdigest(),
            "configured_library_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
            "original_backup": str(backup.relative_to(root)), "signature": "local ad-hoc signature after Mach-O relocation",
            "all_modified_files_inside_venv": True,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest["venv_local_loader_configuration"], indent=2))
        return
    if manifest_path.exists() or library.exists():
        raise FileExistsError("Preserving an existing local runtime installation.")
    formula = get(FORMULA_URL).decode()
    if BOTTLE_SHA256 not in formula or f"openmp-{VERSION}" not in formula:
        raise RuntimeError("Pinned bottle checksum/version not found in the official formula.")
    token_info = json.loads(get("https://ghcr.io/token?service=ghcr.io&scope=repository:homebrew/core/libomp:pull"))
    archive_bytes = get(BOTTLE_URL, {"Authorization": f"Bearer {token_info['token']}"})
    if hashlib.sha256(archive_bytes).hexdigest() != BOTTLE_SHA256:
        raise RuntimeError("Official libomp bottle checksum mismatch.")
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile() and member.name.endswith("/lib/libomp.dylib")]
        if len(members) != 1:
            raise RuntimeError("Expected exactly one real libomp.dylib in the official bottle.")
        content = archive.extractfile(members[0]).read()
        with library.open("xb") as handle:
            handle.write(content)
        license_members = [member for member in archive.getmembers() if member.isfile() and Path(member.name).name.upper() in ("LICENSE.TXT", "LICENSE")]
        licenses = []
        if license_members:
            directory = venv / "share/licenses/libomp-v4"
            directory.mkdir(parents=True, exist_ok=True)
            for index, member in enumerate(license_members):
                target = directory / f"LICENSE-{index + 1}.txt"
                target.write_bytes(archive.extractfile(member).read())
                licenses.append(str(target.relative_to(venv)))
    manifest = {
        "version": VERSION, "architecture": "x86_64", "bottle_tag": "sonoma", "formula_url": FORMULA_URL,
        "bottle_url": BOTTLE_URL, "bottle_sha256": BOTTLE_SHA256,
        "installed_library": str(library.relative_to(root)), "library_sha256": hashlib.sha256(content).hexdigest(),
        "licenses": licenses, "system_level_changes": False,
        "authorization": "User explicitly selected a venv-local native runtime after the missing-libomp error was reported.",
    }
    with manifest_path.open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
