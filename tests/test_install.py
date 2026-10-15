"""Exercise installer functions without downloading or starting Immich."""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = (ROOT / "install.sh").read_text()


def installer_function(name):
    match = re.search(rf"^{name} \(\) \{{\n.*?^\}}", INSTALLER, re.M | re.S)
    if match is None:
        raise AssertionError(f"Installer function not found: {name}")
    return match.group(0)


STUB = r'''#!/bin/bash
set -eu
name=$(basename "$0")
printf '%s\t' "$name" "$@" >> "$COMMAND_LOG"
printf '\n' >> "$COMMAND_LOG"
case "$name" in
    python3|python3.11|python3.12|custom-python)
        if [ "${1:-}" = -m ] && [ "${2:-}" = venv ]; then
            [ "${VENV_EXIT_CODE:-0}" = 0 ] || exit "$VENV_EXIT_CODE"
            venv=$3
            mkdir -p "$venv/bin" "$venv/lib/python3.12/site-packages/onnxruntime/capi"
            cp "$0" "$venv/bin/python"
            cat > "$venv/bin/activate" <<EOF
export VIRTUAL_ENV='$venv'
export PATH='$venv/bin':\$PATH
EOF
            touch "$venv/lib/python3.12/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-312-x86_64-linux-gnu.so"
        elif [ "${1:-}" = --version ]; then
            echo 'Python 3.11.17'
        fi
        ;;
    python)
        if [ "${1:-}" = -m ] && [ "${2:-}" = pip ]; then
            exit "${PIP_EXIT_CODE:-0}"
        fi
        printf '%s\t%s\n' "${IMMICH_HOST:-}" "${IMMICH_PORT:-}" > "$BIND_LOG"
        ;;
    uv) exit "${UV_EXIT_CODE:-0}" ;;
    patchelf) exit "${PATCHELF_EXIT_CODE:-0}" ;;
esac
'''


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="immich installer ")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source = self.base / "source"
        self.ml_source = self.source / "machine-learning"
        self.app = self.base / "app"
        self.ml_app = self.app / "machine-learning"
        self.bin = self.base / "bin"
        for folder in (self.ml_source / "ann", self.ml_source / "immich_ml", self.ml_app, self.bin):
            folder.mkdir(parents=True)
        (self.ml_source / "immich_ml/log_conf.json").write_text("{}\n")
        (self.ml_source / "immich_ml/gunicorn_conf.py").write_text("\n")
        (self.ml_source / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.12"\n')
        (self.ml_source / "uv.lock").write_text("version = 1\n")
        self.log = self.base / "commands.log"
        self.bind_log = self.base / "bind.log"
        for name in ("python", "python3", "python3.11", "python3.12", "custom-python", "pip", "pip3", "poetry", "uv", "patchelf"):
            stub = self.bin / name
            stub.write_text(STUB)
            stub.chmod(0o755)
        self.env = dict(os.environ)
        for key in ("ML_PYTHON", "VIRTUAL_ENV", "IMMICH_HOST", "IMMICH_PORT", "MACHINE_LEARNING_HOST", "MACHINE_LEARNING_PORT"):
            self.env.pop(key, None)
        self.env.update(
            PATH=f"{self.bin}:{os.environ['PATH']}",
            COMMAND_LOG=str(self.log),
            BIND_LOG=str(self.bind_log),
            SCRIPT_DIR=str(ROOT),
            REPO_TAG="v3.3.3",
            INSTALL_DIR_src=str(self.source),
            INSTALL_DIR_app=str(self.app),
            INSTALL_DIR_ml=str(self.ml_app),
            INSTALL_DIR=str(self.base),
            isCUDA="false",
            PROXY_POETRY="",
            NVM_DIR=str(self.base / "nvm"),
        )

    def run_install(self, **env):
        return subprocess.run(
            ["bash", "-c", "set -euo pipefail\n" + installer_function("install_immich_machine_learning") + "\ninstall_immich_machine_learning"],
            env={**self.env, **env}, capture_output=True, text=True,
        )

    def commands(self):
        return [line.rstrip("\t").split("\t") for line in self.log.read_text().splitlines()]

    def test_locked_accelerator_extras(self):
        for choice, extra in (("false", "cpu"), ("true", "cuda"), ("openvino", "openvino"), ("rocm", "rocm")):
            with self.subTest(choice=choice):
                self.log.write_text("")
                result = self.run_install(isCUDA=choice)
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = self.commands()
                self.assertEqual(commands[0], ["python3.12", "-m", "venv", str(self.ml_app / "venv")])
                sync = next(cmd for cmd in commands if cmd[0] == "uv")
                for flag in ("--frozen", "--active", "--no-dev", "--no-editable", "--no-install-project", "--no-python-downloads"):
                    self.assertIn(flag, sync)
                self.assertEqual(sync[sync.index("--extra") + 1], extra)
                self.assertEqual(sync[sync.index("--python") + 1], str(self.ml_app / "venv/bin/python"))
                self.assertEqual(sync[sync.index("--link-mode") + 1], "copy")
                self.assertFalse(any(cmd[0] in ("poetry", "pip3") for cmd in commands))
                patched = [cmd for cmd in commands if cmd[0] == "patchelf"]
                self.assertEqual(bool(patched), extra == "openvino")
                if patched:
                    self.assertEqual(patched[0][1], "--clear-execstack")
                self.assertTrue((self.ml_app / "immich_ml/log_conf.json").is_file())

    def test_interpreter_override(self):
        result = self.run_install(ML_PYTHON=str(self.bin / "custom-python"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.commands()[0][0], "custom-python")

    def test_missing_interpreter_fails(self):
        result = self.run_install(ML_PYTHON=str(self.bin / "missing-python"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not installed", result.stdout)
        self.assertFalse((self.ml_app / "immich_ml").exists())

    def test_proxy_and_pinned_bootstrap(self):
        proxy = "https://mirror.example.test/simple/"
        result = self.run_install(PROXY_POETRY=proxy)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(["python", "-m", "pip", "install", "uv==0.8.15", "-i", proxy], self.commands())
        sync = next(cmd for cmd in self.commands() if cmd[0] == "uv")
        self.assertEqual(sync[sync.index("--default-index") + 1], proxy)
        self.assertEqual((self.ml_source / "uv.lock").read_text(), "version = 1\n")

    def test_failures_do_not_copy_incomplete_runtime(self):
        for key in ("VENV_EXIT_CODE", "PIP_EXIT_CODE", "UV_EXIT_CODE", "PATCHELF_EXIT_CODE"):
            with self.subTest(command=key):
                result = self.run_install(isCUDA="openvino", **{key: "7"})
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertFalse((self.ml_app / "immich_ml").exists())

    def test_legacy_poetry_releases_still_work(self):
        (self.ml_source / "uv.lock").unlink()
        result = self.run_install(REPO_TAG="v1.135.3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.commands()[0][0], "python3.11")
        self.assertIn(["poetry", "install", "--no-root", "--extras", "cpu"], self.commands())
        self.assertFalse(any(cmd[0] == "uv" for cmd in self.commands()))

    def test_pre_33_uses_311_even_with_uv_lock(self):
        result = self.run_install(REPO_TAG="v3.2.4", isCUDA="openvino")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.commands()[0][0], "python3.11")
        self.assertIn(["poetry", "add", "rapidocr<3.4.2"], self.commands())
        self.assertIn(["poetry", "install", "--no-root", "--extras", "openvino"], self.commands())
        self.assertFalse(any(cmd[0] == "uv" for cmd in self.commands()))

    def test_33_requires_uv_lock(self):
        (self.ml_source / "uv.lock").unlink()
        result = self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires its upstream ML uv.lock", result.stdout)

    def test_start_script_delegates_to_upstream_module(self):
        # The rest of the old installer assumes installation paths have no
        # spaces. Only generated-launcher tests use a space-free directory.
        with tempfile.TemporaryDirectory(prefix="immich-start-") as directory:
            base = Path(directory)
            ml = base / "app/machine-learning"
            (ml / "venv/bin").mkdir(parents=True)
            (ml / "venv/bin/activate").write_text(f"export PATH='{self.bin}':$PATH\n")
            (base / "runtime.env").write_text("MACHINE_LEARNING_HOST=::1\nMACHINE_LEARNING_PORT=3010\n")
            env = {**self.env, "INSTALL_DIR": str(base), "INSTALL_DIR_app": str(base / "app"), "INSTALL_DIR_ml": str(ml)}
            result = subprocess.run(["bash", "-c", "set -euo pipefail\n" + installer_function("create_custom_start_script") + "\ncreate_custom_start_script"], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(["bash", str(ml / "start.sh")], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(["python", "-m", "immich_ml"], self.commands())
            self.assertEqual(self.bind_log.read_text().strip(), "::1\t3010")


class GeodataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="immich-geodata-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.geo = self.base / "geodata"
        self.app = self.base / "app"
        self.bin = self.base / "bin"
        for folder in (self.geo, self.app, self.bin):
            folder.mkdir()
        for name in ("cities500.zip", "admin1CodesASCII.txt", "admin2Codes.txt", "ne_10m_admin_0_countries.geojson"):
            (self.geo / name).write_text("cached\n")
        self.log = self.base / "wget.log"
        stub = self.bin / "wget"
        stub.write_text("""#!/bin/bash
set -eu
printf '%s\\n' "$*" >> "$COMMAND_LOG"
[ "${WGET_EXIT_CODE:-0}" = 0 ] || exit "$WGET_EXIT_CODE"
printf 'US\\tUSA\\t840\\tUS\\tUnited States\\n' > "$2"
""")
        stub.chmod(0o755)
        self.env = {**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}",
                    "COMMAND_LOG": str(self.log), "INSTALL_DIR_geo": str(self.geo),
                    "INSTALL_DIR_app": str(self.app), "INSTALL_DIR": str(self.base)}

    def download(self, **env):
        return subprocess.run(
            ["bash", "-c", "set -euo pipefail\n" + installer_function("download_geonames") + "\ndownload_geonames"],
            env={**self.env, **env}, capture_output=True, text=True,
        )

    def test_add_country_names_to_cached_geodata(self):
        result = self.download()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("United States", (self.geo / "countryInfo.txt").read_text())
        self.assertEqual(self.log.read_text().count("https://"), 1)
        self.assertIn("https://download.geonames.org/export/dump/countryInfo.txt", self.log.read_text())

    def test_keep_existing_country_names(self):
        (self.geo / "countryInfo.txt").write_text("existing country names\n")
        result = self.download()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.log.exists())
        self.assertEqual((self.geo / "countryInfo.txt").read_text(), "existing country names\n")

    def test_country_download_failure_is_fatal(self):
        result = self.download(WGET_EXIT_CODE="7")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertFalse((self.geo / "countryInfo.txt").exists())


class PythonVersionTests(unittest.TestCase):
    def test_release_boundary(self):
        cases = {"v1.135.3": "3.11", "v2.7.5": "3.11", "v3.2.4": "3.11", "v3.3.0-rc.1": "3.12", "v3.3.1": "3.12", "3.3.3": "3.12", "v3.12.0": "3.12", "v4.0.0": "3.12"}
        for tag, version in cases.items():
            with self.subTest(tag=tag):
                result = subprocess.run(["bash", str(ROOT / "python-version.sh"), tag], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), version)

    def test_invalid_tag_fails(self):
        for tag in ("", "null", "latest", "v3.3"):
            with self.subTest(tag=tag):
                result = subprocess.run(["bash", str(ROOT / "python-version.sh"), tag], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)


class WorkflowTests(unittest.TestCase):
    def test_base_image_lookup_validates_api_metadata(self):
        sha = "a" * 40
        dockerfile = "FROM ghcr.io/immich-app/base-server-dev:202610131110@sha256:digest AS builder"
        for filename in ("auto_build_immich.yml", "build_dependencies.yml"):
            workflow = (ROOT / ".github/workflows" / filename).read_text()
            step = workflow.split("      - name: Find base-image commit\n", 1)[1].split("      - name: Checkout base-images repo\n", 1)[0]
            code = textwrap.dedent(step.split("        run: |\n", 1)[1]).replace("${{ github.token }}", "test-token")
            self.assertIn('Authorization: Bearer', code)
            self.assertIn('curl -fsSL --retry 3', code)
            for case, body, ref, curl_exit in (
                ("valid", dockerfile, '{"object":{"sha":"' + sha + '"}}', "0"),
                ("empty Dockerfile", '{"message":"API rate limit exceeded"}', '{}', "0"),
                ("null ref", dockerfile, '{"object":{"sha":null}}', "0"),
                ("invalid SHA", dockerfile, '{"object":{"sha":"main"}}', "0"),
                ("HTTP error", dockerfile, '{}', "22"),
            ):
                with self.subTest(workflow=filename, response=case), tempfile.TemporaryDirectory(prefix="immich-api-") as directory:
                    base = Path(directory)
                    stub = base / "curl"
                    stub.write_text("""#!/bin/bash
set -eu
[ "$CURL_EXIT_CODE" = 0 ] || exit "$CURL_EXIT_CODE"
case "$*" in
    */contents/server/Dockerfile*) printf '%s\\n' "$DOCKERFILE" ;;
    */git/refs/tags/*) printf '%s\\n' "$GIT_REF" ;;
    *) exit 99 ;;
esac
""")
                    stub.chmod(0o755)
                    output = base / "github-env"
                    result = subprocess.run(["bash", "-c", code], env={**os.environ,
                        "PATH": f"{base}:{os.environ['PATH']}", "VERSION": "v3.3.3", "GITHUB_ENV": str(output),
                        "DOCKERFILE": body, "GIT_REF": ref, "CURL_EXIT_CODE": curl_exit}, capture_output=True, text=True)
                    if case == "valid":
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(output.read_text(), f"base_commit={sha}\n")
                    else:
                        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertFalse(output.exists())

    def test_workflows_use_ubuntu_python_without_global_rewrites(self):
        for filename in ("auto_build_immich.yml", "build_server.yml"):
            with self.subTest(workflow=filename):
                workflow = (ROOT / ".github/workflows" / filename).read_text()
                self.assertIn("runs-on: ubuntu-24.04", workflow)
                self.assertIn('PYTHON_VERSION=$(bash python-version.sh "$VERSION")', workflow)
                self.assertIn('"ML_PYTHON=/usr/bin/python$ML_PYTHON_VERSION"', workflow)
                self.assertIn('if [ "$PYTHON_VERSION" = 3.11 ]; then', workflow)
                self.assertNotIn("python3.11", workflow)
                self.assertNotIn("ln -sf /usr/bin/python", workflow)
                self.assertNotIn("poetry add", workflow)
                self.assertIn("Verify machine-learning runtime artifact", workflow)
                self.assertIn("Verify geodata artifact", workflow)
                self.assertIn("countryInfo.txt", workflow)
                build = workflow[workflow.index("      - name: Build immich server"):]
                self.assertIn("set -euo pipefail", build[:build.index("./install.sh")])


if __name__ == "__main__":
    unittest.main()
