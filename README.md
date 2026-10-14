# Immich with CUDA/ROCm Support in LXC (w/o Docker)

[![Build immich on new release](https://github.com/tautomer/immich-in-lxc/actions/workflows/auto_build_immich.yml/badge.svg)](https://github.com/tautomer/immich-in-lxc/actions/workflows/auto_build_immich.yml)

This fork targets Immich **3.3.x**. Builds use **Ubuntu 24.04**, its packaged
**Python 3.12**, and **uv** for machine-learning dependencies. Existing installs
using the old Python 3.11 workaround must rebuild or replace their ML venv;
changing its interpreter symlinks is not a supported upgrade.

A complete guide for installing Immich in LXC, VM, or bare-metal without Docker, but with 

- **CUDA/ROCm support for machine-learning** (if one choose so), 
- **hardware acceleration for transcoding**,
- **HEIF, RAW, and JXL support**,
- Experimental Intel iGPU/dGPU/NPU support for machine-learning (if one choose so), 
- easy and fast upgrade, and
- accessible proxy settings for PyPi and NPM registry.

## Introduction

[Immich](https://github.com/immich-app/immich) is a

> High performance self-hosted photo and video management solution

I really like Immich and its coherent experience across both mobile and web. However, the official Documents only provides Docker installation guide, which is less than ideal for a LXC user. But, in fairness to Immich, not providing a bare-metal installation guide can be justified, as it is more than a simple binary and does require some efforts to set up in current state.

**This guide is heavily inspired by another guide [Immich Native](https://github.com/arter97/immich-native), and the install script & service files are modified from the ones in that repo. KUDO to its author, arter97!** 

Compared to Immich Native, this repo additionally offers the support for CUDA-accelerated machine learning and (out-of-box) support for processing HEIF, i.e. common smart phone image format, and RAW, i.e. common fancy big camera image format, images.

### Why hardware acceleration?

> I paid for the whole Speedometer, I'm gonna use the whole Speedometer.
>
> -- Abraham Lincoln

Jokes aside, hardware acceleration really helps during importing library containing many videos, or live photos (essentially a photo and video bundle), or when one would like to switch to or test out a bigger and better machine learning model to improve smart search or face search functionality, which requires a redo of the entire indexing process. However, during current stage and foreseeable future, the heavy work of generating thumbnails will remain on using SIMD commands on CPU, and cannot be accelerated by GPU.

Lastly, by using this repo, one could reliably set up a hardware-accelerated Immich instance without much hassle. So why not.

## Immich Components


<details>
<summary>Not so important</summary>

- Immich
    - Web Server
    - Machine Learning Server
- Database
    - Redis
    - Postgresql
        - VectorChord
- System
    - Jellyfin-ffmpeg
    - Node.js
    - git
- (Optional) Reverse Proxy
    - Nginx
- (Optional) NVIDIA
    - Driver (i.e. CUDA Runtime)
    - CuDNN (Version 9)
- (Optional) AMD
    - ROCm driver (6.4.1)

As one could tell, it is a lot of works, and a lot of things to get right. However, Immich is quite resilience and will fall-back to a baseline default when hardware acceleration does not work.

For the simplicity of the guide, all the components are installed in a single LXC container. However, it is always possible to run different components in different LXC containers. As it is always a design choice.

<br>
</details>

## Host setup

<details>
<summary>Not so important</summary>

I am using `Proxmox VE 8` as the LXC host, which is based on `Debian 12`, and I have a NVIDIA GPU, with a proprietary driver (550) installed. Some others are using a N100 mini PC box with Intel Quick Sync. And all of these do not matter.

However, if possible, use an LXC or VM with `Ubuntu 24.04 LTS` as it offers an easier set-up.

<br>
</details>

## Prepare the LXC container, or whatever

First, create a LXC/VM normally. Make sure there is reasonable amount CPU and memory, because we are going to install and compile a lot of things, and it would not hurt to give it a bit more. For a CPU-only Immich server, there should be at least 8 GiB of storage, and one with nVidia GPU, at least 16 GiB storage needs to be available. However, once one starts using Immich, it will create a lot of cache (for thumbnails and low-res transcoded videos), so don't forget to resize the LXC volumes accordingly. 

Also, there is no need for a privileged container (which is not recommended in almost all scenarios), if one does not plan to mount a file system, e.g., NFS, SMB, etc., directly inside the LXC container.

The current CI and prebuilt artifacts target **Ubuntu 24.04 LTS, x86-64**.
Historical Debian 12 instructions below are not a current 3.3.x compatibility
claim: Debian 12's default Python 3.11 is too old, and Ubuntu-built native
libraries must not be assumed compatible with Debian. For a source install on
another distribution, provide a compatible Python interpreter, development
headers, system libraries, and GPU drivers yourself.

## Python and machine-learning requirements

| Immich release | ML Python/runtime policy |
| --- | --- |
| **Earlier than 3.3** (`1.x`, `2.x`, `3.0`–`3.2`) | **Python 3.11**, legacy Poetry install; existing prebuilt venvs must remain on 3.11. |
| **3.3 and later** | **Python >=3.12**, upstream uv lock; native Ubuntu 24.04 builds and new prebuilt artifacts use **3.12**. Python 3.13 is an optional compatible source-build override. |

Immich 3.3 requires **Python >=3.12**, not strictly greater than 3.12.
Upstream's v3.3.3 CPU/OpenVINO Docker images use Python 3.13, but the 3.3.x lockfiles also support Python 3.12 and
include a CPython 3.12 Linux wheel for ONNX Runtime OpenVINO. This repo uses
Ubuntu 24.04's Python 3.12 to avoid installing another interpreter or replacing
Ubuntu's system Python.

Run as **root** before a source install:

```bash
apt update
apt install -y python3.12-venv python3.12-dev patchelf
/usr/bin/python3.12 --version
```

`pre-install.sh` also installs the common venv/build dependencies and `patchelf`.
`patchelf` clears the executable-stack flag on the OpenVINO extension so it can
load on newer glibc. It is not needed for the normal CPU wheel.

References: [upstream v3.3.3 ML dependencies](https://github.com/immich-app/immich/blob/v3.3.3/machine-learning/pyproject.toml),
[upstream lockfile](https://github.com/immich-app/immich/blob/v3.3.3/machine-learning/uv.lock),
and [the community-script ML fix](https://github.com/community-scripts/ProxmoxVE/pull/17770).

- The native installer uses `python3.12` for Immich 3.3+ and `python3.11`
  for earlier versions. Set `ML_PYTHON` in `.env` to select another
  compatible, already-installed interpreter, such as `/usr/bin/python3.13`.
  The installer does not download Python. This override is for **source builds**,
  not for swapping the interpreter in a prebuilt venv.
- New 3.3+ prebuilt artifacts require **`/usr/bin/python3.12`** at runtime. They contain
  Python 3.12 native extensions and are installed at `/home/immich`.
- uv installs the selected accelerator extra from the **frozen upstream lock**,
  including the Git-sourced `immich-model` package and upstream dependency
  overrides. For 3.3+, do not add the former `numpy<2` or `rapidocr<3.4.2` Poetry pins.
- All releases before 3.3 retain the legacy Poetry/Python 3.11 installer,
  even if that upstream tag includes a uv lockfile. The old RapidOCR workaround
  remains scoped to 3.2/OpenVINO; it is not applied to 3.3+.

### Installing a version earlier than 3.3

For a legacy release, keep **Python 3.11** and select that release in `REPO_TAG`.
On Ubuntu 24.04, install the separate interpreter as root; do not change
`/usr/bin/python3` or manually alter venv symlinks:

```bash
add-apt-repository -y ppa:deadsnakes/ppa
apt update
apt install -y python3.11 python3.11-venv python3.11-dev
```

The installer selects `python3.11` automatically for tags below 3.3, and CI only
adds this PPA for those legacy builds. Old prebuilt artifacts require
`/usr/bin/python3.11`. Keep that interpreter for rollback when moving to 3.3+.

### Migrating from the old Python 3.11 workaround

If an earlier workflow or manual setup changed `/usr/bin/python3` to point at
Python 3.11, restore Ubuntu 24.04's package-managed interpreter as **root**:

```bash
apt install --reinstall python3-minimal
/usr/bin/python3 --version       # Should report 3.12 on Ubuntu 24.04
/usr/bin/python3.12 --version
```

Do not repoint `venv/bin/python*` to a new Python version. Stop the services,
back up the existing installation, and follow the source-update or prebuilt
update procedure below to create a fresh Python 3.12 environment. Keep Python
3.11 installed until you no longer need the old installation for rollback.

## Hardware-accelerated machine learning

<details>
<summary>Nvidia</summary>

Firstly, prepare a LXC with GPU available by following the guide at [another repository](https://github.com/loeeeee/loe-handbook-of-gpu-in-lxc/blob/main/src/gpu-passthrough.md) of mine. This process is referred to as NVIDIA GPU pass-through in LXC.

After finishing all of the steps in that guide, the guest OS should execute command `nvidia-smi` without any error.

The major component that Immch requires is [ONNX runtime](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirementsto), and here we are installing its dependency.

<details>
<summary>Ubuntu 24.04</summary>

For Immich machine learning support in `Ubuntu`, we need to install CuDNN and CUDA Toolkit. The default cuDNN version in apt is version 8, which is no longer supported by ONNX Runtime. Thus, we need to install the latest version 9.

The CuDNN install commands are from [official website of NVIDIA](https://developer.nvidia.com/cudnn-downloads), and should all be run as root. Also, one should check the NVIDIA website for updates.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update

apt-get -y install cudnn-cuda-12
```

In addition to the cuDNN, we also need libcublas12 things.

```bash
apt install -y libcublaslt12 libcublas12 libcurand10
```

<br>
</details>

<details>
<summary>Debian 12</summary>

For Immich machine learning support in `Debian`, we need to install CuDNN and CUDA Toolkit.

We install the entire CUDA Toolkit because install `libcublas` depends on CUDA Toolkit, and when install the toolkit, this right version of this component will be included.

The CuDNN install commands are from [official website of NVIDIA](https://developer.nvidia.com/cudnn-downloads), and should all be run as root. Also, one should check the NVIDIA website for updates.

```bash
# CuDNN part
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
add-apt-repository contrib
apt-get update
apt-get -y install cudnn
## Specified by NVIDIA, but does not seem to install anything
apt-get -y install cudnn-cuda-12

# CUDA Toolkit part
apt install -y cuda-toolkit
```

<br>
</details>

Zu easy, innit?

<br>
</details>


<details>
<summary>Intel/OpenVINO</summary>

This part is intended for users who would like to utilize Intel's OpenVINO execution provider. ([System requirement](https://docs.openvino.ai/2024/about-openvino/release-notes-openvino/system-requirements.html), [List of supported devices](https://docs.openvino.ai/2024/about-openvino/compatibility-and-support/supported-devices.html)) The document listed the support for not only Intel iGPU and dGPU, but also its NPU, which seems very cool.

Disclaimer: This part is not yet tested by the repo owner, and it is composed based on documentation. However, success have been reported ([Issue #58](https://github.com/loeeeee/immich-in-lxc/issues/58)), even though one could not see the background tasks ([Issue #62](https://github.com/loeeeee/immich-in-lxc/issues/62)). 

<details>
<summary>Moe</summary>
Firstly, prepare a LXC with proper hardware available. For iGPU user, one could use `intel_gpu_top` to see its availability.

Then, install the dependency specified by Immich for Intel.

```bash
./dep-intel.sh
```

Before running `install.sh`, set the following in the installation `.env` file described below. Run `dep-intel.sh` as root; hardware access and driver setup are separate from Python dependency installation.

```env
isCUDA=openvino
```

I know, this is ugly as hell, but whatever, it works.

Now, when installing Immich, it will be using OpenVINO as its ML backend.

<br>
</details>

<br>
</details>


<details>
<summary>Others</summary>

Since Immich depends on ONNX runtime, it is **possible** that other hardware that is not officially supported by Immich can be used to do machine learning tasks. The idea here is that installing the dependency for the hardware following [ONNX's instruction](https://onnxruntime.ai/docs/execution-providers/#summary-of-supported-execution-providers). 

Some users have also reported successful results using GPU Transcoding in Immich by following the Proxmox configurations from this video: [iGPU Transcoding In Proxmox with Jellyfin Media Center](https://www.youtube.com/watch?v=XAa_qpNmzZs) - Just avoid all the Jellyfin stuff and do the configurations on the Immich container instead. At the end, you should be able to use your iGPU Transcoding in Immich by going to needs to go to `Administration > Settings > Video Transcoding Settings > Hardware Acceleration > Acceleration API` and select `Quick Sync` to explicitly use the GPU to do the transcoding.

Good luck and have fun!

<br>
</details>


## Immich Installation

### User creation

First of all, create a Immich user, if you already done so in the above optional section, you may safely skip the following code block. The user created here will run Immich server.

```bash
adduser --shell /bin/bash --disabled-password immich --comment "Immich Mich"
# --shell changes the default shell the immich user is using. In this case it will use /bin/bash, instead of the default /bin/sh, which lacks many eye-candy
# --disabled-password skips creating password, and (sort of) only allows using su to access the user. If you need to change the password of the user, use the command: passwd immich
# --comment adds user contact info, not super useful but mandatory, probably thanks to Unix legacy.
apt install -y git
usermod -aG video,render immich
```

### Clone the repo

I have make some helper script in this repo, so all one needs to do is clone the repo. We change to the user immich so that the files we cloned will have proper permission. And, just in case one does not know, the commands are as follow.

```bash
su - immich
cd ~
git clone https://github.com/tautomer/immich-in-lxc.git
```

Additionally, it is recommend to have our working directory set to the repo's directory.

```bash
cd immich-in-lxc
```

### Install custom photo-processing library

Likely because of license issue, many libraries included by distribution package managers do not support all the image format we want, e.g., HEIF, RAW, etc. Thus, we need compile these libraries from source. It can be painful to figure out how to do this, but luckily, I have already sorted out for you.

#### Install compile tools and compile 始める


Now `exit` the immich user, as the upcoming commands should be run as `root` user.

```bash
# As root, from the cloned repository
cd /home/immich/immich-in-lxc
./pre-install.sh
```

It is just so satisfying to see the compiling log rolling down the terminal, ain't it? Look carefully at the log, though. There should not be any error. However, some warning about relink will pop up, which is normal.

### Config PostgreSQL like a champ


The following steps apply to both `Debian 12` and `Ubuntu 24.04` instances.

Enter PostgreSQL control interface

```bash
# As root
su postgres
# Now you are user:postgres
psql
```

In the psql interface, run:
```SQL
CREATE DATABASE immich;
CREATE USER immich WITH ENCRYPTED PASSWORD 'A_SEHR_SAFE_PASSWORD';
GRANT ALL PRIVILEGES ON DATABASE immich to immich;
ALTER USER immich WITH SUPERUSER;
\q
```

Note: change password, seriously.

Note: To change back to the pre-su user, `exit` should do the trick.

#### Database Migration for Existing Users (v1.133.0+)

**Note:** Starting with Immich v1.133.0, the project has migrated from pgvecto.rs to [VectorChord](https://github.com/tensorchord/VectorChord) for better performance and stability.

If you're upgrading from a version prior to v1.133.0 and have an existing Immich installation, you may need to perform a database migration. The migration from pgvecto.rs to VectorChord is automatic, but you should:

1. **Backup your database** before upgrading
2. Ensure you're upgrading from at least v1.107.2 or later

**Note:** If you have an existing `$INSTALL_DIR/runtime.env` (e.g. /home/immich/runtime.env) file with `DB_VECTOR_EXTENSION=pgvector`, you should update it to `DB_VECTOR_EXTENSION=vectorchord` for the new VectorChord extension.

For more details on the VectorChord migration, see the [official Immich v1.133.0 release notes](https://github.com/immich-app/immich/releases/tag/v1.133.0).

### Install Immich Server

The star of the show is the install script, i.e. `install.sh` in this repo. It installs or updates the current Immich instance. The Immich instance itself is stateless, thanks to its design. Thus, it is safe to delete the `app` folder that will resides inside `INSTALL_DIR` folder that we are about to config. 

Note: **DO NOT DELETE THE UPLOAD FOLDER SPECIFIED BY `UPLOAD_DIR` IN `.env`**. It stores all the user-uploaded content.

Also note: One should always do a snapshot of the media folder during the updating or installation process, just in case something goes horribly wrong.

#### The installation environment

Run the source installer as **`immich`**, never as root. From its checkout,
create and edit `.env` explicitly (or run `./install.sh` once to create it and
exit without building):

```bash
su - immich
cd /home/immich/immich-in-lxc
cp -n install.env .env
nano .env
```

| Variable | Purpose |
| --- | --- |
| `REPO_TAG` | Exact Immich tag to build, for example `v3.3.3`. Review upstream upgrade notes before changing versions. |
| `INSTALL_DIR` | Parent of `app`, `source`, `geodata`, and `runtime.env`; defaults to `/home/immich`. Use paths without spaces. |
| `UPLOAD_DIR` | Persistent media directory. Must not be inside the replaceable `app` directory. |
| `isCUDA` | `false` = CPU, `true` = CUDA, `openvino` = Intel OpenVINO, `rocm` = ROCm. GPU transcoding is configured separately. |
| `ML_PYTHON` | Optional source-build interpreter override. Empty selects `python3.12` for 3.3+ or `python3.11` for earlier releases. Install its matching venv/development packages first. |
| `PROXY_NPM` | Optional npm/pnpm registry mirror; empty uses the normal registry. |
| `PROXY_NPM_DIST` | Optional Node distribution mirror for nvm/node-gyp. |
| `PROXY_POETRY` | Historical name retained for compatibility. Sets the uv bootstrap/default index or the legacy Poetry index; empty uses PyPI. Frozen lockfile URLs and the Git dependency still require access to their original hosts. |

The `immich` user needs read/write access to `INSTALL_DIR` and `UPLOAD_DIR`.
`.env` configures installation only; `$INSTALL_DIR/runtime.env` configures the
running services. Existing `.env` files can omit `ML_PYTHON` to use the default.

CI reads the Node.js and pnpm versions from the selected Immich tag's `.nvmrc`
and `package.json`. For a source build, use the same toolchain if you have
installed Node yourself; the installer otherwise bootstraps Node through nvm.

#### Run the script

After the `.env` is properly configured, we are now ready to do the actual installation.

```bash
./install.sh
```

Note, `install.sh` should be executed as user `immich`, or the user who is going to run immich.

It should go without errors, just like ever dev says.

After several minutes, ideally, it would say,

```bash
Done. Please install the systemd services to start using Immich.
```

Lastly, we need to review and modify the `runtime.env` that is inside your specified `INSTALL_DIR` (not the runtime.env inside this repo). The default values could also work, unless you changed the `DB_PASSWORD` when installing Postgres. 

**Note:** If your `DB_PASSWORD` contains special characters (such as `$`, `!`, etc.), you must wrap the value in single quotes, e.g., `DB_PASSWORD='your$pec!alP@ss'`. This prevents shell expansion issues when the environment file is sourced. (See [issue #95](https://github.com/loeeeee/immich-in-lxc/issues/95) for details.)

For Timezones `TZ`, you can consult them in the [TZ Database Wiki](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones#List).

#### Post install script

Run `post-install.sh` as **root** from the repository. It creates the log directory and copies service files only if they do not already exist; it does not overwrite existing units. Review installed units on upgrades, especially their paths and `ExecStart`.

```bash
# As root
cd /home/immich/immich-in-lxc
./post-install.sh
```

Then, modify the `service` files to make sure every path name is spelled correctly. You might need to modify the variables `WorkingDirectory`, `EnvironmentFile`, `ExecStart` with the `INSTALL_DIR` specified in the `.env` file (in case you didn't leave the default INSTALL_DIR).

```bash
nano /etc/systemd/system/immich-ml.service # Modify WorkingDirectory, EnvironmentFile, and ExecStart with your INSTALL_DIR, in case you changed it.
```
```bash
nano /etc/systemd/system/immich-web.service # Modify ExecStart with your INSTALL_DIR, in case you changed it.
```

The generated ML launcher activates `app/machine-learning/venv` and executes
`python -m immich_ml`. Upstream owns the Gunicorn options and packaged logging
configuration. `MACHINE_LEARNING_HOST` and `MACHINE_LEARNING_PORT` in `runtime.env`
retain the native defaults `127.0.0.1:3003`; worker settings are read by upstream.

After reviewing `runtime.env` and both units, start the services:

```bash
systemctl daemon-reload && \
systemctl start immich-ml && \
systemctl start immich-web
```

The default setting exposes the Immich web server on port `2283` on all available address. For security reason, one should put a reverse proxy, e.g. Nginx, HAProxy, in front of the immich instance and add SSL to it.

To make the service persistent and start after reboot,

```bash
systemctl enable immich-ml && \
systemctl enable immich-web
```

#### Immich config

Because we are install Immich instance in a none docker environment, some DNS lookup will not work. For instance, we need to change the URL inside `Administration > Settings > Machine Learning Settings > URL` to `http://localhost:3003`, otherwise the web server cannot communicate with the ML backend.

Additionally, for LXC with CUDA or other GPU Transcoding support enabled, one needs to go to `Administration > Settings > Video Transcoding Settings > Hardware Acceleration > Acceleration API` and select your GPU Transcoding (e.g., `NVENC` - for CUDA) to explicitly use the GPU to do the transcoding.

## Update a source-built instance

Back up the database, persistent media, and `runtime.env` before upgrading, and
review the selected Immich release's upgrade requirements. Do not jump from an
old 1.x installation to 3.3 solely because the template now contains a 3.3 tag.
The older VectorChord migration notes above still apply where relevant.

If your existing checkout still points to the original `loeeeee` repository,
`git pull` will not fetch this fork's fixes. Check `git remote -v` and, if you
want to use this fork, run as `immich` from the checkout:

```bash
git remote set-url origin https://github.com/tautomer/immich-in-lxc.git
```

1. As **root**, install the Python prerequisites above and stop the services:

   ```bash
   systemctl stop immich-ml immich-web
   ```

2. As **`immich`**, keep a copy of the old application for rollback, update this
   repository, and edit `.env` to select the target `REPO_TAG`. Remove any old
   `ML_PYTHON` override pointing at Python 3.11. Keep `INSTALL_DIR` and
   `UPLOAD_DIR` unchanged.

   ```bash
   su - immich
   cd /home/immich/immich-in-lxc
   git pull --ff-only
   cp -a /home/immich/app /home/immich/app.backup-$(date +%Y%m%d-%H%M%S)
   nano .env
   ./install.sh
   ```

   `install.sh` recreates `app`, including the ML venv. It preserves media at
   `UPLOAD_DIR` and an existing `runtime.env`. Check the command's exit status;
   a failed install is not a completed update.

3. As **root**, review the installed systemd units and restart:

   ```bash
   systemctl daemon-reload
   systemctl reset-failed immich-ml immich-web
   systemctl start immich-ml immich-web
   systemctl status immich-ml immich-web
   ```

## Install or update prebuilt artifacts

This procedure is for an existing **Ubuntu 24.04 x86-64** installation at
**`/home/immich`**. The tarballs do not provision PostgreSQL/VectorChord, Redis,
Jellyfin FFmpeg, the `immich` user, GPU drivers, or runtime configuration. New
installations must set up those prerequisites first. The published ML artifact
uses OpenVINO, not the source installer's default CPU extra.

Download `dependencies.tar.gz` and `server.tar.gz` from the **same release** in
[this fork's releases](https://github.com/tautomer/immich-in-lxc/releases). Do not
mix libraries and server artifacts from different tags. These commands are for
**3.3+** releases built with the Python 3.12/uv workflow. **For a release earlier
than 3.3**, install Python 3.11 using the legacy instructions above and use
`python3.11-venv python3.11-dev` in place of the 3.12 packages below; the Stow,
extraction, and service steps are otherwise the same. Old artifacts remain
Python 3.11 artifacts even if their installation instructions are viewed later.

### Root: prepare runtime and dependency libraries

Install Python support, back up your database and configuration, and stop the
services. No PPA or global Python symlink replacement is necessary:

```bash
apt update
apt install -y python3.12-venv python3.12-dev stow
systemctl stop immich-ml immich-web
```

If replacing an existing Stow installation, first unstow its packages using
its **original** Stow directory. For the layout used below:

```bash
cd /opt/stow/deps/stow
for package in */; do stow -D -t /usr/local "${package%/}"; done
```

Skip that block if this is the first Stow installation. If the old libraries
were built manually, uninstall them from their original build directories
instead; do not blindly delete files from `/usr/local`.

With the new `dependencies.tar.gz` saved as `/root/dependencies.tar.gz`, keep a
backup of old packages and extract the new archive without stored ownership:

```bash
mkdir -p /opt/stow
cd /opt/stow
if [ -d deps ]; then mv deps "deps.old-$(date +%Y%m%d-%H%M%S)"; fi
tar -xzf /root/dependencies.tar.gz --no-same-owner
cd /opt/stow/deps/stow
for package in */; do stow -v -t /usr/local "${package%/}"; done
ldconfig
```

The archive layout is `deps/stow/<package>/` for libraries and `deps/json/` for
metadata. Keep both. The server archive already contains `app/build-lock.json`;
there is no need to copy metadata from a directory that has just been moved.

### Immich user: replace application and geodata

Save the matching `server.tar.gz` in `/home/immich`, then run as **`immich`**:

```bash
su - immich
cd /home/immich
stamp=$(date +%Y%m%d-%H%M%S)
if [ -d app ]; then mv app "app.old-$stamp"; fi
if [ -d geodata ]; then mv geodata "geodata.old-$stamp"; fi
tar -xzf server.tar.gz

# Set this to your existing persistent media directory; never delete it.
UPLOAD_DIR=/your/existing/upload/path
ln -sfnT "$UPLOAD_DIR" /home/immich/app/upload
ln -sfnT "$UPLOAD_DIR" /home/immich/app/machine-learning/upload

# Confirm that the packaged venv uses the required interpreter.
/home/immich/app/machine-learning/venv/bin/python --version
```

Do not relocate this venv, change its Python symlinks, or run Poetry over it.
Keep the existing `/home/immich/runtime.env`; it is not part of the server
archive. Ensure the services point to `app/start.sh` and
`app/machine-learning/start.sh`, then use the root restart commands in the
source-update section. Check both services and ML functionality before removing
old application, geodata, or dependency backups. Rollback may also require the
old dependency libraries and a database backup, not just the old `app` folder.

## Verification and troubleshooting

- A `Current Python version ... is not allowed` error means the ML interpreter
  does not satisfy the selected release. For 3.3, check `ML_PYTHON` and
  `/usr/bin/python3.12`; do not weaken `requires-python` or repoint an old venv.
- uv requires access to the lockfile's package hosts and GitHub's `ml-models`
  repository. A registry mirror does not proxy that Git dependency. Resolve
  network errors rather than marking an incomplete venv as successfully built.
- An OpenVINO executable-stack loading error requires `patchelf` and a fresh
  install with the current script. The workflow verifies ONNX Runtime imports
  before archiving. GPU passthrough/driver failures are separate concerns.
- Check `/var/log/immich/ml.log`, `/var/log/immich/web.log`, and
  `journalctl -u immich-ml -u immich-web` after deployment. A successful compile
  does not prove that a service, database migration, or GPU is working.

For contributors, local regression tests require only Python and Bash:

```bash
bash -n install.sh pre-install.sh python-version.sh
python3 -m unittest discover -s tests -v
```

The installer-test workflow additionally installs and imports the real CPU and
OpenVINO ML dependencies on Ubuntu 24.04. The full build verifies both Sharp and
ML runtime artifacts before publishing. After your branch is available on
GitHub, validate a full build without publishing a release:

```bash
gh workflow run auto_build_immich.yml --ref YOUR_BRANCH \
  -f version=v3.3.3 -f action=force-rebuild-all -f skip_release=true
```
