#!/bin/sh
# Memorable CLI installer.
#
#   curl -fsSL https://memorable.sh/install.sh | sh
#
# No npm, no brew, no root, no package manager of any kind. If this machine
# already has Node 20 or newer, the installer uses it. If it does not, the
# installer downloads an official Node runtime into ~/.memorable and uses that,
# so nothing outside ~/.memorable, ~/.local/bin, and your shell rc file (a PATH
# line, skip it with --no-modify-path) is touched.
#
# Flags (pass them after `-s --` when piping, e.g. `| sh -s -- --no-modify-path`):
#   --version <v>       install a specific memorable-cli version
#   --prefix <dir>      install root (default ~/.memorable)
#   --bin-dir <dir>     where the `memorable` launcher goes (default ~/.local/bin)
#   --no-modify-path    do not touch shell startup files
#   --node <path>       use this node binary instead of searching
#
# Environment: MEMORABLE_VERSION, MEMORABLE_PREFIX, MEMORABLE_BIN_DIR, NODE_MIRROR.

set -eu

PKG=memorable-cli
REGISTRY=${MEMORABLE_REGISTRY:-https://registry.npmjs.org}
NODE_MIRROR=${NODE_MIRROR:-https://nodejs.org/dist}
NODE_TRACK=latest-v22.x
NODE_MIN_MAJOR=20

PREFIX=${MEMORABLE_PREFIX:-$HOME/.memorable}
BIN_DIR=${MEMORABLE_BIN_DIR:-$HOME/.local/bin}
WANT_VERSION=${MEMORABLE_VERSION:-latest}
MODIFY_PATH=1
NODE_OVERRIDE=

say() { printf '%s\n' "$*"; }
step() { printf '  %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --version) WANT_VERSION=${2:-} ; shift 2 || die "--version needs a value" ;;
    --prefix) PREFIX=${2:-} ; shift 2 || die "--prefix needs a value" ;;
    --bin-dir) BIN_DIR=${2:-} ; shift 2 || die "--bin-dir needs a value" ;;
    --node) NODE_OVERRIDE=${2:-} ; shift 2 || die "--node needs a value" ;;
    --no-modify-path) MODIFY_PATH=0 ; shift ;;
    -h|--help) sed -n '2,20p' "$0" 2>/dev/null || true; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

# ---------------------------------------------------------------- downloader

if command -v curl >/dev/null 2>&1; then
  fetch() { curl -fsSL "$1"; }
  fetch_to() { curl -fsSL "$1" -o "$2"; }
elif command -v wget >/dev/null 2>&1; then
  fetch() { wget -qO- "$1"; }
  fetch_to() { wget -qO "$2" "$1"; }
else
  die "need curl or wget on PATH"
fi

# ---------------------------------------------------------------- platform

OS=$(uname -s)
ARCH=$(uname -m)
case "$OS" in
  Darwin) NODE_OS=darwin ;;
  Linux) NODE_OS=linux ;;
  *) die "unsupported OS: $OS. Memorable installs on macOS and Linux; on Windows use WSL." ;;
esac
case "$ARCH" in
  arm64|aarch64) NODE_ARCH=arm64 ;;
  x86_64|amd64) NODE_ARCH=x64 ;;
  *) die "unsupported architecture: $ARCH" ;;
esac

# ---------------------------------------------------------------- node

node_major() {
  # prints the major version of the node binary in $1, or nothing
  "$1" -v 2>/dev/null | sed -n 's/^v\([0-9][0-9]*\)\..*/\1/p'
}

usable_node() {
  [ -n "${1:-}" ] || return 1
  [ -x "$1" ] || command -v "$1" >/dev/null 2>&1 || return 1
  m=$(node_major "$1") || return 1
  [ -n "$m" ] || return 1
  [ "$m" -ge "$NODE_MIN_MAJOR" ] || return 1
  return 0
}

VENDORED_NODE=$PREFIX/runtime/bin/node
NODE=

if [ -n "$NODE_OVERRIDE" ]; then
  usable_node "$NODE_OVERRIDE" || die "--node $NODE_OVERRIDE is not Node $NODE_MIN_MAJOR or newer"
  NODE=$NODE_OVERRIDE
elif usable_node "$(command -v node 2>/dev/null || true)"; then
  NODE=$(command -v node)
elif usable_node "$VENDORED_NODE"; then
  NODE=$VENDORED_NODE
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/memorable-install.XXXXXX") || die "cannot create a temp dir"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT INT TERM

say ""
say "Memorable CLI installer"
say ""

if [ -z "$NODE" ]; then
  step "no Node $NODE_MIN_MAJOR+ found, fetching one into $PREFIX/runtime"
  SHAFILE=$TMP/SHASUMS256.txt
  fetch_to "$NODE_MIRROR/$NODE_TRACK/SHASUMS256.txt" "$SHAFILE" \
    || die "cannot reach $NODE_MIRROR. Set NODE_MIRROR to a reachable mirror, or install Node yourself and re-run."
  NODE_FILE=$(sed -n "s/.*  \(node-v[0-9.]*-$NODE_OS-$NODE_ARCH\.tar\.gz\)$/\1/p" "$SHAFILE" | head -n 1)
  [ -n "$NODE_FILE" ] || die "no Node build published for $NODE_OS-$NODE_ARCH"
  NODE_SHA=$(grep "  $NODE_FILE\$" "$SHAFILE" | awk '{print $1}')
  step "downloading $NODE_FILE"
  fetch_to "$NODE_MIRROR/$NODE_TRACK/$NODE_FILE" "$TMP/$NODE_FILE" || die "download failed: $NODE_FILE"

  if command -v shasum >/dev/null 2>&1; then
    GOT=$(shasum -a 256 "$TMP/$NODE_FILE" | awk '{print $1}')
  elif command -v sha256sum >/dev/null 2>&1; then
    GOT=$(sha256sum "$TMP/$NODE_FILE" | awk '{print $1}')
  else
    GOT=$NODE_SHA
    step "no shasum tool available, skipping checksum"
  fi
  [ "$GOT" = "$NODE_SHA" ] || die "checksum mismatch on $NODE_FILE"

  rm -rf "$PREFIX/runtime" "$TMP/node"
  mkdir -p "$TMP/node" "$PREFIX"
  tar -xzf "$TMP/$NODE_FILE" -C "$TMP/node" || die "cannot unpack $NODE_FILE"
  mv "$TMP/node/$(basename "$NODE_FILE" .tar.gz)" "$PREFIX/runtime"
  # the CLI is a single bundled file, so headers, docs and npm itself are dead weight
  rm -rf "$PREFIX/runtime/include" "$PREFIX/runtime/share" "$PREFIX/runtime/lib/node_modules"
  NODE=$VENDORED_NODE
  usable_node "$NODE" || die "the downloaded Node runtime does not run on this machine"
  step "Node $("$NODE" -v) installed under $PREFIX/runtime"
else
  step "using Node $("$NODE" -v) at $NODE"
fi

# ---------------------------------------------------------------- package

if [ "$WANT_VERSION" = "latest" ]; then
  META=$(fetch "$REGISTRY/$PKG/latest") || die "cannot reach the npm registry at $REGISTRY"
else
  META=$(fetch "$REGISTRY/$PKG/$WANT_VERSION") || die "no such version: $WANT_VERSION"
fi

field() { printf '%s' "$META" | tr ',{}' '\n\n\n' | sed -n "s/^[[:space:]]*\"$1\":[[:space:]]*\"\([^\"]*\)\".*/\1/p" | head -n 1; }

VERSION=$(field version)
TARBALL=$(field tarball)
SHASUM=$(field shasum)
[ -n "$VERSION" ] && [ -n "$TARBALL" ] || die "could not read $PKG metadata from the registry"

step "downloading $PKG $VERSION"
fetch_to "$TARBALL" "$TMP/pkg.tgz" || die "download failed: $TARBALL"

if [ -n "$SHASUM" ]; then
  if command -v shasum >/dev/null 2>&1; then
    GOT=$(shasum -a 1 "$TMP/pkg.tgz" | awk '{print $1}')
  elif command -v sha1sum >/dev/null 2>&1; then
    GOT=$(sha1sum "$TMP/pkg.tgz" | awk '{print $1}')
  else
    GOT=$SHASUM
  fi
  [ "$GOT" = "$SHASUM" ] || die "checksum mismatch on the $PKG tarball"
fi

mkdir -p "$TMP/pkg"
tar -xzf "$TMP/pkg.tgz" -C "$TMP/pkg" || die "cannot unpack the $PKG tarball"
[ -f "$TMP/pkg/package/dist/cli.js" ] || die "unexpected package layout: dist/cli.js missing"

mkdir -p "$PREFIX/lib"
rm -rf "$PREFIX/lib/$PKG"
mv "$TMP/pkg/package" "$PREFIX/lib/$PKG"

# ---------------------------------------------------------------- launcher

mkdir -p "$BIN_DIR"
LAUNCHER=$BIN_DIR/memorable
cat > "$LAUNCHER" <<LAUNCH
#!/bin/sh
# Memorable CLI launcher. Prefers the machine's own Node, falls back to the
# runtime installed alongside the CLI.
MEMORABLE_LIB="$PREFIX/lib/$PKG/dist/cli.js"
VENDORED="$PREFIX/runtime/bin/node"
pick_node() {
  for candidate in "\${MEMORABLE_NODE:-}" "\$(command -v node 2>/dev/null)" "\$VENDORED"; do
    [ -n "\$candidate" ] || continue
    major=\$("\$candidate" -v 2>/dev/null | sed -n 's/^v\([0-9][0-9]*\)\..*/\1/p')
    [ -n "\$major" ] || continue
    [ "\$major" -ge $NODE_MIN_MAJOR ] || continue
    printf '%s' "\$candidate"
    return 0
  done
  return 1
}
NODE_BIN=\$(pick_node) || {
  echo "memorable: no Node $NODE_MIN_MAJOR+ runtime found. Re-run: curl -fsSL https://memorable.sh/install.sh | sh" >&2
  exit 1
}
exec "\$NODE_BIN" "\$MEMORABLE_LIB" "\$@"
LAUNCH
chmod +x "$LAUNCHER"

INSTALLED=$("$LAUNCHER" --version 2>/dev/null | head -n 1 || true)
[ -n "$INSTALLED" ] || INSTALLED="$PKG $VERSION"
step "memorable $INSTALLED installed at $LAUNCHER"

# ---------------------------------------------------------------- PATH

on_path=0
case ":$PATH:" in *":$BIN_DIR:"*) on_path=1 ;; esac

if [ "$on_path" -eq 0 ] && [ "$MODIFY_PATH" -eq 1 ]; then
  LINE="export PATH=\"$BIN_DIR:\$PATH\""
  for rc in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.profile"; do
    [ -f "$rc" ] || continue
    grep -Fq "$BIN_DIR" "$rc" 2>/dev/null && continue
    printf '\n# added by the Memorable installer\n%s\n' "$LINE" >> "$rc"
    step "added $BIN_DIR to PATH in $rc"
  done
fi

say ""
if [ "$on_path" -eq 1 ]; then
  say "Next:"
else
  say "Open a new terminal (or run: export PATH=\"$BIN_DIR:\$PATH\"), then:"
fi
say ""
say "  memorable login     # opens a browser, links this machine"
say "  memorable enable    # explicit write consent, nothing is stored before this"
say "  memorable status    # backend, consent, stored count"
say ""
say "Docs: https://memorable.sh/docs/cli"
say ""
