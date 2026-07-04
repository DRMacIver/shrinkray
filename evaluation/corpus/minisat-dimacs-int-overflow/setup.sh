#!/bin/bash
set -eu
DEST="$TOOLS_DIR/minisat-int-overflow"
BIN="$DEST/minisat_asan"
[ -x "$BIN" ] && exit 0
rm -rf "$DEST"
# Pin to a specific commit of the canonical MiniSat repo.
git clone https://github.com/niklasso/minisat "$DEST"
cd "$DEST"
git checkout 37dc6c67e2af26379d88ce349eb9c4c6160e8543

# MiniSat's decade-old source no longer compiles with a modern clang; apply the
# well-known minimal portability fixes (see minisat issues #13, #16, #1). None
# of these touch the parsing / clause-addition logic that the bug lives in.
# 1. Space before the inttypes PRI* macros (reserved-user-defined-literal).
grep -rl '"PRI' minisat | while read -r f; do
  sed -i.bak 's/"PRI/" PRI/g' "$f" && rm -f "$f.bak"
done
# 2. mkLit: default argument belongs on the definition, not the friend decl.
sed -i.bak 's/friend Lit mkLit(Var var, bool sign = false);/friend Lit mkLit(Var var, bool sign);/' minisat/core/SolverTypes.h
sed -i.bak 's/inline  Lit  mkLit     (Var var, bool sign) {/inline  Lit  mkLit     (Var var, bool sign = false) {/' minisat/core/SolverTypes.h
rm -f minisat/core/SolverTypes.h.bak
# 3. __APPLE__ memUsedPeak signature must match the header (takes a bool).
perl -0pi -e 's/(#elif defined\(__APPLE__\).*?double Minisat::memUsedPeak)\(\)/$1(bool)/s' minisat/utils/System.cc

# Build the simp binary with AddressSanitizer so the out-of-bounds read the bug
# causes is turned into a deterministic abort rather than silent memory corruption.
c++ -std=c++11 -I. -O1 -g -fsanitize=address -fno-omit-frame-pointer \
  minisat/simp/Main.cc minisat/core/Solver.cc minisat/simp/SimpSolver.cc \
  minisat/utils/Options.cc minisat/utils/System.cc \
  -lz -o "$BIN"
